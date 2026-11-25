"""Especificações candidatas (família, ligação, preditor linear) do 2o turno.

Os notebooks `02-MLG` (comparação e Parte 11, validação cruzada) usam as
mesmas definições. As colunas entram por papéis genéricos (`MAPA_2026`), para
que o backtest troque os anos sem mudar o código. A padronização (z-score) e os
níveis das dummies são fixados no treino; o teste só é transformado.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy.build import build_design_matrices
from patsy.highlevel import dmatrix
from scipy import stats
from scipy.special import expit
from statsmodels.othermod.betareg import BetaModel

from src import modelos as M

L = sm.families.links
LIGACOES = {"logit": L.Logit, "probit": L.Probit, "cloglog": L.CLogLog,
            "loglog": L.LogLog, "cauchit": L.Cauchy}

# Regra: nenhuma covariável do 2º turno de 2026 entra no modelo, porque ela é desconhecida
# enquanto a apuração de uma UF não chegou. Só `y` e `n` vêm de 2t_2026. Comparecimento é o do
# 1º turno de 2026 e `delta_comp` (fora do bb_selecao) é o delta do 2º turno de 2022.
MAPA_2026 = {
    "y": "votos_pt_2t_2026", "n": "n_2t_2026", "lag": "logit_pt_2t_2022",
    "nota_dep": "nota_media_dep_2026", "pct_dep_esq": "pct_dep_esq_2026",
    "bloco_pref": "bloco_prefeito_2024",
    "comparecimento": "taxa_comparecimento_1t_2026",
    "delta_comp": "delta_comparecimento_2t_2022",
    "log_pop": "log_populacao_2022", "log_dens": "log_densidade_2022",
    "pct_mulheres": "pct_mulheres_2026", "pct_16_24": "pct_16_24_2026",
    "pct_60_mais": "pct_60_mais_2026", "pct_superior": "pct_superior_completo_2026",
    "pct_analfabeto": "pct_analfabeto_2026", "cobertura_pbf": "cobertura_pbf_2026",
    "pib_pc": "pib_per_capita_2023",  # o log é aplicado em montar_X
    "pct_vab_agro": "pct_vab_agro_2021", "pct_evangelicos": "pct_evangelicos_2022",
    "pct_pretos_pardos": "pct_pretos_pardos_2022", "pct_urbana": "pct_urbana_2022",
    "pct_fund_inc": "pct_ate_fund_incompleto_2026", "pct_sem_religiao": "pct_sem_religiao_2022",
    "pct_indigenas": "pct_indigenas_2022",
    "regiao": "regiao", "uf": "sigla_uf",
    "suplementar": "prefeito_suplementar_2024",
}
MAPA_2026_PRE = MAPA_2026  # apelido até o `02-MLG` e o `03-apuracao-parcial` deixarem de citá-lo
CONTINUAS = ["lag", "nota_dep", "pct_dep_esq", "comparecimento", "delta_comp",
             "log_pop", "log_dens", "pct_mulheres", "pct_16_24", "pct_60_mais",
             "pct_superior", "pct_analfabeto", "cobertura_pbf", "pib_pc",
             "pct_vab_agro", "pct_evangelicos", "pct_pretos_pardos", "pct_urbana",
             "pct_fund_inc", "pct_sem_religiao", "pct_indigenas"]
REGIOES = ["Centro-Oeste", "Nordeste", "Norte", "Sul"]
REGIAO_REF = "Sudeste"
UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
       "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]
UF_REF = "SP"
# níveis brutos do bloco do prefeito fora da referência (referência: direita)
BLOCOS = ["esquerda", "centro", "sem_prefeito"]
BLOCOS_PADRAO: tuple[tuple[str, ...], ...] = (("esquerda",), ("centro", "sem_prefeito"))
_COL_OFFSET = "lag_offset"


def _coluna(df, mapa, chave):
    v = df[mapa[chave]].to_numpy(float)
    return np.log(v) if chave == "pib_pc" else v


# --------------------------------------------------------------------------
# Padronização e matriz de desenho
# --------------------------------------------------------------------------
class Padronizador:
    """Média e desvio das contínuas (e base do spline do lag), ajustados no treino."""

    def __init__(self, mapa):
        self.mapa = mapa

    def fit(self, df):
        self.media, self.desvio = {}, {}
        for k in CONTINUAS:
            v = _coluna(df, self.mapa, k)
            self.media[k], self.desvio[k] = float(v.mean()), float(v.std(ddof=0)) or 1.0
        for k in self.mapa:  # defasagens espaciais (SLX): padronizadas só no treino
            if k.startswith("w_"):
                v = _coluna(df, self.mapa, k)
                self.media[k], self.desvio[k] = float(v.mean()), float(v.std(ddof=0)) or 1.0
        lag = _coluna(df, self.mapa, "lag")
        self.lag_min, self.lag_max = float(lag.min()), float(lag.max())
        # log n da dispersão (montar_Z): média e desvio só do treino
        ln = np.log(np.maximum(_coluna(df, self.mapa, "n"), 1.0))
        self.media_logn, self.desvio_logn = float(ln.mean()), float(ln.std(ddof=0)) or 1.0
        self._desenho_spline = dmatrix(
            "bs(x, df=4, degree=3, include_intercept=False) - 1", {"x": lag},
            return_type="dataframe").design_info
        return self

    def z(self, df, chave):
        return (_coluna(df, self.mapa, chave) - self.media[chave]) / self.desvio[chave]

    def spline_lag(self, df):
        """Base B-spline cúbica do lag; valores fora do suporte do treino são limitados."""
        x = np.clip(_coluna(df, self.mapa, "lag"), self.lag_min, self.lag_max)
        b = build_design_matrices([self._desenho_spline], {"x": x},
                                  return_type="dataframe")[0]
        b.columns = [f"lag_bs{i + 1}" for i in range(b.shape[1])]
        return b


DISPERSOES = ("const", "regiao", "regiao_logn")


@dataclass(frozen=True)
class Termos:
    """Preditor linear. `lag`: 'livre' | 'offset' (coef. fixo 1) | 'spline'."""
    vars: tuple = tuple(CONTINUAS)
    lag: str = "livre"
    usa_bloco: bool = True
    regiao_x_lag: bool = False
    usa_uf: bool = False          # UF no lugar da região
    uf_aleatoria: bool = False    # UF com todos os níveis (para o GLMM)
    # dispersão do beta-binomial: 'const' (theta único) | 'regiao' | 'regiao_logn'
    disp: str = "const"
    esf: int = 0                  # nº de autovetores espaciais (ev_001..) em X
    # SLX: chaves base cuja defasagem espacial W·x entra em X como `w_<chave>`
    slx: tuple = ()
    # níveis de região fundidos numa só dummy (ex.: (("Nordeste", "Norte"),)); um grupo com a
    # referência (Sudeste) funde seus níveis na referência (as dummies saem), porque a regra de
    # fusão da Parte 8.1 do `02-MLG` aceita fundir um nível na referência.
    regiao_fundir: tuple[tuple[str, ...], ...] = ()
    # grupos de níveis do bloco, uma dummy por grupo. Nível fora de todos os grupos, ou grupo
    # com "direita", vai para a referência. O padrão (esquerda; centro+sem_prefeito) é a
    # codificação histórica; a seleção por LRT parte dos níveis brutos e decide os grupos.
    bloco_grupos: tuple[tuple[str, ...], ...] = BLOCOS_PADRAO
    # prefeito eleito em eleição suplementar (binária); entra só se o LRT de adição der p <= 0,05.
    usa_suplementar: bool = False

    def __post_init__(self):
        if self.disp not in DISPERSOES:
            raise ValueError(f"disp deve ser um de {DISPERSOES}, veio {self.disp!r}")
        if self.esf < 0:
            raise ValueError("esf deve ser >= 0")
        if self.regiao_fundir:
            usados = [r for g in self.regiao_fundir for r in g]
            if any(r not in REGIOES + [REGIAO_REF] for r in usados) or len(set(usados)) != len(usados):
                raise ValueError(f"regiao_fundir só aceita níveis de {REGIOES + [REGIAO_REF]} sem repetição")
            if self.usa_uf or self.regiao_x_lag:
                raise ValueError("regiao_fundir não combina com usa_uf nem regiao_x_lag")
        usados_b = [b for g in self.bloco_grupos for b in g]
        if (any(b not in BLOCOS + ["direita"] for b in usados_b)
                or len(set(usados_b)) != len(usados_b)):
            raise ValueError(f"bloco_grupos só aceita níveis de {BLOCOS} sem repetição")
        # `vars` carrega também os w_<chave>; sem duplicar quando `replace` reaproveita
        extras = tuple(f"w_{k}" for k in self.slx if f"w_{k}" not in self.vars)
        if extras:
            object.__setattr__(self, "vars", tuple(self.vars) + extras)

    @property
    def usa_dispersao(self) -> bool:
        """Caminho com submodelo de dispersão (gamma) no beta-binomial."""
        return self.disp != "const" or self.esf > 0


def montar_X(df, mapa, padronizador, termos=Termos()) -> pd.DataFrame:
    """Matriz de desenho com const e dummies de colunas fixas (alinha folds de teste)."""
    idx = df.index
    cols = {"const": np.ones(len(df))}
    for k in termos.vars:
        if k == "lag" and termos.lag != "livre":
            continue
        cols[k] = padronizador.z(df, k)
    X = pd.DataFrame(cols, index=idx)
    if termos.lag == "spline":
        X = pd.concat([X, padronizador.spline_lag(df).set_axis(idx)], axis=1)
    if termos.usa_bloco:
        b = df[mapa["bloco_pref"]].astype(str)
        for grupo in termos.bloco_grupos:
            if "direita" in grupo:
                continue
            # valor fora dos quatro níveis (inclusive NaN) cai no grupo de sem_prefeito
            # (regra histórica: bloco_outro = tudo que não é esquerda nem direita)
            pertence = b.isin(grupo)
            if "sem_prefeito" in grupo:
                pertence = pertence | ~b.isin(["esquerda", "centro", "sem_prefeito", "direita"])
            nome = "+".join(grupo)
            if "centro" in grupo and "sem_prefeito" in grupo:
                nome = "+".join("outro" if g == "centro" else g for g in grupo
                                if g != "sem_prefeito")
            X["bloco_" + nome] = pertence.astype(float).to_numpy()
    if termos.usa_suplementar:
        X["prefeito_suplementar"] = (df[mapa["suplementar"]] == True).astype(float).to_numpy()
    if termos.usa_uf:
        niveis = UFS if termos.uf_aleatoria else [u for u in UFS if u != UF_REF]
        g, pref = df[mapa["uf"]].astype(str), "uf_"
    else:
        niveis, g, pref = REGIOES, df[mapa["regiao"]].astype(str), "reg_"
    dummies = pd.DataFrame({pref + n: (g == n).astype(float).to_numpy() for n in niveis},
                           index=idx)
    X = pd.concat([X, dummies], axis=1)
    for grupo in termos.regiao_fundir:
        cols_g = [f"reg_{r}" for r in grupo if r != REGIAO_REF]
        if REGIAO_REF not in grupo:
            X["reg_" + "+".join(grupo)] = X[cols_g].sum(axis=1)
        X = X.drop(columns=cols_g)
    if termos.regiao_x_lag:
        z = padronizador.z(df, "lag")
        for n in REGIOES:
            X[f"reg_{n}:lag"] = dummies[pref + n].to_numpy() * z
    if termos.esf:
        # autovetores de Moran pré-calculados (variância 1): entram brutos, sem padronizar
        ev = [f"ev_{i:03d}" for i in range(1, termos.esf + 1)]
        faltam = [c for c in ev if c not in df.columns]
        if faltam:
            raise KeyError(f"autovetores espaciais ausentes em df: {faltam[:3]}...")
        X = pd.concat([X, df[ev].astype(float)], axis=1)
    if termos.lag == "offset":
        X[_COL_OFFSET] = _coluna(df, mapa, "lag")  # bruto, última coluna
    return X


def montar_Z(df, mapa, padronizador, termos=Termos()) -> pd.DataFrame:
    """Matriz de dispersão (log theta = Z gamma): const, + dummies de região, + log n padronizado."""
    idx = df.index
    Z = pd.DataFrame({"const": np.ones(len(df))}, index=idx)
    if termos.disp in ("regiao", "regiao_logn"):
        g = df[mapa["regiao"]].astype(str)
        for r in REGIOES:
            Z[f"reg_{r}"] = (g == r).astype(float).to_numpy()
    if termos.disp == "regiao_logn":
        ln = np.log(np.maximum(_coluna(df, mapa, "n"), 1.0))
        Z["log_n"] = (ln - padronizador.media_logn) / padronizador.desvio_logn
    return Z


# --------------------------------------------------------------------------
# Modelos auxiliares
# --------------------------------------------------------------------------
class BetaBinomialOffset(M.BetaBinomial):
    """Beta-binomial cuja última coluna de exog é um offset (coeficiente fixo em 1)."""

    def __init__(self, endog, exog, n, link=None, **kwds):
        super().__init__(endog, exog, n, link, **kwds)
        self.nparams = self.exog.shape[1]
        nomes = self.exog_names
        assert nomes is not None
        del nomes[-2]

    def _pi(self, params, exog=None):
        exog = self.exog if exog is None else np.asarray(exog)
        eta = exog[:, :-1] @ params[:-1] + exog[:, -1]
        return M._pi_de_eta(eta, self.link)

    def _start(self):
        y, n, E = self.endog, self.n, self.exog
        g = sm.GLM(np.column_stack([y, n - y]), E[:, :-1], offset=E[:, -1],
                   family=sm.families.Binomial(link=self.link)).fit()
        pi = np.clip(g.fittedvalues, 1e-12, 1 - 1e-12)
        rho = np.sum((y - n * pi) ** 2 / (n * pi * (1 - pi)) - 1) / np.sum(n - 1)
        return np.append(np.asarray(g.params), np.log(1 / float(np.clip(rho, 1e-8, 0.5)) - 1))


@dataclass
class ModeloL1:
    colunas: list
    coef: np.ndarray
    intercepto: float
    C: float
    C_min: float = np.nan

    def predict(self, X):
        return expit(np.asarray(X[self.colunas], float) @ self.coef + self.intercepto)

    @property
    def mantidas(self):
        return [c for c, b in zip(self.colunas, self.coef) if abs(b) > 1e-10]


@dataclass
class ModeloGLMM:
    colunas: list
    beta: np.ndarray
    sigma2: float
    phi: float
    cov: np.ndarray

    def predict(self, X):
        return expit(np.asarray(X[self.colunas], float) @ self.beta)


def _ajustar_l1(X, y, n, graus=None):
    from sklearn.linear_model import LogisticRegression
    cols = [c for c in X.columns if c != "const"]
    A, p = X[cols].to_numpy(float), y / n
    A2 = np.vstack([A, A])
    alvo = np.r_[np.ones(len(A)), np.zeros(len(A))]

    def ajusta(idx, C):
        i2 = np.r_[idx, idx + len(A)]
        w = np.r_[p[idx], 1 - p[idx]]
        return LogisticRegression(l1_ratio=1, solver="liblinear", C=C, tol=1e-6,
                                  max_iter=500, random_state=0).fit(A2[i2], alvo[i2], sample_weight=w)

    grade, folds = np.logspace(-3, 2, 11), M.folds_kfold(len(A), 5, seed=0)
    perda = np.zeros((len(grade), len(folds)))
    for i, C in enumerate(grade):
        for j, (tr, te) in enumerate(folds):
            q = np.clip(ajusta(tr, C).predict_proba(A[te])[:, 1], 1e-9, 1 - 1e-9)
            perda[i, j] = -np.mean(p[te] * np.log(q) + (1 - p[te]) * np.log(1 - q))
    media = perda.mean(axis=1)
    i_min = int(np.argmin(media))
    ep = perda[i_min].std(ddof=1) / np.sqrt(perda.shape[1])
    i_1se = int(np.flatnonzero(media <= media[i_min] + ep)[0])  # menor C (mais penalização) dentro de 1 EP
    C = float(grade[i_1se])
    m = ajusta(np.arange(len(A)), C)
    return ModeloL1(cols, m.coef_.ravel().copy(), float(m.intercept_[0]), C, float(grade[i_min])), \
        pd.DataFrame({"C": grade, "perda_media": media})


def _ajustar_glmm(X, y, n, nomes_uf, iter_ext=25):
    """Intercepto aleatório por UF por PQL (IRLS penalizado) com variância por EM.

    Os pesos são divididos pelo phi de Pearson (sobredispersão); equivale a um
    GLMM quase-binomial. BinomialBayesMixedGLM exige resposta 0/1 e não serve
    a dados agrupados.
    """
    A = X.to_numpy(float)
    ju = np.array([c in nomes_uf for c in X.columns])
    beta, sigma2, phi = np.zeros(A.shape[1]), 0.05, 1.0
    p0 = np.clip(y.sum() / n.sum(), 1e-6, 1 - 1e-6)
    beta[0] = np.log(p0 / (1 - p0))
    for _ in range(iter_ext):
        P = np.diag(np.where(ju, 1.0 / sigma2, 0.0))
        for _ in range(30):
            pi = np.clip(expit(A @ beta), 1e-9, 1 - 1e-9)
            w = n * pi * (1 - pi) / phi
            z = A @ beta + (y / n - pi) / (pi * (1 - pi))
            H = A.T @ (w[:, None] * A) + P
            novo = np.linalg.solve(H, A.T @ (w * z))
            ok = np.max(np.abs(novo - beta)) < 1e-8
            beta = novo
            if ok:
                break
        pi = np.clip(expit(A @ beta), 1e-9, 1 - 1e-9)
        phi = float(np.sum((y - n * pi) ** 2 / (n * pi * (1 - pi))) / (len(y) - (A.shape[1] - ju.sum())))
        cov = np.linalg.inv(H)
        u, vu = beta[ju], np.diag(cov)[ju]
        sigma2 = float(np.mean(u ** 2 + vu))
    return ModeloGLMM(list(X.columns), beta, sigma2, phi, cov)


# --------------------------------------------------------------------------
# Candidatos
# --------------------------------------------------------------------------
@dataclass
class Ajuste:
    """Modelo ajustado + padronizador + metadados para tabela e coeficientes."""
    modelo: Any           # resultado do statsmodels ou ModeloL1/ModeloGLMM
    pad: Padronizador
    nomes: list           # nomes dos parâmetros de `modelo.params` (quando houver)
    k: int                # parâmetros livres (para AIC/BIC)
    loglik: float         # NaN se a resposta/verossimilhança não é binomial
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Candidato:
    nome: str
    familia: str   # binomial|quasi|betabin|betareg|ols|wls|l1|glmm
    ligacao: str
    termos: Termos = Termos()

    @property
    def usa_uf(self) -> bool:
        return self.termos.usa_uf

    def ajustar(self, df, mapa=MAPA_2026) -> Ajuste:
        pad = Padronizador(mapa).fit(df)
        X = montar_X(df, mapa, pad, self.termos)
        y, n = df[mapa["y"]].to_numpy(float), df[mapa["n"]].to_numpy(float)
        off = self.termos.lag == "offset"
        link = LIGACOES.get(self.ligacao)
        f = self.familia
        if f in ("binomial", "quasi"):
            if link is None:
                raise ValueError(f"ligação desconhecida: {self.ligacao}")
            Xd = X.drop(columns=_COL_OFFSET) if off else X
            mod = sm.GLM(np.column_stack([y, n - y]), Xd, family=sm.families.Binomial(link=link()),
                         offset=X[_COL_OFFSET].to_numpy() if off else None)
            res = mod.fit()
            pi = np.asarray(res.fittedvalues)
            phi = 1.0
            if f == "quasi":  # phi de Pearson; scale='X2' do statsmodels sai errado com y de duas colunas
                phi = float(np.sum((y - n * pi) ** 2 / (n * pi * (1 - pi))) / (len(y) - Xd.shape[1]))
                res = mod.fit(scale=phi)
            ll = float(stats.binom.logpmf(y, n, pi).sum())
            return Ajuste(res, pad, list(Xd.columns), Xd.shape[1] + (f == "quasi"), ll,
                          {"phi": phi})
        if f == "betabin" and self.termos.usa_dispersao:
            return self._ajustar_betabin_disp(df, mapa, pad, X, y, n, link, off)
        if f == "betabin":
            cls = BetaBinomialOffset if off else M.BetaBinomial
            mod = cls(y, X.to_numpy(float), n, link=link)
            res = mod.fit()
            # o BFGS padrão para cedo (gtol frouxo): polir com L-BFGS e ficar com o melhor
            pol = mod.fit(start_params=res.params, method="lbfgs", maxiter=3000)
            res = pol if pol.llf >= res.llf else res
            nomes = [c for c in X.columns if c != _COL_OFFSET] + ["log_theta"]
            return Ajuste(res, pad, nomes, len(nomes), float(res.llf),
                          {"theta": M.theta_de(res), "rho": M.icc(res)})
        if f == "betareg":
            if link is None:
                raise ValueError(f"ligação desconhecida: {self.ligacao}")
            p = np.clip(y / n, 1e-6, 1 - 1e-6)
            res = BetaModel(p, X, link=link()).fit(disp=False)
            return Ajuste(res, pad, list(res.params.index), len(res.params), np.nan)
        if f in ("ols", "wls"):
            p = (y + 0.5) / (n + 1.0)
            el = np.log(p / (1 - p))
            res = (sm.OLS(el, X) if f == "ols" else sm.WLS(el, X, weights=n * p * (1 - p))).fit()
            return Ajuste(res, pad, list(X.columns), X.shape[1] + 1, np.nan)
        if f == "l1":
            m, perda = _ajustar_l1(X, y, n)
            return Ajuste(m, pad, m.colunas, len(m.mantidas) + 1,
                          float(stats.binom.logpmf(y, n, np.clip(m.predict(X), 1e-9, 1 - 1e-9)).sum()),
                          {"C": m.C, "C_min": m.C_min, "mantidas": m.mantidas, "perda_cv": perda})
        if f == "glmm":
            m = _ajustar_glmm(X, y, n, {c for c in X.columns if c.startswith("uf_")})
            return Ajuste(m, pad, m.colunas, X.shape[1] - len(UFS) + 1, np.nan,
                          {"sigma_uf": float(np.sqrt(m.sigma2)), "phi": m.phi})
        raise ValueError(f)

    def _ajustar_betabin_disp(self, df, mapa, pad, X, y, n, link, off):
        if off:
            raise NotImplementedError("dispersão não é suportada com lag em offset")
        Z = montar_Z(df, mapa, pad, self.termos)
        mod = M.BetaBinomial(y, X.to_numpy(float), n, link=link, exog_disp=Z.to_numpy(float))
        res = mod.fit()
        pol = mod.fit(start_params=res.params, method="lbfgs", maxiter=3000)
        res = pol if pol.llf >= res.llf else res
        # Z só com a constante: o gamma é o próprio log_theta
        nz = ["log_theta"] if list(Z.columns) == ["const"] else [f"disp_{c}" for c in Z.columns]
        nomes = list(X.columns) + nz
        th = M.theta_de(res)
        reg = df[mapa["regiao"]].astype(str).to_numpy()
        extra = {"theta": float(np.median(th)), "rho": float(np.median(M.icc(res))),
                 "theta_regiao": {r: float(np.median(th[reg == r])) for r in np.unique(reg)},
                 "termos": self.termos, "n_beta": X.shape[1]}
        return Ajuste(res, pad, nomes, len(nomes), float(res.llf), extra)

    def prever(self, ajuste: Ajuste, df, mapa=MAPA_2026):
        X = montar_X(df, mapa, ajuste.pad, self.termos)
        m, f = ajuste.modelo, self.familia
        if f in ("l1", "glmm"):
            return np.asarray(m.predict(X))
        if self.termos.lag == "offset":
            if f in ("binomial", "quasi"):
                return np.asarray(m.predict(X.drop(columns=_COL_OFFSET),
                                            offset=X[_COL_OFFSET].to_numpy()))
            return np.asarray(m.model.predict(m.params, X.to_numpy(float)))
        if f == "betabin":
            return np.asarray(m.model.predict(m.params, X.to_numpy(float)))
        if f in ("ols", "wls"):
            return expit(np.asarray(m.predict(X)))
        return np.asarray(m.predict(X))


def prever_theta(ajuste: Ajuste, df, mapa=MAPA_2026) -> np.ndarray:
    """theta_i das linhas de `df` (array constante quando o ajuste tem theta único)."""
    ex = ajuste.extra
    if "termos" in ex:
        Z = montar_Z(df, mapa, ajuste.pad, ex["termos"]).to_numpy(float)
        return np.exp(Z @ np.asarray(ajuste.modelo.params)[ex["n_beta"]:])
    if "theta" not in ex:
        raise ValueError("ajuste sem theta (só o beta-binomial tem precisão)")
    return np.full(len(df), float(ex["theta"]))


def _t(**kw):
    return Termos(**kw)


def _montar_candidatos() -> dict:
    c = {}
    for lig in LIGACOES:
        c[f"bin_{lig}"] = Candidato(f"bin_{lig}", "binomial", lig)
    c["quasi_logit"] = Candidato("quasi_logit", "quasi", "logit")
    c["bb_logit"] = Candidato("bb_logit", "betabin", "logit")
    c["betareg"] = Candidato("betareg", "betareg", "logit")
    c["ols_elogit"] = Candidato("ols_elogit", "ols", "identidade")
    c["wls_elogit"] = Candidato("wls_elogit", "wls", "identidade")
    variantes = {
        "offset": _t(lag="offset"), "spline": _t(lag="spline"),
        "reglag": _t(regiao_x_lag=True), "uf": _t(usa_uf=True),
    }
    for sufixo, termos in variantes.items():
        c[f"bb_{sufixo}"] = Candidato(f"bb_{sufixo}", "betabin", "logit", termos)
        c[f"quasi_{sufixo}"] = Candidato(f"quasi_{sufixo}", "quasi", "logit", termos)
    c["glmm_uf"] = Candidato("glmm_uf", "glmm", "logit", _t(usa_uf=True, uf_aleatoria=True))
    c["l1_logit"] = Candidato("l1_logit", "l1", "logit")
    return c


CANDIDATOS = _montar_candidatos()


def registrar_selecao(variaveis: Iterable[str],
                      regiao_fundir: tuple[tuple[str, ...], ...] = (),
                      bloco_grupos: tuple[tuple[str, ...], ...] = BLOCOS_PADRAO) -> Candidato:
    """Adiciona 'bb_selecao' (beta-binomial só com as contínuas selecionadas).

    `regiao_fundir` leva ao candidato os níveis de região que a Parte 8 do `02-MLG`
    decidiu fundir (teste de igualdade de níveis) e `bloco_grupos` os grupos de bloco do
    prefeito que a mesma Parte decidiu manter separados (o resto vira a referência).
    """
    vs = tuple(v for v in CONTINUAS if v in set(variaveis) | {"lag"})
    cand = Candidato("bb_selecao", "betabin", "logit",
                     Termos(vars=vs, regiao_fundir=regiao_fundir, bloco_grupos=bloco_grupos))
    CANDIDATOS["bb_selecao"] = cand
    return cand


EXTENSOES: dict = {}  # variantes do bb_selecao; fora de CANDIDATOS (o Parte 11 do `02-MLG` itera CANDIDATOS)


def registrar_extensao(nome, **mudancas_em_termos) -> Candidato:
    """Registra em EXTENSOES o bb_selecao com `Termos` alterados (disp, esf, ...)."""
    if "bb_selecao" not in CANDIDATOS:
        raise KeyError("bb_selecao não registrado: chame registrar_selecao antes")
    base = CANDIDATOS["bb_selecao"]
    cand = Candidato(nome, base.familia, base.ligacao, replace(base.termos, **mudancas_em_termos))
    EXTENSOES[nome] = cand
    return cand


# --------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------
def tabela_coeficientes(ajuste: Ajuste) -> pd.DataFrame:
    m = ajuste.modelo
    if not hasattr(m, "bse"):
        return pd.DataFrame({"coef": m.coef if hasattr(m, "coef") else m.beta},
                            index=ajuste.nomes)
    est, ep = np.asarray(m.params, float), np.asarray(m.bse, float)
    z = est / ep
    return pd.DataFrame({"coef": est, "ep": ep, "z": z, "p": 2 * stats.norm.sf(np.abs(z))},
                        index=ajuste.nomes[: len(est)])


def resumo_ajuste(cand: Candidato, ajuste: Ajuste, df, mapa=MAPA_2026) -> dict:
    """Linha da tabela de comparação (dentro da amostra)."""
    y, n = df[mapa["y"]].to_numpy(float), df[mapa["n"]].to_numpy(float)
    pi = np.clip(cand.prever(ajuste, df, mapa), 1e-12, 1 - 1e-12)
    N, k = len(y), ajuste.k
    met = M.metricas(y, n, pi)
    phi = float(np.sum((y - n * pi) ** 2 / (n * pi * (1 - pi))) / (N - k))
    ll = ajuste.loglik
    quasi = cand.familia in ("quasi", "glmm")
    ok = np.isfinite(ll) and not quasi
    ph = ajuste.extra.get("phi", np.nan)
    return {
        "modelo": cand.nome, "familia": cand.familia, "ligacao": cand.ligacao, "k": k,
        "loglik": ll if ok else np.nan,
        "aic": -2 * ll + 2 * k if ok else np.nan,
        "bic": -2 * ll + k * np.log(N) if ok else np.nan,
        "qaic": -2 * ll / ph + 2 * k if (cand.familia == "quasi" and np.isfinite(ll)) else np.nan,
        "deviance": met["deviance"], "deviance_gl": met["deviance"] / (N - k),
        "phi_pearson": phi, "mae_pond_pp": met["mae_pond_pp"], "rmse_pond_pp": met["rmse_pond_pp"],
    }


def tabela_comparacao(resultados) -> pd.DataFrame:
    """Reúne linhas de `resumo_ajuste` (lista ou dict de dicts); ordena por AIC, NaN por último."""
    linhas = list(resultados.values()) if isinstance(resultados, dict) else list(resultados)
    return pd.DataFrame(linhas).sort_values("aic", na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------
# Validação cruzada em paralelo (Parte 11 do `02-MLG`). Os dados ficam em variável do
# módulo para que processos criados por fork os herdem sem copiar por tarefa.
# --------------------------------------------------------------------------
_DADOS_CV: dict = {}


def n_estimado(df: pd.DataFrame, ano_ref: int = 2022, ano_alvo: int = 2026) -> np.ndarray:
    """n̂ = eleitorado do ano-alvo x (n_2t / aptos_2t do ano de referência)."""
    return (df[f"eleitorado_{ano_alvo}"] * df[f"n_2t_{ano_ref}"] / df[f"aptos_2t_{ano_ref}"]).to_numpy(float)


def n_loso(df: pd.DataFrame) -> np.ndarray:
    """n̂ do 2º turno de 2026 de uma UF ainda não apurada (regra do backtest)."""
    return n_estimado(df, 2022, 2026)


def validar_mapa_sem_2t_2026(mapa: dict[str, str]) -> None:
    """Falha se algum papel além de y/n apontar para coluna do 2º turno de 2026."""
    vazados = {k: v for k, v in mapa.items() if "2t_2026" in v and k not in ("y", "n")}
    if vazados:
        raise ValueError(f"covariáveis do 2º turno de 2026 no mapa: {vazados}")


def definir_dados_cv(df: pd.DataFrame, mapa: dict[str, str] = MAPA_2026) -> None:
    validar_mapa_sem_2t_2026(mapa)
    _DADOS_CV.update(df=df, mapa=mapa, n_hat=n_loso(df))


def prever_fold(nome, treino, teste):
    """pi fora da amostra do candidato `nome` para as linhas `teste` (ajuste só em `treino`)."""
    df, mapa = _DADOS_CV["df"], _DADOS_CV["mapa"]
    c = {**CANDIDATOS, **EXTENSOES}[nome]
    a = c.ajustar(df.iloc[treino], mapa)
    # A UF de teste não foi apurada: o peso n é o estimado, não o real.
    te = df.iloc[teste].copy()
    te[mapa["n"]] = _DADOS_CV["n_hat"][teste]
    return np.asarray(c.prever(a, te, mapa), dtype=float)


def parametros_ajuste(nome) -> int:
    """Número de parâmetros livres (k) do candidato ajustado a todos os dados."""
    df, mapa = _DADOS_CV["df"], _DADOS_CV["mapa"]
    return int({**CANDIDATOS, **EXTENSOES}[nome].ajustar(df, mapa).k)


def ajustar_candidato(nome: str, df: pd.DataFrame, mapa: dict[str, str],
                      pasta_tabelas: Path | None = None) -> tuple[Ajuste, dict]:
    """Ajusta `CANDIDATOS[nome]` em todos os dados e devolve (ajuste, linha-resumo).

    Com `pasta_tabelas`, grava `coef_<nome>.csv` (coeficientes do ajuste).
    """
    c = CANDIDATOS[nome]
    a = c.ajustar(df, mapa)
    linha = resumo_ajuste(c, a, df, mapa)
    if pasta_tabelas is not None:
        tabela_coeficientes(a).to_csv(Path(pasta_tabelas) / f"coef_{nome}.csv", index_label="termo")
    return a, linha


def montar_diagnostico(cand: Candidato, ajuste: Ajuste | None, df: pd.DataFrame,
                       mapa: dict[str, str], seg_ajuste: float | None = None) -> dict:
    """Tudo que os testes de pressupostos pedem de um modelo ajustado em todos os dados.

    Com `ajuste` dado, o modelo não é reajustado (o chamador já o tem) e `seg_ajuste`
    é o tempo daquele ajuste; sem `ajuste`, ajusta aqui e mede o tempo.
    """
    t0 = time.time()
    a = ajuste if ajuste is not None else cand.ajustar(df, mapa)
    seg = seg_ajuste if seg_ajuste is not None else time.time() - t0
    return {"cand": cand, "aj": a, "res": a.modelo, "df": df, "mapa": mapa,
            "pi": np.asarray(cand.prever(a, df, mapa), float), "theta": prever_theta(a, df, mapa),
            "seg_ajuste": seg}


def descrever_diagnostico(nome: str, m: dict, y: np.ndarray, n: np.ndarray) -> str:
    """Uma linha: nº de parâmetros, tempo do ajuste, phi de Pearson e theta mediano."""
    pi, k = m["pi"], len(m["res"].params)
    phi = np.sum((y - n * pi) ** 2 / (n * pi * (1 - pi))) / (len(y) - k)
    return (f"{nome}: {k} parâmetros | {m['seg_ajuste']:.1f} s por ajuste | phi de Pearson = "
            f"{phi:.1f} | theta mediano = {np.median(m['theta']):.0f}")
