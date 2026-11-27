"""Modelos da família binomial para o 2o turno por município.

Y_i = votos do candidato A, n_i = votos válidos (A + B), pi_i = P(voto em A).
Os modelos aqui complementam o statsmodels: beta-binomial, agregação por
votos, validação cruzada, métricas, resíduos quantílicos e intervalos.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy import stats
from scipy.special import betaln, digamma, expit, gammaln, polygamma, xlogy
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults

_EPS = 1e-12
# Argumentos posicionais de `_otimizar` depois de start_params: method, maxiter, full_output, disp, callback, retall.
type _ArgsOtimizar = tuple[str, int, int, bool, Callable[[np.ndarray], None] | None, int]
# Teto da precisão theta de um grupo de dispersão que corre para o limite binomial. O fator de
# variância da beta-binomial é 1 + (n-1)/(1+theta); com o maior n da base (7e6 votos válidos de
# um município em 2026) e theta = 1e9 ele fica a 0,7 % do binomial, dentro da tolerância de 1 %
# adotada. Acima disso a verossimilhança é plana (theta é praticamente irrelevante) e o
# otimizador fica sem direção de subida, sem convergir.
THETA_TETO = 1e9
# Um grupo só é candidato ao teto quando o theta de todos os seus municípios passa deste piso
# (correlação intraclasse rho = 1/(1+theta) <= 1e-5): abaixo disso theta ainda é identificado
# pelos dados e o ajuste não é tocado. Os ajustes regionais do 02-MLG ficam em theta <= 3e4.
THETA_CANDIDATO_TETO = 1e5
AVISO_NAO_PSD = "Covariância não PSD"


def cov_psd(cov: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Simetriza `cov` e zera autovalores negativos (a covariância de Conley não garante PSD).

    Negativo dentro de tol x maior autovalor é ruído numérico e é cortado em silêncio; abaixo
    disso avisa com a razão mín/máx. Só para amostrar: os erros-padrão reportados não mudam.
    """
    c = np.asarray(cov, float)
    c = (c + c.T) / 2
    val, vet = np.linalg.eigh(c)
    if val[0] >= 0:
        return c
    if val[0] < -tol * val[-1]:
        warnings.warn(
            f"{AVISO_NAO_PSD} (autovalor mín/máx = {val[0] / val[-1]:.3g}); "
            "autovalores negativos cortados em 0 para amostrar",
            stacklevel=2,
        )
    return (vet * np.clip(val, 0.0, None)) @ vet.T


def _link(link):
    """Aceita classe ou instância de link do statsmodels (padrão: logit)."""
    if link is None:
        return sm.families.links.Logit()
    return link() if isinstance(link, type) else link


def _pi_de_eta(eta, link):
    # Limita pi para evitar log(0) quando eta é extremo.
    return np.clip(np.asarray(link.inverse(eta), dtype=float), _EPS, 1 - _EPS)


# --------------------------------------------------------------------------
# Beta-binomial
# --------------------------------------------------------------------------
class BetaBinomial(GenericLikelihoodModel):
    """Beta-binomial com média pi = g^-1(x'beta) e precisão theta = exp(log_theta).

    a = pi*theta, b = (1-pi)*theta; Var(Y) = n*pi*(1-pi)*(1 + (n-1)*rho),
    rho = 1/(1+theta) (correlação intraclasse). Os parâmetros são
    (beta..., log_theta); log_theta garante theta > 0.
    """

    def __init__(self, endog, exog, n, link=None, exog_disp=None, **kwds):
        super().__init__(endog, exog, **kwds)
        self.n = np.asarray(n, dtype=float)
        self.link = _link(link)
        self.exog_disp = None
        self._p = None  # None: caminho antigo (theta constante, log_theta por último)
        if exog_disp is None:
            nomes_exog = self.exog_names
            assert nomes_exog is not None
            nomes_exog.append("log_theta")
            self.k_extra = 1
            self.nparams = self.exog.shape[1] + 1
        else:
            # Submodelo de dispersão (double GLM, Smyth 1989): log theta_i = z_i'gamma.
            nomes = [f"disp_{c}" for c in getattr(exog_disp, "columns", [])]
            Z = np.asarray(exog_disp, dtype=float)
            if Z.ndim == 1:
                Z = Z[:, None]
            if len(Z) != len(self.endog):
                raise ValueError("exog_disp deve ter uma linha por observação")
            self.exog_disp = Z
            self._p = self.exog.shape[1]
            nomes_exog = self.exog_names
            assert nomes_exog is not None
            nomes_exog.extend(nomes or [f"disp_{j}" for j in range(Z.shape[1])])
            self.k_extra = Z.shape[1]
            self.nparams = self._p + Z.shape[1]
        self._log_binom = gammaln(self.n + 1) - gammaln(self.endog + 1) - gammaln(self.n - self.endog + 1)
        # Grupos no teto: dummies de dispersão fixadas (`congelados`, índices em params) e as
        # linhas desses grupos, que passam a ter theta = THETA_TETO sem depender de params.
        self.congelados: list[int] = []
        self._linhas_teto: np.ndarray | None = None
        self._teto_avaliado = False

    def _pi(self, params, exog=None):
        exog = self.exog if exog is None else exog
        p = self._p if self._p is not None else len(params) - 1
        return _pi_de_eta(np.asarray(exog) @ np.asarray(params)[:p], self.link)

    def _theta(self, params, exog_disp=None):
        if self.exog_disp is None:
            return np.exp(params[-1])
        Z = self.exog_disp if exog_disp is None else np.asarray(exog_disp, float)
        theta = np.exp(Z @ np.asarray(params)[self._p :])
        if exog_disp is None and self._linhas_teto is not None:
            theta = np.where(self._linhas_teto, THETA_TETO, theta)
        return theta

    def loglikeobs(self, params):
        pi = self._pi(params)
        theta = self._theta(params)
        a, b = pi * theta, (1 - pi) * theta
        y, n = self.endog, self.n
        return self._log_binom + betaln(y + a, n - y + b) - betaln(a, b)

    def loglike(self, params):
        return self.loglikeobs(params).sum()

    def score_obs(self, params, **kwds):
        """Gradiente analítico (digamma) só no caminho com dispersão; o antigo é numérico."""
        if self.exog_disp is None:
            return super().score_obs(params, **kwds)
        params = np.asarray(params, float)
        eta = self.exog @ params[: self._p]
        pi = _pi_de_eta(eta, self.link)
        theta = self._theta(params)
        a, b = pi * theta, (1 - pi) * theta
        y, n = self.endog, self.n
        comum = digamma(theta) - digamma(n + theta)
        da = digamma(y + a) - digamma(a) + comum
        db = digamma(n - y + b) - digamma(b) + comum
        dpi = np.asarray(self.link.inverse_deriv(eta), float)
        g_eta = theta * dpi * (da - db)
        g_lt = theta * (pi * da + (1 - pi) * db)
        if self._linhas_teto is not None:
            g_lt = np.where(self._linhas_teto, 0.0, g_lt)  # theta fixo no teto: não depende de gamma
        return np.hstack([self.exog * g_eta[:, None], self.exog_disp * g_lt[:, None]])

    def score(self, params):
        if self.exog_disp is None:
            return super().score(params)
        return self.score_obs(params).sum(0)

    def hessian(self, params: np.ndarray) -> np.ndarray:
        """Hessiana analítica (trigamma) no caminho com dispersão; o antigo segue numérico.

        A numérica do statsmodels (diferenças segundas da log-verossimilhança somada) erra ~1e-3
        do maior elemento quando n chega a 1e6. Com autovalores de 0,04 a 3e6 (corte 2026/5 da
        apuração) isso a torna indefinida: o statsmodels zera cov_params e o pão de Conley sai errado.
        """
        if self.exog_disp is None:
            return super().hessian(params)
        params = np.asarray(params, float)
        eta = self.exog @ params[: self._p]
        pi = _pi_de_eta(eta, self.link)
        theta = self._theta(params, exog_disp=None)
        a, b = pi * theta, (1 - pi) * theta
        y, n = self.endog, self.n
        comum = digamma(theta) - digamma(n + theta)
        da = digamma(y + a) - digamma(a) + comum
        db = digamma(n - y + b) - digamma(b) + comum
        ab = polygamma(1, theta) - polygamma(1, n + theta)  # ∂²ℓ/∂a∂b
        aa = polygamma(1, y + a) - polygamma(1, a) + ab
        bb = polygamma(1, n - y + b) - polygamma(1, b) + ab
        d1 = np.asarray(self.link.inverse_deriv(eta), float)
        d2 = np.asarray(self.link.inverse_deriv2(eta), float)
        t1 = theta * d1  # ∂a/∂η = −∂b/∂η; ∂a/∂log θ = a, ∂b/∂log θ = b
        h_ee = t1**2 * (aa + bb - 2 * ab) + theta * d2 * (da - db)
        h_el = t1 * (a * (aa - ab) + b * (ab - bb) + da - db)
        h_ll = a**2 * aa + 2 * a * b * ab + b**2 * bb + a * da + b * db
        if self._linhas_teto is not None:
            h_el, h_ll = np.where(self._linhas_teto, 0.0, h_el), np.where(self._linhas_teto, 0.0, h_ll)
        X, Z = self.exog, self.exog_disp
        H = np.block(
            [
                [X.T @ (X * h_ee[:, None]), X.T @ (Z * h_el[:, None])],
                [Z.T @ (X * h_el[:, None]), Z.T @ (Z * h_ll[:, None])],
            ]
        )
        for j in self.congelados:
            H[j, j] = -1.0  # parâmetro fora do ajuste: bloco inversível e sem covariância com os demais
        return H

    def predict(self, params, exog=None, *args, **kwds):
        """Retorna pi (média da proporção de votos)."""
        return self._pi(np.asarray(params), exog)

    def _start(self):
        # Chute inicial: GLM binomial; theta pelo momento da dispersão de Pearson.
        y, n = self.endog, self.n
        glm = sm.GLM(
            np.column_stack([y, n - y]),
            self.exog,
            family=sm.families.Binomial(link=self.link),
        ).fit()
        pi = np.clip(glm.fittedvalues, _EPS, 1 - _EPS)
        # E[Pearson^2 / n-ish]: phi = 1 + (n-1)rho  =>  rho médio por mínimos momentos.
        z2 = (y - n * pi) ** 2 / (n * pi * (1 - pi))
        w = n - 1
        rho = np.sum(z2 - 1) / np.sum(w)
        rho = float(np.clip(rho, 1e-8, 0.5))
        lt = np.log(1 / rho - 1)
        if self.exog_disp is not None:
            # gamma0 resolve log theta médio quando a 1a coluna de Z é a constante.
            gam = np.zeros(self.exog_disp.shape[1])
            gam[0] = lt
            return np.append(np.asarray(glm.params), gam)
        return np.append(np.asarray(glm.params), lt)

    def _otimizar(
        self,
        start_params: np.ndarray,
        method: str,
        maxiter: int,
        full_output: int,
        disp: bool,
        callback: Callable[[np.ndarray], None] | None,
        retall: int,
        **kwds: object,
    ) -> GenericLikelihoodModelResults:
        return super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            callback=callback,
            retall=retall,
            **kwds,
        )

    def fit(
        self, start_params=None, method="bfgs", maxiter=2000, full_output=1, disp=False, callback=None, retall=0, **kwds
    ):
        if start_params is None:
            start_params = self._start()
        # gtol sobre a log-verossimilhança SOMADA: um limite proporcional a N (1e-4*N)
        # parava ~390 unidades abaixo do ótimo; 1e-3 absoluto chega ao ótimo (< 0,05).
        # Só o BFGS aceita gtol; o L-BFGS usa pgtol e emite aviso com gtol.
        if method == "bfgs":
            kwds.setdefault("gtol", 1e-3)
        args = (method, maxiter, full_output, disp, callback, retall)
        res = self._otimizar(start_params, *args, **kwds)
        if self.exog_disp is None:
            return res
        if not self._teto_avaliado:
            res = self._aplicar_teto(res, args, kwds)
        if self.congelados:
            res = self._polir_newton(res)
            self._reposicionar_congelados(res.params)
        return res

    def _polir_newton(self, res: GenericLikelihoodModelResults) -> GenericLikelihoodModelResults:
        """Newton com a hessiana analítica a partir do ajuste com grupo no teto.

        Com theta fixo no teto sobra uma hessiana de condição ~1e7 (n chega a 1e6): BFGS e L-BFGS
        param com gradiente de ~0,7 na soma, sem convergir. Newton fecha em 3 passos. Só roda
        quando há grupo no teto, para os demais ajustes ficarem como estavam.
        """
        try:
            pol = self._otimizar(np.asarray(res.params, float), "newton", 50, 1, False, None, 0)
        except np.linalg.LinAlgError:
            return res
        return pol if pol.llf >= res.llf else res

    def _colunas_no_teto(self, params: np.ndarray) -> list[int]:
        """Dummies de dispersão cujo grupo tem theta >= THETA_CANDIDATO_TETO em toda linha."""
        Z, p = self.exog_disp, self._p
        assert Z is not None and p is not None, "só roda no caminho com submodelo de dispersão"
        theta = self._theta(params, exog_disp=None)
        achadas: list[int] = []
        for j in range(Z.shape[1]):
            col = Z[:, j]
            if not (set(np.unique(col)) <= {0.0, 1.0}) or col.sum() in (0, len(col)):
                continue
            g = col == 1.0
            if np.all(theta[g] >= THETA_CANDIDATO_TETO):
                achadas.append(p + j)
        return achadas

    def _aplicar_teto(
        self, res: GenericLikelihoodModelResults, args: _ArgsOtimizar, kwds: dict[str, Any]
    ) -> GenericLikelihoodModelResults:
        """Refaz o ajuste com os grupos no limite binomial fixados em THETA_TETO.

        Sem candidato, o resultado volta intacto. O teto só vale se a verossimilhança não cair
        mais que o LRT a 5 % admite (meio qui-quadrado com um gl por grupo): o grupo é então
        indistinguível do binomial.
        """
        self._teto_avaliado = True
        cols = self._colunas_no_teto(np.asarray(res.params))
        if not cols:
            return res
        Z, p = self.exog_disp, self._p
        assert Z is not None and p is not None, "só roda no caminho com submodelo de dispersão"
        linhas = np.any(Z[:, [c - p for c in cols]] == 1.0, axis=1)
        livres = [j for j in range(Z.shape[1]) if p + j not in cols]
        if linhas.all() or np.linalg.matrix_rank(Z[~linhas][:, livres]) < len(livres):
            return res  # grupo de referência no limite: sem dummy para fixar, o ajuste fica como está
        self.congelados, self._linhas_teto = cols, linhas
        ini = np.asarray(res.params, float)
        res_t = self._otimizar(ini, *args, **kwds)
        if res_t.llf < res.llf - stats.chi2.ppf(0.95, len(cols)) / 2:
            self.congelados, self._linhas_teto = [], None
            return res
        return res_t

    def _reposicionar_congelados(self, params: np.ndarray) -> None:
        """Põe a dummy fixada onde `exp(Z gamma)` do grupo dá THETA_TETO (previsões fora do treino usam Z gamma)."""
        Z, p = self.exog_disp, self._p
        assert Z is not None and p is not None, "só roda no caminho com submodelo de dispersão"
        for c in self.congelados:
            j = c - p
            g = Z[:, j] == 1.0
            resto = Z[g] @ params[p:] - params[c]
            params[c] = np.log(THETA_TETO) - resto.mean()


def _theta_ajustado(resultado):
    m = resultado.model
    if getattr(m, "exog_disp", None) is None:
        return float(np.exp(resultado.params[-1]))
    return m._theta(np.asarray(resultado.params))  # theta_i das linhas de treino


def icc(resultado):
    """rho = 1/(1+theta): correlação intraclasse (float; array se theta varia)."""
    th = _theta_ajustado(resultado)
    return float(1.0 / (1.0 + th)) if np.ndim(th) == 0 else 1.0 / (1.0 + th)


def theta_de(resultado):
    """theta (float no caminho antigo; array de theta_i do treino com dispersão)."""
    return _theta_ajustado(resultado)


# --------------------------------------------------------------------------
# Previsão e agregação
# --------------------------------------------------------------------------
def prever_pi(resultado, X):
    """pi estimado para GLM, BetaBinomial ou OLS sobre logit(share)."""
    modelo = getattr(resultado, "model", None)
    if isinstance(modelo, BetaBinomial):
        return modelo.predict(resultado.params, X)
    if isinstance(modelo, sm.GLM):
        return np.asarray(resultado.predict(X))
    return expit(np.asarray(resultado.predict(X)))


def agregar(pi, n, grupos) -> pd.Series:
    """Share ponderado por votos por grupo: sum(pi_i n_i) / sum(n_i).

    A série inclui o total nacional no índice 'Brasil'.
    """
    pi, n = np.asarray(pi, float), np.asarray(n, float)
    df = pd.DataFrame({"g": np.asarray(grupos), "v": pi * n, "n": n})
    s = df.groupby("g")[["v", "n"]].sum()
    out = s["v"] / s["n"]
    out.loc["Brasil"] = (pi * n).sum() / n.sum()
    return out


# --------------------------------------------------------------------------
# Validação cruzada
# --------------------------------------------------------------------------
def folds_kfold(n_obs: int, k: int = 5, seed: int = 0):
    """Lista de (idx_treino, idx_teste); cada índice é teste exatamente uma vez."""
    perm = np.random.default_rng(seed).permutation(n_obs)
    partes = np.array_split(perm, k)
    return [(np.sort(np.concatenate(partes[:i] + partes[i + 1 :])), np.sort(p)) for i, p in enumerate(partes)]


def folds_por_grupo(grupos):
    """Leave-one-group-out (ex.: um estado fora por vez)."""
    grupos = np.asarray(grupos)
    return [(np.flatnonzero(grupos != g), np.flatnonzero(grupos == g)) for g in pd.unique(grupos)]


def predicoes_cv(ajustar, prever, dados: pd.DataFrame, folds) -> np.ndarray:
    """pi fora da amostra alinhado a `dados`.

    ajustar(dados_treino) -> modelo; prever(modelo, dados_teste) -> pi.
    Em leave-one-state-out o chamador não deve usar efeitos fixos de estado:
    o nível do estado de teste não existe no treino e a previsão falharia
    ou seria arbitrária. Os callables decidem o tratamento de níveis novos.
    """
    saida = np.full(len(dados), np.nan)
    for tr, te in folds:
        modelo = ajustar(dados.iloc[tr])
        saida[te] = np.asarray(prever(modelo, dados.iloc[te]), dtype=float)
    return saida


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def metricas(y, n, pi) -> dict:
    """Erros do share em pontos percentuais (p.p.) e deviance binomial."""
    y, n, pi = (np.asarray(v, float) for v in (y, n, pi))
    pi = np.clip(pi, _EPS, 1 - _EPS)
    e = (pi - y / n) * 100
    w = n / n.sum()
    dev = 2 * np.sum(xlogy(y, y / (n * pi)) + xlogy(n - y, (n - y) / (n * (1 - pi))))
    return {
        "mae_pond_pp": float(np.sum(w * np.abs(e))),
        "rmse_pond_pp": float(np.sqrt(np.sum(w * e**2))),
        "mae_pp": float(np.mean(np.abs(e))),
        "deviance": float(dev),
        "erro_nacional_pp": float((np.sum(pi * n) - y.sum()) / n.sum() * 100),
    }


# --------------------------------------------------------------------------
# Resíduos quantílicos e envelope
# --------------------------------------------------------------------------
def residuos_quantilicos(y, n, pi, theta=None, rng=None) -> np.ndarray:
    """Resíduos quantílicos aleatorizados (Dunn & Smyth, 1996).

    u ~ U(F(y-1), F(y)), r = Phi^-1(u). theta=None: binomial; senão
    beta-binomial com a = pi*theta, b = (1-pi)*theta.
    """
    rng = np.random.default_rng(rng)
    y = np.asarray(y).astype(np.int64)
    n = np.asarray(n).astype(np.int64)
    pi = np.clip(np.asarray(pi, float), _EPS, 1 - _EPS)
    if theta is None:
        F = lambda k: stats.binom.cdf(k, n, pi)
    else:
        F = lambda k: stats.betabinom.cdf(k, n, pi * theta, (1 - pi) * theta)
    lo, hi = F(y - 1), F(y)
    u = np.clip(lo + rng.uniform(size=y.shape) * (hi - lo), 1e-10, 1 - 1e-10)
    return stats.norm.ppf(u)


def envelope_meio_normal(residuos_fn, simular_fn, ajustar_fn, B=99, nivel=0.95, rng=None) -> dict:
    """Envelope simulado para o gráfico half-normal de |resíduos|.

    ajustar_fn(y=None) -> resultado (None: ajusta aos dados observados);
    residuos_fn(resultado) -> resíduos; simular_fn(resultado, rng) -> y*.
    Retorna dict com 'observado' (|r| ordenado), 'teorico' (quantis
    half-normal), 'inferior', 'mediana', 'superior'.
    """
    rng = np.random.default_rng(rng)
    res0 = ajustar_fn(None)
    obs = np.sort(np.abs(residuos_fn(res0)))
    m = len(obs)
    sims = np.empty((B, m))
    for b in range(B):
        res_b = ajustar_fn(simular_fn(res0, rng))
        sims[b] = np.sort(np.abs(residuos_fn(res_b)))
    alfa = (1 - nivel) / 2
    teorico = stats.norm.ppf((np.arange(1, m + 1) + m - 1 / 8) / (2 * m + 1 / 2))
    return {
        "observado": obs,
        "teorico": teorico,
        "inferior": np.quantile(sims, alfa, axis=0),
        "mediana": np.median(sims, axis=0),
        "superior": np.quantile(sims, 1 - alfa, axis=0),
    }


# --------------------------------------------------------------------------
# Intervalos de previsão para agregados
# --------------------------------------------------------------------------
def simular_agregado(
    beta_hat, cov_beta, X, n, grupos, link=None, theta=None, B=1000, rng=None, lote=100
) -> pd.DataFrame:
    """Intervalos de previsão paramétricos para o share por grupo.

    beta ~ N(beta_hat, cov); pi = g^-1(X beta); Y ~ Bin(n, pi) ou, com theta,
    Y ~ BetaBin(n, pi*theta, (1-pi)*theta) (via p ~ Beta, Y ~ Bin(n, p)).
    Incorpora a incerteza de beta e a de amostragem, não a de theta.
    Colunas: ponto, q025, q50, q975 (frações); inclui o total 'Brasil'.
    """
    rng = np.random.default_rng(rng)
    link = _link(link)
    X, n = np.asarray(X, float), np.asarray(n, np.int64)
    cods, nomes = pd.factorize(np.asarray(grupos))
    G = sp.csr_matrix((np.ones(len(n)), (np.arange(len(n)), cods)), shape=(len(n), len(nomes)))
    G = sp.hstack([G, sp.csr_matrix(np.ones((len(n), 1)))]).tocsr()  # coluna do total
    nomes = list(nomes) + ["Brasil"]
    den = G.T @ n.astype(float)
    ponto = agregar(_pi_de_eta(X @ np.asarray(beta_hat), link), n, grupos)
    partes = []
    feitos = 0
    while feitos < B:
        b = min(lote, B - feitos)
        betas = rng.multivariate_normal(beta_hat, cov_psd(cov_beta), size=b)
        pi = _pi_de_eta(betas @ X.T, link)
        if theta is not None:
            pi = rng.beta(pi * theta, (1 - pi) * theta)
        nb = n[feitos : feitos + b] if n.ndim == 2 else n[None, :]
        Y = rng.binomial(nb, pi)
        partes.append((G.T @ Y.T.astype(float)).T / den)
        feitos += b
    sh = np.vstack(partes)
    q = np.quantile(sh, [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame(
        {"ponto": ponto.reindex(nomes).to_numpy(), "q025": q[0], "q50": q[1], "q975": q[2]},
        index=nomes,
    )


def simular_votos_grupo(
    beta_hat,
    cov_beta,
    X,
    n,
    grupos,
    link=None,
    theta=None,
    sigma=0.0,
    B=1000,
    rng=None,
    lote=100,
    offset=None,
    gl_uf: int | None = None,
):
    """Votos simulados por grupo: matriz B x G (colunas na ordem de np.unique(grupos)).

    beta ~ N(beta_hat, cov); choque de grupo delta_g ~ N(0, sigma^2) somado ao
    preditor linear de todos os municípios do grupo (um estado inteiro erra junto);
    pi = g^-1(X beta + delta); Y ~ Bin(n, pi) ou, com theta, BetaBin(n, pi*theta,
    (1-pi)*theta). Devolve a contagem de votos (não o share), para somar grupos
    (regiões, Brasil) antes de dividir pelo total de votos. `offset` (opcional) é
    somado ao preditor linear sem coeficiente (modelos com defasagem fixa). `n` pode ter
    forma (B, municípios): um total de votos por simulação (incerteza do restante).
    `gl_uf` (opcional): σ̂ é estimado de `gl_uf` resíduos de UF, então o choque de uma UF nova é
    σ̂·T com T de Student com `gl_uf` graus de liberdade; `None` mantém o choque normal.
    """
    rng = np.random.default_rng(rng)
    link = _link(link)
    X, n = np.asarray(X, float), np.asarray(n, np.int64)
    nomes, cods = np.unique(np.asarray(grupos), return_inverse=True)
    G = sp.csr_matrix((np.ones(X.shape[0]), (np.arange(X.shape[0]), cods)), shape=(X.shape[0], len(nomes)))
    partes, feitos = [], 0
    while feitos < B:
        b = min(lote, B - feitos)
        eta = rng.multivariate_normal(beta_hat, cov_psd(cov_beta), size=b) @ X.T
        if offset is not None:
            eta = eta + np.asarray(offset, float)[None, :]
        if sigma > 0:
            if gl_uf is None:
                choque = rng.normal(0.0, sigma, size=(b, len(nomes)))
            else:
                choque = sigma * rng.standard_t(gl_uf, size=(b, len(nomes)))
            eta = eta + choque[:, cods]
        pi = _pi_de_eta(eta, link)
        if theta is not None:
            pi = rng.beta(pi * theta, (1 - pi) * theta)
        nb = n[feitos : feitos + b] if n.ndim == 2 else n[None, :]
        Y = rng.binomial(nb, pi)
        partes.append((G.T @ Y.T.astype(float)).T)
        feitos += b
    return np.vstack(partes)
