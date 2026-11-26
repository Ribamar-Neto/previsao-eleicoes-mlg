"""Diagnóstico dos pressupostos de um beta-binomial já ajustado (nenhum modelo muda aqui).

Funções puras sobre o resultado do statsmodels (`BetaBinomial`): teste de ligação,
linearidade por spline natural, VIF/GVIF, influência de um passo e reajuste sem
observações. Os testes de Wald usam a covariância dada (Conley ou do modelo).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import espacial as S
from . import modelos as M

if TYPE_CHECKING:
    import numpy.typing as npt

    from .especificacoes import Ajuste, Candidato

Matriz = np.ndarray


def reajustar(res: Any, exog: Matriz, manter: np.ndarray | None = None) -> Any:
    """Reajusta o mesmo beta-binomial (dados, n, ligação e dispersão) com outra matriz de média.

    `exog` tem uma linha por observação do ajuste original; colunas a mais que as originais
    entram com chute 0 no fim dos coeficientes da média. `manter` (máscara booleana) tira
    observações do ajuste.
    """
    mod = res.model
    p0 = mod.exog.shape[1]
    params = np.asarray(res.params, float)
    novas = exog.shape[1] - p0
    if novas < 0:
        raise ValueError("exog deve conservar as colunas originais")
    inicio = np.concatenate([params[:p0], np.zeros(novas), params[p0:]])
    k = np.ones(len(mod.endog), bool) if manter is None else np.asarray(manter, bool)
    Z = None if mod.exog_disp is None else mod.exog_disp[k]
    novo = M.BetaBinomial(mod.endog[k], exog[k], mod.n[k], link=mod.link, exog_disp=Z)
    out = novo.fit(start_params=inicio)
    pol = novo.fit(start_params=out.params, method="lbfgs", maxiter=3000)
    return pol if pol.llf >= out.llf else out


def eta_padronizado(res: Any) -> np.ndarray:
    """Preditor linear η̂ do ajuste, padronizado (média 0, desvio 1)."""
    mod = res.model
    eta = mod.exog @ np.asarray(res.params, float)[: mod.exog.shape[1]]
    return (eta - eta.mean()) / eta.std(ddof=0)


def base_spline_natural(x: np.ndarray, gl: int = 4) -> np.ndarray:
    """Base de spline cúbica natural (Hastie et al.) com `gl` graus de liberdade, sem intercepto.

    Colunas: x (linear) e gl-1 termos não lineares; nós nos quantis igualmente espaçados
    entre 5% e 95% (gl + 1 nós).
    """
    x = np.asarray(x, float)
    nos = np.quantile(x, np.linspace(0.05, 0.95, gl + 1))
    ultimo = nos[-1]

    def d(k: int) -> np.ndarray:
        return (np.maximum(x - nos[k], 0) ** 3 - np.maximum(x - ultimo, 0) ** 3) / (ultimo - nos[k])

    cols = [x] + [d(k) - d(len(nos) - 2) for k in range(len(nos) - 2)]
    return np.column_stack(cols)


def extras_nao_lineares(x: np.ndarray, gl: int = 4) -> np.ndarray:
    """Os gl-1 termos não lineares da spline natural, ortogonais a (1, x) e com variância 1.

    Acrescentá-los à média com x já presente gera o mesmo modelo que trocar x pela spline.
    """
    B = base_spline_natural(x, gl)
    A = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    resto = B[:, 1:] - A @ np.linalg.lstsq(A, B[:, 1:], rcond=None)[0]
    resto = resto / resto.std(axis=0, ddof=0)
    return resto


def _dpi_max(p_novo: np.ndarray, p_base: np.ndarray) -> tuple[float, float]:
    d = 100 * np.abs(p_novo - p_base)
    return float(np.quantile(d, 0.95)), float(d.max())


def _cov_modelo(res: Any) -> np.ndarray:
    """Covariância do modelo: inversa de -H (hessiana da log-verossimilhança no ótimo)."""
    return np.linalg.inv(-np.asarray(res.model.hessian(res.params), float))


def wald_extras(res: Any, p0: int, q: int, cov: np.ndarray) -> tuple[float, float]:
    """Wald dos `q` coeficientes extras (posições p0..p0+q-1): (qui-quadrado, p)."""
    chi2, _, p = S.wald_robusto(res.params, cov, np.arange(p0, p0 + q))
    return chi2, p


def teste_ligacao(res: Any, lat: np.ndarray, lon: np.ndarray, corte_km: float) -> dict[str, float]:
    """Teste de Pregibon/Hinkley: η̂² padronizado como covariável extra na média."""
    mod = res.model
    p0 = mod.exog.shape[1]
    z = eta_padronizado(res)
    X2 = np.column_stack([mod.exog, z**2])
    novo = reajustar(res, X2)
    _, p_con = wald_extras(novo, p0, 1, np.asarray(S.cov_conley(novo, lat, lon, corte_km), float))
    _, p_mod = wald_extras(novo, p0, 1, _cov_modelo(novo))
    dp95, dmax = _dpi_max(novo.model.predict(novo.params), mod.predict(res.params))
    return {
        "coef": float(novo.params[p0]),
        "p_conley": p_con,
        "p_modelo": p_mod,
        "dpi_p95_pp": dp95,
        "dpi_max_pp": dmax,
        "dloglik": float(novo.llf - res.llf),
    }


def teste_spline(
    res: Any, coluna: int, lat: np.ndarray, lon: np.ndarray, corte_km: float, gl: int = 4
) -> dict[str, float]:
    """Linearidade de uma covariável: Wald Conley dos gl-1 termos não lineares da spline natural."""
    mod = res.model
    p0 = mod.exog.shape[1]
    ext = extras_nao_lineares(mod.exog[:, coluna], gl)
    novo = reajustar(res, np.column_stack([mod.exog, ext]))
    chi2_c, p_con = wald_extras(novo, p0, ext.shape[1], np.asarray(S.cov_conley(novo, lat, lon, corte_km), float))
    _, p_mod = wald_extras(novo, p0, ext.shape[1], _cov_modelo(novo))
    dp95, dmax = _dpi_max(novo.model.predict(novo.params), mod.predict(res.params))
    return {
        "chi2_conley": chi2_c,
        "p_conley": p_con,
        "p_modelo": p_mod,
        "dpi_p95_pp": dp95,
        "dpi_max_pp": dmax,
        "dloglik": float(novo.llf - res.llf),
    }


def residuo_parcial(res: Any, coluna: int) -> np.ndarray:
    """Resíduo de trabalho (y/n - π)/(dπ/dη) mais β_j x_j."""
    mod = res.model
    pi = mod.predict(res.params)
    dpi = np.asarray(mod.link.inverse_deriv(mod.exog @ np.asarray(res.params)[: mod.exog.shape[1]]), float)
    trab = (mod.endog / mod.n - pi) / dpi
    return trab + float(res.params[coluna]) * mod.exog[:, coluna]


def medias_por_faixa(x: np.ndarray, v: np.ndarray, peso: np.ndarray, faixas: int = 20) -> pd.DataFrame:
    """Médias ponderadas de x e v em `faixas` faixas de quantil de x."""
    x, v, peso = (np.asarray(a, float) for a in (x, v, peso))
    corte = np.quantile(x, np.linspace(0, 1, faixas + 1))
    g = np.clip(np.searchsorted(corte, x, side="right") - 1, 0, faixas - 1)
    linhas = []
    for f in range(faixas):
        m = g == f
        if m.any():
            linhas.append({"x": np.average(x[m], weights=peso[m]), "v": np.average(v[m], weights=peso[m])})
    return pd.DataFrame(linhas)


def vif(X: pd.DataFrame) -> pd.Series:
    """VIF de cada covariável bruta, padronizada, com constante (não é o da matriz do modelo: ver `gvif`)."""
    cols = list(X.columns)
    Z = (X - X.mean()) / X.std()
    Z.insert(0, "const", 1.0)
    return pd.Series([variance_inflation_factor(Z.values, i + 1) for i in range(len(cols))], index=cols, name="VIF")


def gvif(X: pd.DataFrame, grupos: dict[str, Sequence[str]] | None = None) -> pd.DataFrame:
    """VIF por coluna e GVIF^(1/(2 gl)) por fator, sobre a matriz de correlação sem a constante.

    `grupos` junta colunas de um mesmo fator (dummies). Colunas fora de grupos têm gl = 1,
    e então GVIF^(1/2) = sqrt(VIF); a coluna `vif` guarda GVIF (= VIF para gl = 1) e
    `gvif_ajustado` o GVIF^(1/(2 gl)). O limite VIF > 10 equivale a ajustado > sqrt(10).
    """
    cols = [c for c in X.columns if c != "const" and X[c].std() > 0]
    R = np.corrcoef(X[cols].to_numpy(float), rowvar=False)
    pos = {c: i for i, c in enumerate(cols)}
    usados = {c for g in (grupos or {}).values() for c in g}
    fatores: dict[str, list[str]] = {c: [c] for c in cols if c not in usados}
    fatores.update({k: [c for c in v if c in pos] for k, v in (grupos or {}).items()})
    det_all = np.linalg.det(R)
    linhas = []
    for nome, membros in fatores.items():
        i = [pos[c] for c in membros]
        r = [j for j in range(len(cols)) if j not in i]
        g = np.linalg.det(R[np.ix_(i, i)]) * np.linalg.det(R[np.ix_(r, r)]) / det_all
        linhas.append({"fator": nome, "gl": len(i), "vif": float(g), "gvif_ajustado": float(g ** (1 / (2 * len(i))))})
    return pd.DataFrame(linhas)


def numero_condicao(X: pd.DataFrame) -> float:
    """Número de condição de Belsley: colunas (com a constante) escaladas a norma 1."""
    A = X.to_numpy(float)
    A = A / np.linalg.norm(A, axis=0)
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1])


def cook_um_passo(score: Matriz, hessiana: Matriz) -> tuple[Matriz, np.ndarray]:
    """DFBETA de um passo e distância de Cook.

    Δβ_i = β_(i) - β̂ ≈ H⁻¹ s_i (H: hessiana da log-verossimilhança, negativa definida;
    s_i: escore da observação i); Cook_i = Δβ_i' (-H) Δβ_i / p.
    """
    H = np.asarray(hessiana, float)
    d = np.linalg.solve(H, np.asarray(score, float).T).T
    cook = np.einsum("ij,jk,ik->i", d, -H, d) / H.shape[0]
    return d, cook


def alavanca_generalizada(res: Any) -> np.ndarray:
    """Alavanca h_ii do submodelo da média do beta-binomial ajustado.

    h_ii = w_i x_i' (X'WX)^-1 x_i, com
    w_i = n_i (dπ/dη)_i² / [π_i (1-π_i) (1 + (n_i-1) ρ_i)] e ρ_i = 1/(1+θ_i).
    É a matriz chapéu de mínimos quadrados ponderados do submodelo da média sob a variância
    beta-binomial (alavanca generalizada, Paula, Modelos de Regressão); com ρ→0 recai na
    alavanca do GLM binomial. Soma = nº de parâmetros da média.
    """
    m = res.model
    params = np.asarray(res.params, float)
    X = np.asarray(m.exog, float)
    p = m._p if m._p is not None else len(params) - 1
    eta = X @ params[:p]
    pi = M._pi_de_eta(eta, m.link)
    dpi = np.asarray(m.link.inverse_deriv(eta), float)
    rho = 1.0 / (1.0 + m._theta(params))
    w = m.n * dpi**2 / (pi * (1 - pi) * (1 + (m.n - 1) * rho))
    Xw = X * np.sqrt(w)[:, None]
    return np.einsum("ij,ij->i", Xw @ np.linalg.inv(Xw.T @ Xw), Xw)


def mudanca_em_ep(b_cheio: np.ndarray, b_sem: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """(β sem - β cheio) / EP, com o EP tirado da diagonal de `cov`."""
    return (np.asarray(b_sem, float) - np.asarray(b_cheio, float)) / np.sqrt(np.diag(cov))


def reajuste_sem(res: Any, indices: np.ndarray) -> Any:
    """Reajusta o mesmo modelo sem as observações `indices`."""
    manter = np.ones(len(res.model.endog), bool)
    manter[np.asarray(indices, int)] = False
    return reajustar(res, res.model.exog, manter)


def residuos_quantilicos_rapido(
    y: npt.ArrayLike,
    n: npt.ArrayLike,
    pi: npt.ArrayLike,
    theta: npt.ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
    base: int = 400,
    c: int = 32,
) -> np.ndarray:
    """Resíduo quantílico aleatorizado (Dunn & Smyth) com a FDA beta-binomial por quadratura.

    A `scipy.stats.betabinom.cdf` soma a pmf termo a termo (~9 s para 5.570 municípios).
    Aqui F(k) = E_p[Binomial(k; n, p)], p ~ Beta(pi*theta, (1-pi)*theta), integrada nos
    quantis do meio da beta. A binomial é um degrau de largura sqrt(p(1-p)/n) em p; por isso o
    número de pontos cresce com n: J = max(base, c*sqrt(n/theta)), arredondado para base*2^k.
    Erro máximo da FDA ~2e-4 (conferido contra a scipy no notebook). theta=None: binomial.
    """
    rng = np.random.default_rng(rng)
    y, n = np.asarray(y).astype(np.int64), np.asarray(n).astype(np.int64)
    pv = np.clip(np.asarray(pi, float), 1e-9, 1 - 1e-9)
    if theta is None:

        def fda(k: np.ndarray) -> np.ndarray:
            return stats.binom.cdf(k, n, pv)
    else:
        tv = np.asarray(theta, float)  # escalar ou um theta por linha
        pontos = np.maximum(base, np.ceil(c * np.sqrt(n / tv))).astype(int)
        nivel = base * 2 ** np.ceil(np.log2(pontos / base)).astype(int)

        def fda(k: np.ndarray) -> np.ndarray:
            out = np.empty(len(k))
            for jb in np.unique(nivel):
                i = np.where(nivel == jb)[0]
                th = tv if tv.ndim == 0 else tv[i]
                p = stats.beta.ppf(((np.arange(jb) + 0.5) / jb)[:, None], pv[i] * th, (1 - pv[i]) * th)
                out[i] = stats.binom.cdf(k[i][None, :], n[i][None, :], p).mean(axis=0)
            return out

    lo, hi = fda(y - 1), fda(y)
    u = np.clip(lo + rng.uniform(size=y.shape) * (hi - lo), 1e-10, 1 - 1e-10)
    return stats.norm.ppf(u)


def envelope_do_modelo(
    cand: Candidato,
    df: pd.DataFrame,
    mapa: dict[str, str],
    theta_fn: Callable[[Ajuste], float | np.ndarray | None],
    B: int,
    seed: int,
) -> dict:
    """Envelope meio-normal por bootstrap paramétrico de um candidato (`Candidato`).

    `theta_fn(ajuste)` dá a precisão do beta-binomial (escalar ou um theta por município)
    ou None para a binomial pura. Os resíduos de cada reajuste usam o y simulado daquele
    reajuste, não o observado.
    """
    y, n = df[mapa["y"]].to_numpy(float), df[mapa["n"]].to_numpy(float)
    estado = {"y": y}

    def ajustar_fn(ys: np.ndarray | None = None) -> Ajuste:
        estado["y"] = y if ys is None else ys
        return cand.ajustar(df if ys is None else df.assign(**{mapa["y"]: ys}), mapa)

    def residuos_fn(a: Ajuste) -> np.ndarray:
        return residuos_quantilicos_rapido(estado["y"], n, cand.prever(a, df, mapa), theta=theta_fn(a), rng=0)

    def simular_fn(a: Ajuste, rng: np.random.Generator) -> np.ndarray:
        pi_ = cand.prever(a, df, mapa)
        th = theta_fn(a)
        if th is not None:
            pi_ = rng.beta(pi_ * th, (1 - pi_) * th)
        return rng.binomial(n.astype(int), pi_).astype(float)

    return M.envelope_meio_normal(residuos_fn, simular_fn, ajustar_fn, B=B, rng=seed)
