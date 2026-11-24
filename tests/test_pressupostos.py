"""Casos pequenos sintéticos para src/pressupostos.py."""
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import modelos as M
from src import pressupostos as P


def _bb(N: int = 400, seed: int = 3, quad: bool = False) -> tuple[Any, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    eta = -0.2 + 0.7 * x + (0.8 * x**2 if quad else 0.0)
    n = rng.integers(2000, 20000, N)
    pi = 1 / (1 + np.exp(-eta))
    y = rng.binomial(n, rng.beta(pi * 300, (1 - pi) * 300))
    X = np.column_stack([np.ones(N), x])
    res = M.BetaBinomial(y, X, n).fit()
    return res, rng.uniform(-30, -5, N), rng.uniform(-60, -35, N)


def test_vif_colunas_ortogonais_vale_um() -> None:
    rng = np.random.default_rng(0)
    A = np.linalg.qr(rng.normal(size=(500, 3)))[0] * 20
    X = pd.DataFrame(A, columns=["a", "b", "c"]).assign(const=1.0)
    tab = P.gvif(X)
    np.testing.assert_allclose(tab["vif"], 1.0, atol=0.05)


def test_vif_detecta_colinearidade_e_grupo() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    X = pd.DataFrame({"const": 1.0, "a": a, "b": a + 0.05 * rng.normal(size=500),
                      "d1": rng.normal(size=500), "d2": rng.normal(size=500)})
    tab = P.gvif(X, {"fator": ["d1", "d2"]}).set_index("fator")
    assert tab.loc["a", "vif"] > 100 and tab.loc["fator", "gl"] == 2
    assert tab.loc["fator", "gvif_ajustado"] < 1.2


def test_ligacao_nao_rejeita_sob_logit_verdadeiro() -> None:
    res, lat, lon = _bb()
    r = P.teste_ligacao(res, lat, lon, corte_km=1.0)
    assert r["p_modelo"] > 0.01 and r["dpi_max_pp"] < 2.0


def test_spline_rejeita_curvatura_e_aceita_reta() -> None:
    res_r, lat, lon = _bb(seed=5)
    res_q, _, _ = _bb(seed=5, quad=True)
    assert P.teste_spline(res_r, 1, lat, lon, 1.0)["p_modelo"] > 0.01
    assert P.teste_spline(res_q, 1, lat, lon, 1.0)["p_modelo"] < 1e-6


def test_base_spline_natural_e_linear_nas_caudas() -> None:
    x = np.linspace(-3, 3, 601)
    B = P.base_spline_natural(x, 4)
    assert B.shape == (601, 4)
    cauda = x > np.quantile(x, 0.95) + 0.1
    seg = np.diff(B[cauda, 1], n=2)
    np.testing.assert_allclose(seg, 0, atol=1e-6 * np.abs(B[:, 1]).max())


def test_dfbeta_um_passo_proximo_do_leave_one_out() -> None:
    rng = np.random.default_rng(2)
    N = 80
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + 0.8 * X[:, 1]))))
    res = sm.Logit(y, X).fit(disp=0)
    d, cook = P.cook_um_passo(res.model.score_obs(res.params), res.model.hessian(res.params))
    for i in (0, 7, 33):
        m = np.ones(N, bool); m[i] = False
        real = sm.Logit(y[m], X[m]).fit(disp=0).params - res.params
        np.testing.assert_allclose(d[i], real, atol=0.03)
    assert cook.shape == (N,) and (cook >= 0).all()


def test_reajuste_sem_remove_observacoes() -> None:
    res, _, _ = _bb(N=200)
    r2 = P.reajuste_sem(res, np.array([0, 1, 2]))
    assert r2.model.endog.shape[0] == 197
    np.testing.assert_allclose(P.mudanca_em_ep(res.params, res.params, np.eye(3)), 0)


def test_vif_bruto_detecta_colinearidade() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    X = pd.DataFrame({"a": a, "b": a + 0.05 * rng.normal(size=300), "c": rng.normal(size=300)})
    v = P.vif(X)
    assert list(v.index) == ["a", "b", "c"] and v.name == "VIF"
    assert v["a"] > 100 and v["b"] > 100 and v["c"] < 1.5


def test_alavanca_generalizada_traco_e_limites() -> None:
    res, _, _ = _bb()
    h = P.alavanca_generalizada(res)
    assert abs(h.sum() - 2) < 1e-6
    assert ((h > 0) & (h < 1)).all()


def test_alavanca_generalizada_theta_grande_iguala_glm_binomial() -> None:
    from types import SimpleNamespace

    res, _, _ = _bb()
    m = res.model
    glm = sm.GLM(np.column_stack([m.endog, m.n - m.endog]), m.exog,
                 family=sm.families.Binomial()).fit()
    falso = SimpleNamespace(model=m, params=np.append(glm.params, 40.0))
    np.testing.assert_allclose(P.alavanca_generalizada(falso),
                               glm.get_influence().hat_matrix_diag, atol=1e-6)
