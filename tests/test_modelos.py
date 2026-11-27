import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.families.links import Link

from src import modelos as m

L = sm.families.links
LINKS = [L.Logit, L.Probit, L.CLogLog, L.LogLog, L.Cauchy]


def simular(N=600, theta=200.0, beta=(-0.3, 0.8), seed=1, link=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    X = np.column_stack([np.ones(N), x])
    n = np.exp(rng.uniform(np.log(1e3), np.log(1e5), N)).astype(int)
    pi = m._pi_de_eta(X @ np.array(beta), m._link(link))
    if theta is None:
        y = rng.binomial(n, pi)
    else:
        y = rng.binomial(n, rng.beta(pi * theta, (1 - pi) * theta))
    return X, y, n, pi


@pytest.mark.parametrize("link", LINKS)
def test_glm_links_prever_pi(link):
    X, y, n, _ = simular(theta=None, beta=(0.1, 0.3), link=link)
    r = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial(link=link())).fit()
    np.testing.assert_allclose(m.prever_pi(r, X), r.fittedvalues, rtol=1e-8)


def test_prever_pi_ols_logit():
    X, y, n, _ = simular()
    z = np.log(y / (n - y))
    r = sm.OLS(z, X).fit()
    np.testing.assert_allclose(m.prever_pi(r, X), 1 / (1 + np.exp(-r.fittedvalues)))


def test_betabinomial_recupera_parametros():
    X, y, n, _ = simular(N=800, theta=200.0, beta=(-0.3, 0.8))
    r = m.BetaBinomial(y, X, n).fit()
    assert np.allclose(r.params[:2], [-0.3, 0.8], atol=0.05)
    assert abs(np.log(m.theta_de(r) / 200.0)) < 0.2
    assert np.isfinite(r.aic) and np.isfinite(r.bic)
    glm = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit()
    assert r.aic < glm.aic
    assert m.icc(r) == pytest.approx(1 / 201, rel=0.25)
    assert np.all((m.prever_pi(r, X) > 0) & (m.prever_pi(r, X) < 1))


def test_betabinomial_link_probit():
    X, y, n, _ = simular(theta=300.0, beta=(0.1, 0.4), link=L.Probit)
    r = m.BetaBinomial(y, X, n, link=L.Probit()).fit()
    assert np.allclose(r.params[:2], [0.1, 0.4], atol=0.05)


def test_agregar_exemplo():
    s = m.agregar([0.5, 0.3, 0.8], [100, 100, 200], ["A", "A", "B"])
    assert s["A"] == pytest.approx(0.4)
    assert s["B"] == pytest.approx(0.8)
    assert s["Brasil"] == pytest.approx((50 + 30 + 160) / 400)


def test_folds_particionam():
    for folds, N in [(m.folds_kfold(103, 5, seed=3), 103), (m.folds_por_grupo(np.repeat(list("abcd"), 7)), 28)]:
        teste = np.concatenate([te for _, te in folds])
        assert sorted(teste) == list(range(N))
        for tr, te in folds:
            assert not set(tr) & set(te) and len(tr) + len(te) == N


def test_predicoes_cv_pior_que_dentro_da_amostra():
    X, y, n, _ = simular(N=300, theta=150.0)
    d = pd.DataFrame({"x": X[:, 1], "y": y, "n": n})

    def ajustar(df):
        return sm.GLM(
            np.column_stack([df.y, df.n - df.y]),
            sm.add_constant(df[["x"]], has_constant="add"),
            family=sm.families.Binomial(),
        ).fit()

    def prever(mod, df):
        return m.prever_pi(mod, sm.add_constant(df[["x"]], has_constant="add"))

    cv = m.predicoes_cv(ajustar, prever, d, m.folds_kfold(len(d), 5, seed=0))
    assert not np.isnan(cv).any()
    dentro = m.metricas(d.y, d.n, prever(ajustar(d), d))
    fora = m.metricas(d.y, d.n, cv)
    assert fora["rmse_pond_pp"] > dentro["rmse_pond_pp"]
    assert fora["deviance"] > dentro["deviance"]


def test_metricas_exemplo():
    r = m.metricas([50, 30], [100, 100], [0.6, 0.3])
    assert r["mae_pp"] == pytest.approx(5.0)
    assert r["mae_pond_pp"] == pytest.approx(5.0)
    assert r["rmse_pond_pp"] == pytest.approx(np.sqrt(50))
    assert r["erro_nacional_pp"] == pytest.approx(5.0)


def test_residuos_quantilicos_normais():
    X, y, n, pi = simular(N=500, theta=None)
    r = m.residuos_quantilicos(y, n, pi, rng=1)
    assert stats.kstest(r, "norm").pvalue > 0.01
    X, y, n, pi = simular(N=500, theta=200.0, seed=4)
    r = m.residuos_quantilicos(y, n, pi, theta=200.0, rng=2)
    assert stats.kstest(r, "norm").pvalue > 0.01


def test_envelope_meio_normal():
    X, y0, n, pi = simular(N=60, theta=None, beta=(0.0, 0.5), seed=7)

    def ajustar(y=None):
        yy = y0 if y is None else y
        return sm.GLM(np.column_stack([yy, n - yy]), X, family=sm.families.Binomial()).fit()

    env = m.envelope_meio_normal(
        lambda r: m.residuos_quantilicos(np.round(r.model.endog * n), n, r.fittedvalues, rng=0),
        lambda r, rng: rng.binomial(n, r.fittedvalues),
        ajustar,
        B=19,
        rng=0,
    )
    assert len(env["observado"]) == 60
    assert np.all(env["inferior"] <= env["mediana"]) and np.all(env["mediana"] <= env["superior"])


def test_simular_agregado_cobertura_e_velocidade():
    rng = np.random.default_rng(11)
    N = 5570
    g = rng.integers(0, 27, N)
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    n = np.exp(rng.uniform(np.log(1e3), np.log(1e6), N)).astype(int)
    beta = np.array([0.0, 0.5])
    pi = 1 / (1 + np.exp(-X @ beta))
    y = rng.binomial(n, pi)
    r = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit()
    out = m.simular_agregado(r.params, r.cov_params(), X, n, g, B=1000, rng=3)
    assert out.shape == (28, 4)
    verdade = (pi * n).sum() / n.sum()
    assert out.loc["Brasil", "q025"] <= verdade <= out.loc["Brasil", "q975"]
    real = pd.Series(y * 1.0).groupby(g).sum() / pd.Series(n * 1.0).groupby(g).sum()
    ic = out.loc[real.index]
    cobre = (ic.q025.to_numpy() <= real.to_numpy()) & (real.to_numpy() <= ic.q975.to_numpy())
    assert cobre.mean() > 0.7
    out2 = m.simular_agregado(r.params, r.cov_params(), X, n, g, theta=500.0, B=50, rng=3)
    assert (out2.q975 - out2.q025 > 0).all()


def test_simular_votos_grupo_forma_media_e_choque():
    rng = np.random.default_rng(1)
    N = 300
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    n = rng.integers(500, 3000, N)
    g = np.repeat(["a", "b", "c"], 100)
    beta, cov = np.array([-0.2, 0.5]), np.diag([1e-4, 1e-4])
    base = m.simular_votos_grupo(beta, cov, X, n, g, B=400, rng=2)
    assert base.shape == (400, 3)
    esperado = np.array([(n[g == k] / (1 + np.exp(-X[g == k] @ beta))).sum() for k in "abc"])
    np.testing.assert_allclose(base.mean(0), esperado, rtol=0.01)
    choque = m.simular_votos_grupo(beta, cov, X, n, g, theta=600.0, sigma=0.1, B=400, rng=2)
    assert choque.shape == (400, 3)
    assert np.all(choque.std(0) > 3 * base.std(0))


# --------------------------------------------------------------------------
# Dispersão por covariáveis (double GLM)
# --------------------------------------------------------------------------
def _simular_disp(N=3000, gam=(np.log(300.0), -1.0), seed=3):
    rng = np.random.default_rng(seed)
    X, _, n, pi = simular(N=N, theta=None, seed=seed)
    g = rng.integers(0, 2, N)
    Z = np.column_stack([np.ones(N), g])
    th = np.exp(Z @ np.array(gam))
    y = rng.binomial(n, rng.beta(pi * th, (1 - pi) * th))
    return X, y, n, Z, th


def test_dispersao_por_grupo_recupera_gamma():
    X, y, n, Z, _ = _simular_disp()
    mod = m.BetaBinomial(y, X, n, exog_disp=pd.DataFrame(Z, columns=["const", "g"]))
    r = mod.fit()
    assert mod.exog_names[-2:] == ["disp_const", "disp_g"]
    assert np.allclose(r.params[-2:], [np.log(300.0), -1.0], atol=0.2)
    assert m.theta_de(r).shape == (len(y),)
    assert m.icc(r).shape == (len(y),)


def test_score_analitico_igual_ao_numerico():
    from statsmodels.tools.numdiff import approx_fprime

    X, y, n, Z, _ = _simular_disp(N=300)
    mod = m.BetaBinomial(y, X, n, exog_disp=Z)
    p = np.array([-0.3, 0.8, np.log(300.0), -1.0]) + 0.05
    num = approx_fprime(p, mod.loglike, centered=True)
    ana = mod.score(p)
    assert np.allclose(ana, num, rtol=1e-5, atol=1e-6 * np.abs(num).max())


@pytest.mark.parametrize("link", [L.Logit, L.Probit])
def test_hessiana_analitica_igual_a_jacobiana_do_score(link: type[Link]) -> None:
    # n até 1e6, como nos municípios grandes: as diferenças segundas da log-verossimilhança
    # somada erram ~1e-3 do maior elemento e, com autovalores de 0,04 a 3e6 (corte 2026/5),
    # tornam a informação indefinida, zeram cov_params e corrompem o pão de Conley.
    from statsmodels.tools.numdiff import approx_fprime

    rng = np.random.default_rng(3)
    N = 800
    X = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
    g = rng.integers(0, 3, N)
    Z = np.column_stack([np.ones(N), g == 1, g == 2]).astype(float)
    n = np.exp(rng.uniform(np.log(50), np.log(1e6), N)).astype(int)
    pi = m._pi_de_eta(X @ np.array([0.1, 0.4, -0.3]), m._link(link))
    th = np.exp(Z @ np.array([np.log(80.0), 0.5, -0.4]))
    y = rng.binomial(n, rng.beta(pi * th, (1 - pi) * th))
    mod = m.BetaBinomial(y, X, n, link=link, exog_disp=Z)
    p = np.array([0.1, 0.4, -0.3, np.log(80.0), 0.5, -0.4]) + 0.03
    jac = approx_fprime(p, mod.score, centered=True)
    hes = mod.hessian(p)
    assert np.abs(hes - jac).max() <= 1e-6 * np.abs(jac).max()
    assert np.allclose(hes, hes.T)


def test_sem_exog_disp_mantem_caminho_antigo():
    X, y, n, _ = simular(N=400, theta=200.0)
    mod = m.BetaBinomial(y, X, n)
    assert mod.exog_names[-1] == "log_theta" and mod.k_extra == 1
    r = mod.fit()
    assert isinstance(m.theta_de(r), float) and isinstance(m.icc(r), float)
    # uma coluna de Z constante reproduz o ajuste com theta único
    r2 = m.BetaBinomial(y, X, n, exog_disp=np.ones((len(y), 1))).fit()
    assert abs(r2.llf - r.llf) < 0.05


def test_theta_vetor_em_residuos_e_simulacao():
    X, y, n, Z, th = _simular_disp(N=400)
    pi = m._pi_de_eta(X @ np.array([-0.3, 0.8]), m._link(None))
    r = m.residuos_quantilicos(y, n, pi, theta=th, rng=1)
    assert r.shape == y.shape and abs(r.mean()) < 0.3 and 0.7 < r.std() < 1.3
    # theta constante em vetor == escalar
    ra = m.residuos_quantilicos(y, n, pi, theta=np.full(len(y), 200.0), rng=1)
    rb = m.residuos_quantilicos(y, n, pi, theta=200.0, rng=1)
    assert np.allclose(ra, rb)
    g = np.array(["a", "b"] * (len(y) // 2))
    cov = np.diag([1e-6, 1e-6])
    out = m.simular_votos_grupo(np.array([-0.3, 0.8]), cov, X, n, g, theta=th, B=20, rng=2)
    assert out.shape == (20, 2)
    esp = np.array([(n * pi)[g == k].sum() for k in ("a", "b")])
    assert np.allclose(out.mean(0), esp, rtol=0.05)


def test_residuos_quantilicos_rapido_theta_vetor():
    from src.pressupostos import residuos_quantilicos_rapido as rap

    X, y, n, Z, th = _simular_disp(N=200)
    pi = m._pi_de_eta(X @ np.array([-0.3, 0.8]), m._link(None))
    r = rap(y, n, pi, theta=th, rng=5)
    ref = m.residuos_quantilicos(y, n, pi, theta=th, rng=5)
    assert np.abs(r - ref).max() < 0.1
    assert np.allclose(rap(y, n, pi, theta=200.0, rng=5), rap(y, n, pi, theta=np.full(200, 200.0), rng=5))


def test_cov_psd_mantem_matriz_psd():
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(m.cov_psd(c), c)


def test_cov_psd_corta_indefinida_e_avisa():
    c = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.warns(UserWarning, match="não PSD"):
        r = m.cov_psd(c)
    assert np.linalg.eigvalsh(r).min() >= -1e-12


def test_simular_votos_grupo_aceita_n_por_simulacao():
    X = np.ones((3, 1))
    n1 = np.array([100, 100, 100])
    n2 = np.vstack([n1 * 0, n1 * 0])
    beta, cov = np.array([0.0]), np.zeros((1, 1))
    g = np.array(["a", "a", "b"])
    a = m.simular_votos_grupo(beta, cov, X, n1, g, B=2, rng=1)
    b = m.simular_votos_grupo(beta, cov, X, np.vstack([n1, n1]), g, B=2, rng=1)
    assert np.allclose(a, b)
    assert (m.simular_votos_grupo(beta, cov, X, n2.astype(int)[:, :3], g, B=2, rng=1) == 0).all()


# --------------------------------------------------------------------------
# Teto de theta (grupo no limite binomial)
# --------------------------------------------------------------------------
def _simular_grupo_binomial(
    N: int = 900, seed: int = 7, theta_grupos: tuple[float | None, ...] = (300.0, 100.0, None)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Três grupos (referência, g1, g2); `None` em theta_grupos = grupo binomial (theta -> infinito)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    n = rng.integers(500, 5000, N)
    g = rng.integers(0, 3, N)
    pi = 1 / (1 + np.exp(-(X @ np.array([-0.3, 0.8]))))
    y = np.empty(N, dtype=int)
    for k, th in enumerate(theta_grupos):
        i = g == k
        y[i] = rng.binomial(n[i], pi[i]) if th is None else rng.binomial(n[i], rng.beta(pi[i] * th, (1 - pi[i]) * th))
    Z = np.column_stack([np.ones(N), g == 1, g == 2]).astype(float)
    return X, y, n, Z, g


def test_grupo_binomial_converge_com_theta_no_teto() -> None:
    X, y, n, Z, g = _simular_grupo_binomial()
    mod = m.BetaBinomial(y, X, n, exog_disp=Z)
    r = mod.fit()
    assert mod.congelados == [X.shape[1] + 2]  # a dummy do grupo binomial
    th = m.theta_de(r)
    assert np.all(th[g == 2] == m.THETA_TETO)
    assert np.allclose(np.median(th[g == 0]), 300.0, rtol=0.5)
    assert np.allclose(np.median(th[g == 1]), 100.0, rtol=0.5)
    assert r.mle_retvals["converged"]
    assert np.abs(mod.score(np.asarray(r.params))).max() < 1e-2  # soma sobre 900 linhas
    assert np.exp(r.params[X.shape[1]] + r.params[X.shape[1] + 2]) == pytest.approx(m.THETA_TETO)  # prever_theta
    H = mod.hessian(np.asarray(r.params))
    assert np.all(np.isfinite(np.linalg.inv(H)))
    assert np.all(np.isfinite(r.bse))


def test_dispersao_dentro_do_teto_fica_igual() -> None:
    X, y, n, Z, _ = _simular_grupo_binomial(theta_grupos=(300.0, 100.0, 200.0))
    mod = m.BetaBinomial(y, X, n, exog_disp=Z)
    r = mod.fit()
    assert mod.congelados == []
    ref = m.BetaBinomial(y, X, n, exog_disp=Z)
    r0 = m.GenericLikelihoodModel.fit(
        ref, start_params=ref._start(), method="bfgs", maxiter=2000, gtol=1e-3, disp=False, full_output=1
    )
    np.testing.assert_array_equal(np.asarray(r.params), np.asarray(r0.params))


def _logit_do_share(gl_uf: int | None) -> np.ndarray:
    """Logit do share de uma única UF quase sem ruído binomial: isola o choque de UF."""
    X, n, g = np.ones((1, 1)), np.array([10**9]), np.array(["a"])
    cov = np.zeros((1, 1))
    y = m.simular_votos_grupo(np.array([0.0]), cov, X, n, g, sigma=0.2, B=20000, rng=3, gl_uf=gl_uf)[:, 0]
    return np.log(y / (n[0] - y))


def test_choque_de_uf_normal_e_o_padrao() -> None:
    X, n, g = np.ones((6, 1)), np.full(6, 5000), np.repeat(["a", "b"], 3)
    args = (np.array([0.1]), np.zeros((1, 1)), X, n, g)
    a = m.simular_votos_grupo(*args, sigma=0.2, B=50, rng=4)
    b = m.simular_votos_grupo(*args, sigma=0.2, B=50, rng=4, gl_uf=None)
    np.testing.assert_array_equal(a, b)


def test_choque_de_uf_t_tem_variancia_sigma2_k_sobre_k_menos_2() -> None:
    k = 10
    var_t = _logit_do_share(k).var()
    assert var_t == pytest.approx(0.2**2 * k / (k - 2), rel=0.08)
    assert _logit_do_share(None).var() == pytest.approx(0.2**2, rel=0.08)
