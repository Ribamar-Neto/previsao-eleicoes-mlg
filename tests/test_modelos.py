import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

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
    for folds, N in [(m.folds_kfold(103, 5, seed=3), 103),
                     (m.folds_por_grupo(np.repeat(list("abcd"), 7)), 28)]:
        teste = np.concatenate([te for _, te in folds])
        assert sorted(teste) == list(range(N))
        for tr, te in folds:
            assert not set(tr) & set(te) and len(tr) + len(te) == N


def test_predicoes_cv_pior_que_dentro_da_amostra():
    X, y, n, _ = simular(N=300, theta=150.0)
    d = pd.DataFrame({"x": X[:, 1], "y": y, "n": n})

    def ajustar(df):
        return sm.GLM(np.column_stack([df.y, df.n - df.y]), sm.add_constant(df[["x"]], has_constant="add"),
                      family=sm.families.Binomial()).fit()

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
        ajustar, B=19, rng=0)
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
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    from graficos_diagnostico import residuos_quantilicos_rapido as rap
    X, y, n, Z, th = _simular_disp(N=200)
    pi = m._pi_de_eta(X @ np.array([-0.3, 0.8]), m._link(None))
    r = rap(y, n, pi, theta=th, rng=5)
    ref = m.residuos_quantilicos(y, n, pi, theta=th, rng=5)
    assert np.abs(r - ref).max() < 0.1
    assert np.allclose(rap(y, n, pi, theta=200.0, rng=5), rap(y, n, pi, theta=np.full(200, 200.0), rng=5))
