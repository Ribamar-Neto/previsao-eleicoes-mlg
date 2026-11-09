import geopandas as gpd
import numpy as np
import pytest
import statsmodels.api as sm
from shapely.geometry import box

from src import espacial as e
from src import modelos as M


def grade(n=12):
    geo = gpd.GeoDataFrame(
        geometry=[box(i, j, i + 1, j + 1) for j in range(n) for i in range(n)], crs=None
    )
    return geo


def simular(N=400, seed=3):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-30, 0, N)
    lon = rng.uniform(-60, -40, N)
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    n = rng.integers(500, 5000, N)
    pi = 1 / (1 + np.exp(-(X @ [-0.3, 0.6])))
    y = rng.binomial(n, pi)
    res = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit()
    return res, lat, lon


def simular_com_grupos(N=400, seed=3):
    # Reajusta com o endog de 2 colunas original: `res.model.endog` já é a proporção, e
    # reajustar com ela perde os n de ensaios (o HC0 sairia de outro modelo).
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-30, 0, N), rng.uniform(-60, -40, N)
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    n = rng.integers(500, 5000, N)
    y = rng.binomial(n, 1 / (1 + np.exp(-(X @ [-0.3, 0.6]))))
    endog = np.column_stack([y, n - y])
    g = rng.integers(0, 20, N)
    return endog, X, lat, lon, g


def test_conley_corte_zero_igual_statsmodels_hc0_binomial():
    endog, X, lat, lon, _ = simular_com_grupos()
    fam = sm.families.Binomial()
    res = sm.GLM(endog, X, family=fam).fit()
    ref = sm.GLM(endog, X, family=fam).fit(cov_type="HC0").cov_params()
    np.testing.assert_allclose(e.cov_conley(res, lat, lon, 1e-6).values, ref, rtol=1e-6)


def test_cluster_igual_statsmodels_binomial():
    endog, X, _, _, g = simular_com_grupos()
    fam = sm.families.Binomial()
    res = sm.GLM(endog, X, family=fam).fit()
    ref = sm.GLM(endog, X, family=fam).fit(
        cov_type="cluster", cov_kwds={"groups": g, "use_correction": False}).cov_params()
    np.testing.assert_allclose(e.cov_cluster(res, g, correcao=False).values, ref, rtol=1e-6)


def hc0_explicito(res):
    S = res.model.score_obs(np.asarray(res.params), scale=1.0)
    H = res.model.hessian(np.asarray(res.params), scale=1.0)
    Hi = np.linalg.inv(H)
    return Hi @ S.T @ S @ Hi


def test_conley_corte_zero_igual_hc0_explicito():
    res, lat, lon = simular()
    cov = e.cov_conley(res, lat, lon, 1e-6)
    np.testing.assert_allclose(cov.values, hc0_explicito(res), rtol=1e-6)


def _gaussiano(N=300, seed=2):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    y = X @ [1.0, 2.0] + rng.normal(size=N) * (1 + np.abs(X[:, 1]))
    return y, X, rng.uniform(-30, 0, N), rng.uniform(-60, -40, N)


def test_conley_corte_zero_vs_statsmodels_hc0_ols():
    # No caso gaussiano o HC0 do OLS é o mesmo sanduíche H^-1 S'S H^-1.
    y, X, lat, lon = _gaussiano()
    glm = sm.GLM(y, X).fit()
    ref = sm.OLS(y, X).fit(cov_type="HC0").cov_params()
    # escore na escala 1 equivale ao do OLS; a hessiana usa a mesma escala
    np.testing.assert_allclose(e.cov_conley(glm, lat, lon, 1e-6).values, ref, rtol=1e-6)


def test_conley_cresce_com_autocorrelacao():
    res, lat, lon = simular()
    assert (np.diag(e.cov_conley(res, lat, lon, 500)) > 0).all()


def test_cluster_igual_statsmodels():
    y, X, _, _ = _gaussiano()
    g = np.random.default_rng(1).integers(0, 20, len(y))
    glm = sm.GLM(y, X).fit()
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g}).cov_params()
    np.testing.assert_allclose(e.cov_cluster(glm, g).values, ref, rtol=1e-6)


def test_conley_beta_binomial_constante():
    rng = np.random.default_rng(5)
    N = 300
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    n = rng.integers(500, 5000, N)
    pi = 1 / (1 + np.exp(-(X @ [-0.2, 0.5])))
    y = rng.binomial(n, pi)
    res = M.BetaBinomial(y, X, n).fit(disp=0)
    lat, lon = rng.uniform(-30, 0, N), rng.uniform(-60, -40, N)
    cov = e.cov_conley(res, lat, lon, 1e-6)
    S = res.model.score_obs(res.params)
    Hi = np.linalg.inv(res.model.hessian(res.params))
    np.testing.assert_allclose(cov.values, Hi @ S.T @ S @ Hi, rtol=1e-6)
    assert cov.shape == (3, 3)
    assert np.isfinite(e.cov_conley(res, lat, lon, 300).values).all()


def test_autovetores_grade():
    geo = grade(12)
    ev, I = e.autovetores_esf(geo, k=10)
    V = ev.values
    N = len(geo)
    assert ev.shape == (N, 10) and list(ev.columns)[:2] == ["ev_001", "ev_002"]
    np.testing.assert_allclose(V.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(V.std(axis=0, ddof=0), 1, atol=1e-8)
    G = V.T @ V / N
    # ortogonais entre si, exceto autovalores repetidos (qualquer base do subespaço vale)
    lam = I.values
    for a in range(10):
        for b in range(a + 1, 10):
            if abs(lam[a] - lam[b]) > 1e-8:
                assert abs(G[a, b]) < 1e-6
    assert (np.diff(lam) <= 1e-12).all()
    assert (np.abs(V).argmax(axis=0) >= 0).all()
    assert (V[np.abs(V).argmax(axis=0), np.arange(10)] > 0).all()


def test_moran_dos_autovetores_igual_esda():
    import esda
    from libpysal.weights import Queen

    geo = grade(12)
    ev, I = e.autovetores_esf(geo, k=5)
    w = Queen.from_dataframe(geo, use_index=False)
    for j in range(5):
        mi = esda.Moran(ev.iloc[:, j].values, w, transformation="b", permutations=0)
        assert mi.I == pytest.approx(I.iloc[j], rel=1e-6)


def test_matriz_contiguidade_simetrica():
    C = e.matriz_contiguidade(grade(6))
    assert (C != C.T).nnz == 0 and C.diagonal().sum() == 0


def test_correlograma_iid_perto_de_zero():
    rng = np.random.default_rng(0)
    N = 600
    lat, lon = rng.uniform(-30, 0, N), rng.uniform(-60, -30, N)
    r = rng.normal(size=N)
    c = e.correlograma(r, lat, lon, [0, 500, 1500, 4000])
    assert (c["pares"] > 100).all()
    assert (c["correlacao"].abs() < 0.05).all()


def test_moran_residuos_reprodutivel():
    from libpysal.weights import Queen

    geo = grade(12)
    w = Queen.from_dataframe(geo, use_index=False)
    w.transform = "r"
    x = np.arange(144) % 12 + np.arange(144) // 12
    a = e.moran_residuos(x.astype(float), w, 99)
    assert a == e.moran_residuos(x.astype(float), w, 99)
    assert a["I"] > 0.5 and a["p_sim"] < 0.05
