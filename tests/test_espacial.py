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


def test_defasagem_espacial_media_dos_vizinhos():
    from libpysal.weights import Queen
    w = Queen.from_dataframe(grade(3), use_index=False)
    w.transform = "r"
    x = np.arange(9.0)
    wx = e.defasagem_espacial(x, w)
    # centro (4): 8 vizinhos queen; canto (0): vizinhos 1, 3, 4
    assert wx[4] == pytest.approx(np.mean([0, 1, 2, 3, 5, 6, 7, 8]))
    assert wx[0] == pytest.approx(np.mean([1, 3, 4]))


def test_defasagem_ilha_recebe_vizinho_mais_proximo():
    geo = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)])
    w = e.pesos_queen(geo)
    wx = e.defasagem_espacial(np.array([1.0, 2.0, 10.0]), w)
    assert wx[2] == pytest.approx(2.0)  # ilha -> KNN k=1 (célula 1)


def test_adicionar_slx_sem_mutar():
    import pandas as pd
    from libpysal.weights import Queen
    w = Queen.from_dataframe(grade(3), use_index=False)
    w.transform = "r"
    df = pd.DataFrame({"a": np.arange(9.0)})
    mapa = {"lag": "a"}
    df2, mapa2 = e.adicionar_slx(df, mapa, ["lag"], w)
    assert list(df.columns) == ["a"] and mapa == {"lag": "a"}
    assert mapa2["w_lag"] == "w_a"
    assert df2["w_a"].iloc[0] == pytest.approx(np.mean([1, 3, 4]))


def test_adicionar_slx_pib_na_escala_do_modelo():
    import pandas as pd
    from libpysal.weights import Queen
    w = Queen.from_dataframe(grade(3), use_index=False)
    w.transform = "r"
    df = pd.DataFrame({"p": np.exp(np.arange(9.0))})
    _, mapa2 = e.adicionar_slx(df, {"pib_pc": "p"}, ["pib_pc"], w)
    df2, _ = e.adicionar_slx(df, {"pib_pc": "p"}, ["pib_pc"], w)
    assert df2[mapa2["w_pib_pc"]].iloc[0] == pytest.approx(np.mean([1, 3, 4]))  # W·log(pib)


@pytest.mark.parametrize("gl_den", [None, 40])
def test_wald_robusto_igual_statsmodels(gl_den):
    res, _, _ = simular()
    rng = np.random.default_rng(1)
    A = rng.normal(size=(2, 2))
    cov = A @ A.T * 1e-3 + np.eye(2) * 1e-3
    est, gl, p = e.wald_robusto(res.params, cov, [0, 1], gl_den)
    R = np.eye(2)
    b = res.params
    ref = float(b @ np.linalg.solve(cov, b))
    if gl_den is None:
        w = res.wald_test(R, cov_p=cov, scalar=True)
        assert est == pytest.approx(float(w.statistic)) and p == pytest.approx(float(w.pvalue))
    else:
        w = res.wald_test(R, cov_p=cov, scalar=True, use_f=True)
        assert est == pytest.approx(ref / 2) and gl == 2


def _campo(N=1500, seed=5):
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-30, 0, N), rng.uniform(-60, -30, N)
    return rng, lat, lon, rng.integers(0, 20, N)


def test_krigagem_nucleo_zero_alem_do_corte():
    lat = np.array([0.0, 0.0, 0.0])
    lon = np.array([0.0, 0.5, 5.0])   # ~55 km e ~555 km do ponto 0
    _rho, k = e.krigagem_residuos(np.array([1.0, 3.0]), lat, lon, [1, 2], [0], np.array([0, 1, 2]), 200.0)
    np.testing.assert_allclose(k, [1.0])        # só o vizinho a 55 km pesa
    _, k0 = e.krigagem_residuos(np.array([1.0, 3.0]), lat, lon, [1, 2], [0], np.array([0, 1, 2]), 10.0)
    assert k0[0] == 0.0                         # sem vizinho no corte


def test_krigagem_rho_zero_em_ruido_branco():
    rng, lat, lon, g = _campo()
    tr, te = np.arange(1200), np.arange(1200, 1500)
    rho, _ = e.krigagem_residuos(rng.normal(size=1200), lat, lon, tr, te, g, 300.0)
    assert rho < 0.15


def test_krigagem_rho_perto_de_um_em_campo_suave():
    _, lat, lon, g = _campo()
    campo = np.sin(lat / 10) + np.cos(lon / 10)
    tr, te = np.arange(1200), np.arange(1200, 1500)
    rho, k = e.krigagem_residuos(campo[tr], lat, lon, tr, te, g, 300.0)
    assert rho > 0.9
    assert np.corrcoef(k, campo[te])[0, 1] > 0.95


def test_pesos_dentro_e_entre_somam_um_por_linha():
    geo = grade(8)
    w = e.pesos_queen(geo)
    g = (np.arange(64) // 8) // 4 * 2 + (np.arange(64) % 8) // 4
    for dentro in (True, False):
        W, mant = e._restringir_pesos(w, g, dentro)
        np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)
        assert W.shape == (len(mant), len(mant))


def test_moran_sem_media_do_grupo_some_com_choque_de_grupo():
    geo = grade(12)
    w = e.pesos_queen(geo)
    idx = np.arange(144)
    g = (idx // 12) // 4 * 3 + (idx % 12) // 4          # 9 blocos 4x4
    rng = np.random.default_rng(0)
    r = rng.normal(size=9)[g] * 3 + rng.normal(size=144) * 0.3
    m = e.moran_decomposto(r, w, g)
    assert m["completo"] > 0.5
    assert abs(m["completo_sem_media_uf"]) < 0.15
    assert m["dentro_uf"] > m["entre_uf"]


def _simular_car(lado=30, tau=0.5, lam=0.8, sigma=0.3, seed=1):
    from libpysal.weights import lat2W

    w = lat2W(lado, lado)
    A = e._adjacencia_binaria(w)
    Q = (e._precisao_leroux(A, lam) / tau).toarray()
    rng = np.random.default_rng(seed)
    u = np.linalg.solve(np.linalg.cholesky(Q).T, rng.normal(size=lado * lado))
    return w, u + rng.normal(scale=sigma, size=lado * lado)


def test_car_recupera_parametros():
    w, r = _simular_car()
    idx = np.arange(len(r))
    p, _ = e.car_residuos(r, w, idx, idx[:0])
    assert 0.3 < p["lam"] <= 0.99
    assert 0.1 < p["tau"] < 2.0
    assert 0.03 < p["sigma2"] < 0.3


def test_car_sem_vazamento_do_teste():
    w, r = _simular_car(lado=20)
    idx = np.arange(len(r))
    te, tr = idx[:60], idx[60:]
    p1, a = e.car_residuos(r[tr], w, tr, te)
    r2 = r.copy()
    r2[te] += 50.0
    p2, b = e.car_residuos(r2[tr], w, tr, te)
    np.testing.assert_allclose(a, b)
    assert p1 == p2


def test_car_no_isolado_fica_finito():
    from libpysal.weights import W

    w = W({0: [1], 1: [0, 2], 2: [1], 3: []})
    r = np.array([0.5, 0.4, 0.6])
    _, prev = e.car_residuos(r, w, np.array([0, 1, 2]), np.array([3]))
    assert np.isfinite(prev).all()
