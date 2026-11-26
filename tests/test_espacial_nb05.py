"""Helpers das Partes 13 a 17 do `02-MLG` em src/espacial.py, caracterizados contra cópias do código antigo do notebook (então `05-modelo-final`)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm
from libpysal.weights import KNN
from scipy.special import expit, logit

from src import espacial as S

FAIXAS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]


def _dados(N=300, seed=5):
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-30, 0, N), rng.uniform(-60, -40, N)
    X = pd.DataFrame({"const": 1.0, "x": rng.normal(size=N), "disp_reg_a": rng.normal(size=N)})
    n = rng.integers(200, 3000, N)
    y = rng.binomial(n, expit(-0.2 + 0.5 * X["x"].to_numpy()))
    uf = rng.choice(["A", "B", "C", "D", "E"], N)
    res = sm.GLM(np.column_stack([y, n - y]), X.to_numpy(), family=sm.families.Binomial()).fit()
    return res, list(X.columns), lat, lon, uf, rng.normal(size=N)


# Cópia do notebook antes da mudança (as globais viraram argumentos)
def _corte_de_antigo(cg):
    ab = cg[cg.correlacao < 0.05]
    return float(ab.iloc[0]["ini_km"]) if len(ab) else float(cg.iloc[-1]["fim_km"])


def _passo_wald_antigo(passo, o, sel, AJ14, R14, lat, lon, uf, UFS_):
    a = AJ14[o]
    mod = a.modelo
    nm = list(a.nomes)
    ind = [i for i, s in enumerate(nm) if sel(s)]
    corte = _corte_de_antigo(S.correlograma(R14[o], lat, lon, FAIXAS))
    covs = {
        "conley_05": S.cov_conley(mod, lat, lon, corte * 0.5),
        "conley_1": S.cov_conley(mod, lat, lon, corte),
        "conley_2": S.cov_conley(mod, lat, lon, corte * 2),
        "cluster_uf": S.cov_cluster(mod, uf, correcao=True),
    }
    linhas = []
    for tag, cv in covs.items():
        for tipo, gd in ([("chi2", None)] if tag == "conley_1" else []) + [("F", len(UFS_) - 1)]:
            est, gl, p = S.wald_robusto(mod.params, cv, ind, gl_den=gd)
            linhas.append(
                dict(  # noqa: C408
                    passo=passo,
                    ajuste=o,
                    termos=";".join(nm[i] for i in ind),
                    covariancia=tag,
                    corte_km=corte * (0.5 if tag == "conley_05" else 2 if tag == "conley_2" else 1)
                    if tag != "cluster_uf"
                    else np.nan,
                    teste=tipo,
                    estat=est,
                    gl=gl,
                    gl_den=gd if gd else np.nan,
                    p=p,
                )
            )
    return pd.DataFrame(linhas), corte


def test_corte_de_igual_ao_antigo_e_casos_de_borda():
    cg = pd.DataFrame({"ini_km": [0, 10, 25], "fim_km": [10, 25, 50], "correlacao": [0.3, 0.04, 0.01]})
    assert S.corte_de(cg) == _corte_de_antigo(cg) == 10.0
    sem = cg.assign(correlacao=[0.3, 0.2, 0.1])
    assert S.corte_de(sem) == _corte_de_antigo(sem) == 50.0


def test_passo_wald_igual_ao_antigo():
    res, nomes, lat, lon, uf, r = _dados()
    aj = SimpleNamespace(modelo=res, nomes=nomes)

    def sel(s):
        return s.startswith("disp_reg_")

    novo, corte = S.passo_wald("A", "regiao", aj, r, sel, lat, lon, uf, FAIXAS)
    ant, corte_a = _passo_wald_antigo("A", "regiao", sel, {"regiao": aj}, {"regiao": r}, lat, lon, uf, sorted(set(uf)))
    assert corte == corte_a
    pd.testing.assert_frame_equal(novo, ant)


def test_ep_e_raiz_da_diagonal():
    np.testing.assert_array_equal(S.ep(np.array([[4.0, 1.0], [1.0, 9.0]])), [2.0, 3.0])


def _pesos(N, seed, k=4):
    rng = np.random.default_rng(seed)
    w = KNN.from_array(rng.uniform(size=(N, 2)), k=k)
    w.transform = "r"
    return w, rng


def test_moran_z_e_moran_oos_iguais_ao_antigo():
    w, rng = _pesos(80, 1)
    n = rng.integers(100, 900, 80).astype(float)
    pi = rng.uniform(0.3, 0.7, 80)
    y = rng.binomial(n.astype(int), pi).astype(float)
    th = np.full(80, 0.01)
    lz = logit((y + 0.5) / (n + 1))
    pi2 = np.clip(pi + 0.01, 0.01, 0.99)
    ant_z = S.moran_residuos(S.residuo_pearson(y, n, pi2, th)[0], w, permutacoes=999, semente=0)["I"]
    ant_o = S.moran_residuos(lz - logit(pi2), w, permutacoes=999, semente=0)["I"]
    assert S.moran_z(y, n, pi2, th, w) == ant_z
    assert S.moran_oos(lz, pi2, w) == ant_o


def test_moran_residuos_devolve_z_sim_e_esperado_do_mesmo_objeto():
    import esda

    w, rng = _pesos(60, 2, k=3)
    x = rng.normal(size=60)
    m = S.moran_residuos(x, w, 99, 0)
    np.random.seed(0)
    mi = esda.Moran(x, w, permutations=99)
    assert (m["I"], m["p_sim"], m["z_sim"], m["esperado"]) == (mi.I, mi.p_sim, mi.z_sim, mi.EI)


def test_substituir_linha_csv_idempotente_e_na_posicao(tmp_path):
    arq = tmp_path / "m.csv"
    pd.DataFrame({"modelo": ["bb_selecao"], "I": [0.1 + 0.2], "p_sim": [0.001]}).to_csv(arq, index=False)
    S.substituir_linha_csv(arq, {"modelo": "bb_slx", "I": 1 / 3, "p_sim": 0.001})
    uma = arq.read_bytes()
    S.substituir_linha_csv(arq, {"modelo": "bb_slx", "I": 1 / 3, "p_sim": 0.001})
    assert arq.read_bytes() == uma
    d = pd.read_csv(arq, float_precision="round_trip")
    assert list(d.modelo) == ["bb_selecao", "bb_slx"] and d.I[0] == 0.1 + 0.2 and d.I[1] == 1 / 3


def test_linha_sens():
    assert S.linha_sens("krig", "rho_media", 0.5, 0.6) == {
        "modelo": "krig",
        "metrica": "rho_media",
        "atual": 0.5,
        "ponderado": 0.6,
        "criterio_atual": None,
        "criterio_ponderado": None,
    }


class _Cand:
    """Candidato mínimo: logit linear em x, por mínimos quadrados."""

    def ajustar(self, d, mapa):
        X = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
        return np.linalg.lstsq(X, logit((d["y"] + 0.5) / (d["n"] + 1)).to_numpy(), rcond=None)[0]

    def prever(self, b, d, mapa):
        return expit(b[0] + b[1] * d["x"].to_numpy())


def _folds_e_dados(N=120, seed=4):
    w, rng = _pesos(N, seed)
    d = pd.DataFrame({"x": rng.normal(size=N), "n": rng.integers(100, 900, N)})
    d["y"] = rng.binomial(d.n, expit(0.3 * d.x))
    uf = rng.choice(list("ABCD"), N)
    folds = [(np.flatnonzero(uf != g), np.flatnonzero(uf == g)) for g in "ABCD"]
    return d, uf, folds, rng.uniform(-30, 0, N), rng.uniform(-60, -40, N), w


def test_rodar_extensao_krig_e_car_iguais_ao_antigo_e_cache(tmp_path):
    d, uf, folds, lat, lon, w = _folds_e_dados()
    cand, lz = _Cand(), logit((d.y.to_numpy() + 0.5) / (d.n.to_numpy() + 1))
    S.definir_contexto_dobras(cand=cand, df=d, mapa=None, lz=lz, lat=lat, lon=lon, uf=uf, corte=300.0, w=w)
    pi, P = S.rodar_extensao("krig", folds, tmp_path / "k.parquet", False, max_workers=2)
    pi_a, rho_a, pic_a = (np.full(len(d), np.nan) for _ in range(3))
    Pc_a = np.full((len(d), 3), np.nan)
    for tr, te in folds:  # cópia de _krig_fold e _car_fold do notebook
        a = cand.ajustar(d.iloc[tr], None)
        r_tr = lz[tr] - logit(np.asarray(cand.prever(a, d.iloc[tr], None), float))
        pi_te = np.asarray(cand.prever(a, d.iloc[te], None), float)
        rho, k = S.krigagem_residuos(r_tr, lat, lon, tr, te, uf, 300.0)
        pi_a[te], rho_a[te] = expit(logit(pi_te) + rho * k), rho
        par, u = S.car_residuos(r_tr, w, tr, te)
        pic_a[te], Pc_a[te] = expit(logit(pi_te) + u), [par["tau"], par["lam"], par["sigma2"]]
    np.testing.assert_array_equal(pi, pi_a)
    np.testing.assert_array_equal(P[:, 0], rho_a)
    pi2, P2 = S.rodar_extensao("krig", folds, tmp_path / "k.parquet", False, max_workers=2)  # lê o cache
    np.testing.assert_array_equal(pi2, pi)
    np.testing.assert_array_equal(P2, P)
    pic, Pc = S.rodar_extensao("car", folds, tmp_path / "c.parquet", False, max_workers=2)
    np.testing.assert_array_equal(pic, pic_a)
    np.testing.assert_array_equal(Pc, Pc_a)
    assert list(pd.read_parquet(tmp_path / "c.parquet").columns) == ["pi", "tau", "lam", "sigma2"]
    assert list(pd.read_parquet(tmp_path / "k.parquet").columns) == ["pi", "rho"]


def test_dobra_ponderada_igual_ao_antigo(monkeypatch):
    from src import especificacoes as E

    d, uf, folds, lat, lon, w = _folds_e_dados()
    cand, th = _Cand(), 0.01
    monkeypatch.setattr(E, "prever_theta", lambda a, dd, mapa=None: np.full(len(dd), th))
    y, n = d.y.to_numpy(float), d.n.to_numpy(float)
    S.definir_contexto_dobras(cand=cand, df=d, mapa=None, y=y, n=n, lat=lat, lon=lon, uf=uf, corte=300.0, w=w)
    tr, te = folds[0]
    r = S.dobra_ponderada((tr, te))
    a = cand.ajustar(d.iloc[tr], None)  # cópia de _pond_fold do notebook
    z_tr, _ = S.residuo_pearson(y[tr], n[tr], cand.prever(a, d.iloc[tr], None), np.full(len(tr), th))
    pi_te = cand.prever(a, d.iloc[te], None)
    sd_te = S.sd_betabinomial(pi_te, n[te], np.full(len(te), th))
    rho, k = S.krigagem_residuos(z_tr, lat, lon, tr, te, uf, 300.0, sd_teste=sd_te)
    par, u = S.car_residuos(z_tr, w, tr, te, sd_teste=sd_te)
    assert r["rho"] == rho and r["lam"] == par["lam"]
    np.testing.assert_array_equal(r["krig"], np.clip(pi_te + k, 1e-6, 1 - 1e-6))
    np.testing.assert_array_equal(r["car"], np.clip(pi_te + u, 1e-6, 1 - 1e-6))
    np.testing.assert_array_equal(r["pi"], pi_te)
