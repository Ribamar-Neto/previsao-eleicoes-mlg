import numpy as np
import pandas as pd
import pytest

from src import backtest as B
from src.especificacoes import MAPA_2026

PARQUET = "data/processed/municipios.parquet"


def test_mapas_tem_mesmos_papeis():
    assert set(B.MAPA_FIT_2022) == set(B.MAPA_PREV_2026) == set(MAPA_2026)
    assert set(B.mapas_entre_turnos()[0]) == set(MAPA_2026)


def test_fit_usa_so_informacao_anterior():
    f = B.MAPA_FIT_2022
    for k in ("lag", "nota_dep", "pct_dep_esq", "comparecimento", "bloco_pref"):
        assert any(a in f[k] for a in ("2018", "2020"))
    assert "delta_comp" not in B.VARS_EXANTE and "comparecimento" in B.VARS_EXANTE


def test_recalibrar_iguala_alvo():
    rng = np.random.default_rng(0)
    pi, n = rng.uniform(0.2, 0.8, 200), rng.integers(100, 5000, 200).astype(float)
    novo, d = B.recalibrar(pi, n, 0.45)
    assert np.sum(novo * n) / n.sum() == pytest.approx(0.45, abs=1e-8)
    assert np.all((novo > 0) & (novo < 1))
    _, d0 = B.recalibrar(pi, n, np.sum(pi * n) / n.sum())
    assert d0 == pytest.approx(0, abs=1e-6)


def test_erros_grupo_sinal_e_total():
    pi = np.array([0.5, 0.6, 0.4, 0.4])
    n = np.array([100.0, 100.0, 100.0, 100.0])
    y = np.array([50.0, 50.0, 40.0, 40.0])
    t = B.erros_grupo(pi, n, y, n, np.array(["a", "a", "b", "b"]))
    assert t.loc["a", "erro_pp"] == pytest.approx(5.0)
    assert t.loc["b", "erro_pp"] == pytest.approx(0.0)
    assert t.loc["Brasil", "real"] == pytest.approx(180 / 400)
    r = B.resumo_erros(pi, n, y, n, np.array(["a", "a", "b", "b"]), np.array(["x", "y", "z", "z"]))
    assert r["erro_nacional_pp"] == pytest.approx(2.5)
    assert r["mae_regiao_pp"] == pytest.approx(2.5)


def test_n_previsto():
    d = pd.DataFrame({"eleitorado_2026": [1000.0], "n_2t_2022": [700.0], "aptos_2t_2022": [1000.0]})
    assert B.n_previsto(d)[0] == pytest.approx(700.0)


def test_preparar_e_ajuste_rapido():
    try:
        df = pd.read_parquet(PARQUET)
    except FileNotFoundError:
        pytest.skip("parquet ausente")
    d = B.preparar(df)
    assert np.isfinite(d["logit_1t_2022"]).all() and (d["outros_1t_2026"] > -1e-9).all()
    ok = B.linhas_modeladas(d, B.MAPA_FIT_2022, B.MAPA_PREV_2026)
    amostra = d[ok].sample(1500, random_state=1).reset_index(drop=True)
    _, pi = B.ajustar_e_prever("quasi_logit", amostra, B.MAPA_FIT_2022, B.MAPA_PREV_2026)
    assert pi.shape == (1500,) and ((pi > 0) & (pi < 1)).all()
    # A elasticidade do lag do 2º turno anterior só faz sentido sem o 1º turno do próprio ciclo,
    # que absorve a informação do lag quando entra junto.
    sem_1t = tuple(v for v in B.VARS_EXANTE if v not in ("lag1t", "outros_1t"))
    aj_lag, _ = B.ajustar_e_prever("quasi_logit", amostra, B.MAPA_FIT_2022, B.MAPA_PREV_2026, sem_1t)
    assert 0.5 < B.coef_lag(aj_lag) < 1.2


def test_tabelar_uma_linha_por_variante_modelo_peso() -> None:
    rng = np.random.default_rng(1)
    k = 12
    regiao = np.array(["N", "S"] * (k // 2))
    uf = np.array(["A", "B", "C", "D"] * (k // 4))
    n_real = rng.integers(100, 500, k).astype(float)
    y_real = np.round(n_real * rng.uniform(0.3, 0.7, k))
    pi_por = {("v1", "m1"): rng.uniform(0.3, 0.7, k), ("v2", "m2"): rng.uniform(0.3, 0.7, k)}
    res, reg, ufs = B.tabelar(pi_por, n_real * 1.1, n_real, y_real, regiao, uf)
    assert len(res) == 4 and set(res["peso"]) == {"n_prev", "n_real"}
    assert len(reg) == 4 * 2 and len(ufs) == 4 * 4
    assert "Brasil" not in set(reg["grupo"]) | set(ufs["grupo"])


def test_derivadas_do_1t_seguem_as_formulas_e_protegem_denominador_zero() -> None:
    df = pd.DataFrame(
        {
            "votos_pt_1t_2022": [30.0, 0.0, 10.0],
            "votos_bolsonaro_1t_2022": [10.0, 0.0, 10.0],
            "validos_1t_2022": [80.0, 50.0, 0.0],
            "votos_pt_1t_2026": [30.0, 0.0, 10.0],
            "votos_bolsonaro_1t_2026": [10.0, 0.0, 10.0],
            "validos_1t_2026": [80.0, 50.0, 0.0],
        }
    )
    d = B.preparar(df)
    assert d["logit_1t_2022"][0] == pytest.approx(np.log(3.0), abs=1e-5)
    assert d["outros_1t_2022"][0] == pytest.approx(1 - 40 / 80)
    assert d["logit_1t_2022"][1] == pytest.approx(0.0) and d["outros_1t_2022"][1] == pytest.approx(
        1.0
    )  # sem finalistas
    assert d["outros_1t_2022"][2] == pytest.approx(0.0)  # sem válidos
    assert np.isfinite(d[["logit_1t_2026", "outros_1t_2026"]].to_numpy()).all()


def test_mapa_de_2022_do_1t_le_so_colunas_de_2022() -> None:
    assert B.MAPA_FIT_2022["lag1t"].endswith("_2022") and B.MAPA_FIT_2022["outros_1t"].endswith("_2022")
    assert B.MAPA_PREV_2026["lag1t"].endswith("_2026") and B.MAPA_PREV_2026["outros_1t"].endswith("_2026")
