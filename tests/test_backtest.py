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
    aj, pi = B.ajustar_e_prever("quasi_logit", amostra, B.MAPA_FIT_2022, B.MAPA_PREV_2026)
    assert pi.shape == (1500,) and ((pi > 0) & (pi < 1)).all()
    assert 0.5 < B.coef_lag(aj) < 1.2
