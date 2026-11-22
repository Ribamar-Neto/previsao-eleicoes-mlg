"""Testes de src/selecao.py com dados sintéticos (2 variáveis ativas entre 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.especificacoes import CONTINUAS, MAPA_2026, REGIOES, UFS, Termos
from src.selecao import (
    Metodo,
    comparar_metodos,
    passo_a_passo,
    stepwise_hosmer_lemeshow,
    todos_os_subconjuntos,
)

CAND = ("nota_dep", "pct_dep_esq", "log_pop", "pct_urbana")
VERDADE = ("nota_dep", "pct_dep_esq")
TERMOS = Termos(vars=("lag", *CAND))


def _dados(n_mun: int, bb: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {mapa_col: rng.normal(size=n_mun) for k, mapa_col in MAPA_2026.items()
            if k in CONTINUAS}
    df = pd.DataFrame(cols)
    df[MAPA_2026["pib_pc"]] = np.exp(df[MAPA_2026["pib_pc"]])  # o log é aplicado em montar_X
    z = {k: df[MAPA_2026[k]].to_numpy() for k in CONTINUAS}
    eta = 0.3 + 0.5 * z["lag"] + 0.6 * z["nota_dep"] - 0.6 * z["pct_dep_esq"]
    p = expit(eta)
    n = rng.integers(200, 2000, n_mun)
    if bb:
        theta = 300.0
        p = rng.beta(p * theta, (1 - p) * theta)
    df[MAPA_2026["n"]] = n.astype(float)
    df[MAPA_2026["y"]] = rng.binomial(n, p).astype(float)
    df[MAPA_2026["regiao"]] = rng.choice(REGIOES + ["Sudeste"], n_mun)
    df[MAPA_2026["uf"]] = rng.choice(UFS, n_mun)
    df[MAPA_2026["bloco_pref"]] = rng.choice(["esquerda", "direita", "centro"], n_mun)
    return df


@pytest.fixture(scope="module")
def df_bin() -> pd.DataFrame:
    return _dados(600, bb=False, seed=1)


@pytest.fixture(scope="module")
def df_bb() -> pd.DataFrame:
    return _dados(600, bb=True, seed=2)


@pytest.mark.parametrize("metodo", ["forward", "backward", "both"])
def test_passo_a_passo_acha_verdade(df_bin: pd.DataFrame, metodo: Metodo) -> None:
    sel, trilha = passo_a_passo(df_bin, MAPA_2026, TERMOS, metodo, "bic",
                                candidatas=CAND, familia="binomial", n_jobs=2)
    assert set(sel) == set(VERDADE)
    assert list(trilha.columns) == ["passo", "acao", "variavel", "valor_criterio", "n_vars"]
    assert trilha.iloc[0]["acao"] == "inicio"
    assert trilha["valor_criterio"].is_monotonic_decreasing
    assert trilha.iloc[-1]["n_vars"] == 2


def test_passo_a_passo_betabin_forward(df_bb: pd.DataFrame) -> None:
    sel, _ = passo_a_passo(df_bb, MAPA_2026, TERMOS, "forward", "bic",
                           candidatas=CAND, familia="betabin", n_jobs=2)
    assert set(sel) == set(VERDADE)


def test_hosmer_lemeshow(df_bin: pd.DataFrame) -> None:
    sel, tab = stepwise_hosmer_lemeshow(df_bin, MAPA_2026, TERMOS, candidatas=CAND,
                                        p_entrada=0.05, p_saida=0.10,
                                        familia="binomial", n_jobs=2)
    assert set(sel) == set(VERDADE)
    assert {"etapa", "iteracao", "acao", "variavel", "delta_deviance", "gl", "pi"} <= set(tab.columns)
    entradas = tab[tab["acao"] == "entra"]
    assert entradas.iloc[0]["variavel"] in VERDADE
    assert (entradas["pi"] < 0.05).all()
    assert (entradas["gl"] == 1).all()


def test_hosmer_lemeshow_interacoes(df_bin: pd.DataFrame) -> None:
    sel, tab = stepwise_hosmer_lemeshow(df_bin, MAPA_2026, TERMOS, candidatas=CAND,
                                        familia="binomial", interacoes=True, n_jobs=2)
    assert set(VERDADE) <= set(sel)
    assert set(tab["etapa"]) <= {1, 2}
    assert not any(v.startswith("w_") for v in tab[tab["etapa"] == 1]["variavel"])


def test_todos_subconjuntos_e_comparacao(df_bb: pd.DataFrame) -> None:
    res = todos_os_subconjuntos(df_bb, MAPA_2026, TERMOS, candidatas=CAND, top=5, n_jobs=2)
    assert res.attrs["n_subconjuntos"] == 16
    assert {"variaveis", "n_vars", "qbic", "bic_bb", "aic_bb", "rank_qbic",
            "rank_bic_bb"} <= set(res.columns)
    assert set(res.iloc[0]["variaveis"]) == set(VERDADE)
    assert res["rank_bic_bb"].iloc[0] == 1

    comp = comparar_metodos({"todos": (res.iloc[0]["variaveis"],), "vazio": ((),)},
                            df_bb, MAPA_2026, TERMOS, n_jobs=2)
    linha = comp.set_index("metodo").loc["vazio"]
    assert set(linha["a_menos_vs_ref"]) == set(VERDADE)
    assert comp.set_index("metodo").loc["todos", "BIC_bb"] < linha["BIC_bb"]
