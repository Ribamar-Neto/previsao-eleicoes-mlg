"""Testes de src/selecao.py com dados sintéticos (2 variáveis ativas entre 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.especificacoes import CONTINUAS, MAPA_2026, REGIOES, UFS, Termos
from src.selecao import (
    Metodo,
    ResultadoHL,
    comparar_metodos,
    niveis_largos,
    passo_a_passo,
    stepwise_hosmer_lemeshow,
    todos_os_subconjuntos,
)

CAND = ("nota_dep", "pct_dep_esq", "log_pop", "pct_urbana")
VERDADE = ("nota_dep", "pct_dep_esq")
TERMOS = Termos(vars=("lag", *CAND))


def _dados(n_mun: int, bb: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {mapa_col: rng.normal(size=n_mun) for k, mapa_col in MAPA_2026.items() if k in CONTINUAS}
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
    sel, trilha = passo_a_passo(df_bin, MAPA_2026, TERMOS, metodo, "bic", candidatas=CAND, familia="binomial", n_jobs=2)
    assert set(sel) == set(VERDADE)
    assert list(trilha.columns) == ["passo", "acao", "variavel", "valor_criterio", "n_vars"]
    assert trilha.iloc[0]["acao"] == "inicio"
    assert trilha["valor_criterio"].is_monotonic_decreasing
    assert trilha.iloc[-1]["n_vars"] == 2


def test_passo_a_passo_betabin_forward(df_bb: pd.DataFrame) -> None:
    sel, _ = passo_a_passo(df_bb, MAPA_2026, TERMOS, "forward", "bic", candidatas=CAND, familia="betabin", n_jobs=2)
    assert set(sel) == set(VERDADE)


def test_hosmer_lemeshow(df_bin: pd.DataFrame) -> None:
    res = stepwise_hosmer_lemeshow(
        df_bin, MAPA_2026, TERMOS, candidatas=CAND, p_entrada=0.05, p_saida=0.10, familia="binomial", n_jobs=2
    )
    assert isinstance(res, ResultadoHL)
    assert set(res.variaveis) == set(VERDADE)
    tab = res.passos
    assert {"etapa", "iteracao", "acao", "variavel", "delta_deviance", "gl", "pi"} <= set(tab.columns)
    entradas = tab[tab["acao"] == "entra"]
    assert entradas.iloc[0]["variavel"] in VERDADE
    assert (entradas["pi"] < 0.05).all()
    assert (entradas["gl"] == 1).all()


def test_hosmer_lemeshow_interacoes(df_bin: pd.DataFrame) -> None:
    res = stepwise_hosmer_lemeshow(
        df_bin, MAPA_2026, TERMOS, candidatas=CAND, familia="binomial", interacoes=True, n_jobs=2
    )
    assert set(VERDADE) <= set(res.variaveis)
    assert set(res.passos["etapa"]) <= {1, 2}
    assert not any(v.startswith("w_") for v in res.passos[res.passos["etapa"] == 1]["variavel"])


def test_niveis_uma_linha_por_candidata_por_iteracao(df_bin: pd.DataFrame) -> None:
    res = stepwise_hosmer_lemeshow(
        df_bin, MAPA_2026, TERMOS, candidatas=CAND, p_entrada=0.05, p_saida=0.10, familia="binomial", n_jobs=2
    )
    nv = res.niveis
    assert {"etapa", "iteracao", "fase", "variavel", "delta_deviance", "gl", "pi", "escolhida"} <= set(nv.columns)
    assert not nv.duplicated(["etapa", "iteracao", "fase", "variavel"]).any()
    primeira = nv[(nv["iteracao"] == 1) & (nv["fase"] == "entrada")]
    assert set(primeira["variavel"]) == set(CAND)
    for (_, fase), g in nv[nv["fase"] == "entrada"].groupby(["iteracao", "fase"]):
        esc = g[g["escolhida"]]
        if len(esc):
            assert esc.iloc[0]["pi"] == g["pi"].min()
    saidas = nv[nv["fase"] == "saida"]
    for _, g in saidas.groupby("iteracao"):
        esc = g[g["escolhida"]]
        if len(esc):
            assert esc.iloc[0]["pi"] == g["pi"].max()
    # as entradas escolhidas são as da trilha
    assert list(nv[(nv["fase"] == "entrada") & nv["escolhida"]]["variavel"]) == list(
        res.passos[res.passos["acao"] == "entra"]["variavel"]
    )
    larga = niveis_largos(nv, 1)
    assert larga.index.name == "iteracao"
    assert set(CAND) <= set(larga.columns)


def test_wald_remove_ruido_forcado(df_bin: pd.DataFrame) -> None:
    sem = stepwise_hosmer_lemeshow(
        df_bin,
        MAPA_2026,
        TERMOS,
        candidatas=CAND,
        p_entrada=1.0,
        p_saida=1.0,
        familia="binomial",
        n_jobs=2,
        alfa_wald=None,
    )
    assert set(sem.variaveis) == set(CAND)
    assert sem.wald.empty
    com = stepwise_hosmer_lemeshow(
        df_bin,
        MAPA_2026,
        TERMOS,
        candidatas=CAND,
        p_entrada=1.0,
        p_saida=1.0,
        familia="binomial",
        n_jobs=2,
        alfa_wald=0.05,
    )
    assert set(VERDADE) <= set(com.variaveis)
    assert set(CAND) - set(VERDADE) - set(com.variaveis)  # ruído caiu
    w = com.wald
    assert {"rodada", "variavel", "coef", "ep", "z", "p", "acao"} <= set(w.columns)
    removidas = w[w["acao"] == "removida"]
    assert (removidas["p"] > 0.05).all()
    assert not set(removidas["variavel"]) & set(VERDADE)


def test_wald_desligado_reproduz_selecao_antiga(df_bin: pd.DataFrame) -> None:
    res = stepwise_hosmer_lemeshow(
        df_bin,
        MAPA_2026,
        TERMOS,
        candidatas=CAND,
        p_entrada=0.05,
        p_saida=0.10,
        familia="binomial",
        n_jobs=2,
        alfa_wald=None,
    )
    assert set(res.variaveis) == set(VERDADE)
    assert res.wald.empty
    ent = res.passos[res.passos["acao"] == "entra"]["variavel"]
    assert set(ent) == set(res.variaveis)


def test_todos_subconjuntos_e_comparacao(df_bb: pd.DataFrame) -> None:
    res = todos_os_subconjuntos(df_bb, MAPA_2026, TERMOS, candidatas=CAND, top=5, n_jobs=2)
    assert res.attrs["n_subconjuntos"] == 16
    assert {"variaveis", "n_vars", "qbic", "bic_bb", "aic_bb", "rank_qbic", "rank_bic_bb"} <= set(res.columns)
    assert set(res.iloc[0]["variaveis"]) == set(VERDADE)
    assert res["rank_bic_bb"].iloc[0] == 1

    comp = comparar_metodos({"todos": (res.iloc[0]["variaveis"],), "vazio": ((),)}, df_bb, MAPA_2026, TERMOS, n_jobs=2)
    linha = comp.set_index("metodo").loc["vazio"]
    assert set(linha["a_menos_vs_ref"]) == set(VERDADE)
    assert comp.set_index("metodo").loc["todos", "BIC_bb"] < linha["BIC_bb"]


# --- tabelas descritivas, teste univariado e Mantel-Haenszel -----------------
def _fator_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    m = 900
    g = rng.choice(["a", "b", "c"], m)
    p = expit(np.array([{"a": 0.0, "b": 0.5, "c": -0.4}[v] for v in g]))
    n = rng.integers(200, 2000, m)
    y = rng.binomial(n, rng.beta(p * 200, (1 - p) * 200))
    return pd.DataFrame(
        {
            "g": g,
            "y": y.astype(float),
            "n": n.astype(float),
            "regiao": rng.choice(["R1", "R2"], m),
            "bin": rng.choice([True, False, None], m, p=[0.2, 0.7, 0.1]),
        }
    )


def test_tabela_descritiva_woolf_a_mao() -> None:
    from src.selecao import tabela_descritiva

    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "y": [30.0, 20.0, 60.0, 40.0], "n": [100.0, 100.0, 100.0, 100.0]})
    t = tabela_descritiva(df, "g", "a", "y", "n")
    assert t.loc["a", "psi"] == 1.0 and np.isnan(t.loc["a", "psi_inf95"])
    # a: 50 sim / 150 não; b: 100 sim / 100 não => psi = (100/100)/(50/150) = 3
    assert t.loc["b", "psi"] == pytest.approx(3.0)
    se = np.sqrt(1 / 100 + 1 / 100 + 1 / 50 + 1 / 150)
    assert t.loc["b", "psi_inf95"] == pytest.approx(3.0 * np.exp(-1.96 * se))
    assert t.loc["b", "psi_sup95"] == pytest.approx(3.0 * np.exp(1.96 * se))
    assert t.loc["b", "pct_lula"] == pytest.approx(50.0)
    assert t.loc["b", "logit_emp"] == pytest.approx(0.0)
    assert t.loc["a", "mediana_pct"] == pytest.approx(25.0)
    assert t.loc["a", "municipios"] == 2 and t.loc["a", "soma_y"] == 50.0


def test_tabela_descritiva_ausente_vira_nivel_na() -> None:
    from src.selecao import tabela_descritiva

    df = pd.DataFrame({"g": ["a", None, "b"], "y": [1.0, 2.0, 3.0], "n": [10.0, 10.0, 10.0]})
    assert "NA" in tabela_descritiva(df, "g", "a", "y", "n").index


def test_teste_univariado_detecta_efeito_do_fator() -> None:
    from src.selecao import teste_univariado

    df = _fator_df()
    mapa = {"y": "y", "n": "n", "g": "g"}
    r = teste_univariado(df, mapa, "g", ref="a")
    assert r["gl"] == 2 and r["p"] < 1e-6 and r["delta"] > 0 and np.isfinite(r["loglik"])


def test_teste_univariado_binaria_com_none_como_falso() -> None:
    from src.selecao import teste_univariado

    df = _fator_df()
    r = teste_univariado(df, {"y": "y", "n": "n", "bin": "bin"}, "bin", ref="False")
    assert r["gl"] == 1 and 0 <= r["p"] <= 1


def test_mantel_haenszel_ordem_e_homogeneidade() -> None:
    from src.selecao import mantel_haenszel

    df = pd.DataFrame(
        {
            "b": [True, False, True, False] * 2,
            "regiao": ["R1"] * 4 + ["R2"] * 4,
            "y": [60.0, 40.0, 30.0, 20.0, 60.0, 40.0, 30.0, 20.0],
            "n": [100.0] * 8,
        }
    )
    r = mantel_haenszel(df, "b", "regiao", "y", "n")
    # bruto: expostos 180 sim/220 não contra não expostos 120 sim/280 não
    sim_t, nao_t = 60 + 30 + 60 + 30, 40 + 70 + 40 + 70
    sim_f, nao_f = 40 + 20 + 40 + 20, 60 + 80 + 60 + 80
    assert r["or_bruto"] == pytest.approx((sim_t / nao_t) / (sim_f / nao_f))
    # estratos idênticos: OR de Mantel-Haenszel = OR do estrato = (90*140)/(110*60)
    assert r["or_mh"] == pytest.approx((90 * 140) / (110 * 60))
    assert r["or_mh_inf95"] < r["or_mh"] < r["or_mh_sup95"]
    assert r["p_homogeneidade"] == pytest.approx(1.0)  # estratos idênticos


def test_mantel_haenszel_ignora_estrato_sem_expostos() -> None:
    from src.selecao import mantel_haenszel

    base = pd.DataFrame(
        {
            "b": [True, False, True, False] * 2,
            "regiao": ["R1"] * 4 + ["R2"] * 4,
            "y": [60.0, 40.0, 30.0, 20.0, 60.0, 40.0, 30.0, 20.0],
            "n": [100.0] * 8,
        }
    )
    sem_expostos = pd.DataFrame({"b": [False, False], "regiao": ["R3", "R3"], "y": [50.0, 10.0], "n": [100.0, 100.0]})
    r0 = mantel_haenszel(base, "b", "regiao", "y", "n")
    r = mantel_haenszel(pd.concat([base, sem_expostos], ignore_index=True), "b", "regiao", "y", "n")
    assert r["or_mh"] == pytest.approx(r0["or_mh"])
    assert r["p_homogeneidade"] == pytest.approx(1.0)
    assert r["estratos_usados"] == 2


def test_mantel_haenszel_none_conta_como_falso() -> None:
    from src.selecao import mantel_haenszel

    df = pd.DataFrame(
        {
            "b": [True, None, True, False],
            "regiao": ["R1", "R1", "R2", "R2"],
            "y": [60.0, 40.0, 30.0, 20.0],
            "n": [100.0] * 4,
        }
    )
    df2 = df.assign(b=[True, False, True, False])
    assert mantel_haenszel(df, "b", "regiao", "y", "n")["or_bruto"] == pytest.approx(
        mantel_haenszel(df2, "b", "regiao", "y", "n")["or_bruto"]
    )
