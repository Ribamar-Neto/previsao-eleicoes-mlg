"""Testes do diagnóstico da apuração parcial (dados sintéticos)."""

import numpy as np
import pandas as pd
import pytest

from src import apuracao as A
from src import apuracao_diagnostico as AD

LIG = "logit"


def _cenario() -> tuple[A.ParcialModelo, pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    idx = np.arange(1, 9)
    base = pd.DataFrame(
        {
            "uf": ["AA", "AA", "BB", "BB", "CC", "CC", "DD", "DD"],
            "regiao": ["N", "N", "N", "N", "S", "S", "S", "S"],
            "y_c": 40.0,
            "n_c": 100.0,
            "n_u": [900.0, 500.0, 700.0, 300.0, 800.0, 600.0, 400.0, 200.0],
            "n_final": 0.0,
            "no_modelo": True,
        },
        index=idx,
    )
    base["n_final"] = base["n_c"] + base["n_u"]
    X = np.column_stack([np.ones(8), rng.normal(size=8)])
    pm = A.ParcialModelo(
        pi=np.full(8, 0.5),
        beta=np.array([0.0, 0.2]),
        cov=np.array([[0.02, 0.0], [0.0, 0.01]]),
        X=X,
        theta=np.full(8, 30.0),
        n_treino=8,
    )
    pi_uf = pd.Series(0.5, index=["AA", "BB", "CC", "DD"])
    return pm, base, base, pi_uf


def _var(B: int = 4000) -> pd.DataFrame:
    pm, base_m, base, pi_uf = _cenario()
    fk = np.exp(np.random.default_rng(5).normal(0, 0.05, size=(B, len(base))))
    return AD.variancias_componentes(pm, base_m, base, pi_uf, LIG, 0.1, B, 7, fk)


def test_piso_e_a_variancia_binomial() -> None:
    pm, _, base, _ = _cenario()
    var = _var()
    # com tudo desligado, π é fixo em expit(Xβ̂) e a variância é Σ n π(1-π) / den²
    pi = 1 / (1 + np.exp(-pm.X @ pm.beta))
    n = base["n_u"].to_numpy()
    m = (base["regiao"] == "N").to_numpy()
    esperado = (n[m] * pi[m] * (1 - pi[m])).sum() / (base.loc[m, "n_c"] + base.loc[m, "n_u"]).sum() ** 2
    assert var.loc["N", "var_piso"] == pytest.approx(esperado, rel=0.15)


def test_cada_componente_acrescenta_variancia() -> None:
    d = AD.decompor(_var())
    assert (d[["inc_beta", "inc_uf", "inc_theta"]] > 0).all().all()
    assert d.loc["Brasil", "var_tudo"] > d.loc["Brasil", "var_piso"]
    assert d.loc["Brasil", "maior_componente"] in {"piso", *AD.COMPONENTES}


def test_choque_de_uf_sem_sigma_nao_muda_o_piso() -> None:
    pm, base_m, base, pi_uf = _cenario()
    fk = np.ones((200, len(base)))
    v = AD.variancias_componentes(pm, base_m, base, pi_uf, LIG, 0.0, 200, 7, fk)
    assert v["var_uf"].to_numpy() == pytest.approx(v["var_piso"].to_numpy())


def test_tudo_e_ic_reproduzem_intervalos() -> None:
    pm, base_m, base, pi_uf = _cenario()
    ic = A.intervalos(pm, base_m, base, pi_uf, LIG, 0.1, 300, 7)
    share, regs = A.simular_shares(pm, base_m, base, pi_uf, LIG, 0.1, 300, 7)
    q = np.quantile(share, [0.025, 0.5, 0.975], axis=0)
    assert regs + ["Brasil"] == list(ic.index)
    assert ic["q025"].to_numpy() == pytest.approx(q[0], abs=0)
    assert ic["q975"].to_numpy() == pytest.approx(q[2], abs=0)


def test_decompor_soma_aditiva() -> None:
    var = pd.DataFrame(
        {
            "var_piso": [1.0],
            "var_beta": [2.0],
            "var_uf": [3.0],
            "var_theta": [1.5],
            "var_nu": [1.0],
            "var_tudo": [6.5],
            "var_ic": [6.0],
        },
        index=["Brasil"],
    )
    d = AD.decompor(var)
    assert d.loc["Brasil", "soma_partes"] == pytest.approx((1 + 1 + 2 + 0.5 + 0) / 6.5)
    assert d.loc["Brasil", "maior_componente"] == "uf"
    assert not d.loc["Brasil", "soma_dentro_10pct"]


def test_meia_largura_formula() -> None:
    _, _, base, _ = _cenario()
    pi = pd.Series(0.5, index=base.index)
    sem_f, com_f = AD.meia_largura_formula(base, pi, 0.1, "N")
    w = np.array([1400, 1000]) / 2400  # AA e BB nos votos restantes de N
    assert sem_f == pytest.approx(1.96 * 0.1 * 0.25 * np.sqrt((w**2).sum()) * 100)
    assert com_f == pytest.approx(sem_f * 2400 / (2400 + 400))


def test_meia_largura_formula_com_t_usa_variancia_k_sobre_k_menos_2() -> None:
    _, _, base, _ = _cenario()
    pi = pd.Series(0.5, index=base.index)
    normal = AD.meia_largura_formula(base, pi, 0.1, "N")
    t = AD.meia_largura_formula(base, pi, 0.1, "N", gl_uf=10)
    assert t[0] == pytest.approx(normal[0] * np.sqrt(10 / 8))
    assert t[1] == pytest.approx(normal[1] * np.sqrt(10 / 8))


def test_variancia_do_uf_com_t_e_maior_que_com_normal() -> None:
    pm, base_m, base, pi_uf = _cenario()
    fk = np.ones((4000, len(base)))
    normal = AD.variancias_componentes(pm, base_m, base, pi_uf, LIG, 0.1, 4000, 7, fk)
    t = AD.variancias_componentes(pm, base_m, base, pi_uf, LIG, 0.1, 4000, 7, fk, gl_uf=5)
    assert (t["var_uf"] > normal["var_uf"]).all()
    pd.testing.assert_series_equal(t["var_piso"], normal["var_piso"])


def test_composicao_capitais_e_apurado() -> None:
    _, _, base, _ = _cenario()
    base.loc[1, "n_c"] = 300.0
    base.loc[2, "n_c"] = 0.0
    c = AD.composicao_do_corte(base, {1}, "N+S").set_index("regiao")
    assert c.loc["N", "mun_com_apuracao"] == pytest.approx(3 / 4)
    assert c.loc["N", "capitais_do_apurado"] == pytest.approx(300 / 500)
    assert c.loc["S", "apurado_do_final"] == pytest.approx(400 / 2400)


def test_alavancagem_e_extrapolacao() -> None:
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]])
    v = AD.alavancagem(X, np.eye(2))
    assert v == pytest.approx([1.0, 2.0, 10.0])
    t = AD.extrapolacao_por_regiao(v[:2], v, np.array([10.0, 10.0, 30.0]), np.array(["N", "N", "S"]))
    assert t.set_index("regiao").loc["S", "extrapolado_nu"] == pytest.approx(1.0)
    assert t.set_index("regiao").loc["N", "extrapolado_nu"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# D4: seleção dentro do município e swing do 1º turno
# --------------------------------------------------------------------------
def _lag(pt: float, bo: float) -> float:
    return float(np.log((pt + 0.5) / (bo + 0.5)))


def _base_partes(lag_c: list[float], lag_u: list[float], n_u: list[float]) -> pd.DataFrame:
    n = len(n_u)
    return pd.DataFrame(
        {
            "uf": ["AA"] * n,
            "regiao": ["N"] * n,
            "lag1t_c": lag_c,
            "lag1t_u": lag_u,
            "fonte_c": "secoes",
            "fonte_u": "secoes",
            "n_u": n_u,
        },
        index=np.arange(1, n + 1),
    )


def test_d4_municipal_exemplo_a_mao() -> None:
    # lacunas +0,2 (n_u 100) e -0,1 (n_u 300): média -10/400 e média do módulo 50/400
    b = _base_partes([0.7, 0.4], [0.5, 0.5], [100.0, 300.0])
    r = AD.selecao_municipal(b)
    assert r.loc["Brasil", "gap"] == pytest.approx(-0.025)
    assert r.loc["Brasil", "gap_abs"] == pytest.approx(0.125)
    assert r.loc["N", "gap_abs"] == pytest.approx(0.125)


def test_d4_zero_quando_partes_iguais() -> None:
    b = _base_partes([0.3, -0.4], [0.3, -0.4], [100.0, 300.0])
    r = AD.selecao_municipal(b)
    assert r.loc["Brasil", "gap"] == pytest.approx(0.0)
    assert r.loc["Brasil", "gap_abs"] == pytest.approx(0.0)


def test_d4_ignora_parte_sem_dado_proprio() -> None:
    b = _base_partes([0.7, 0.4], [0.5, 0.5], [100.0, 300.0])
    b.loc[2, "fonte_u"] = "municipal"
    r = AD.selecao_municipal(b)
    assert r.loc["Brasil", "gap"] == pytest.approx(0.2)


def _secoes_uf() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    # município 1: apurada 90/10, restante 80/20; município 2: apurada 10/90, restante 20/80
    chave = pd.DataFrame(
        {
            "uf": "AA",
            "cd_municipio_tse": [1, 1, 2, 2],
            "zona": 1,
            "secao": [1, 2, 1, 2],
        }
    )
    s2 = chave.copy()
    pt = [90.0, 80.0, 10.0, 20.0]
    bo = [10.0, 20.0, 90.0, 80.0]
    s1 = chave.assign(ano=2022, votos_lula=pt, votos_bolsonaro=bo, validos_todos=100.0)
    rec = pd.Series([True, False, True, False])
    base = _base_partes([_lag(90, 10), _lag(10, 90)], [_lag(80, 20), _lag(20, 80)], [100.0, 100.0])
    return s2, s1, rec, base


def test_d4_uf_difere_da_media_municipal() -> None:
    s2, s1, rec, base = _secoes_uf()
    mun = AD.selecao_municipal(base)
    uf = AD.selecao_uf(s2, s1, rec, 2022, base)
    # no agregado da UF as duas partes têm 100 x 100: lacuna nula; nos municípios não
    assert uf.loc["Brasil", "gap_abs"] == pytest.approx(0.0, abs=1e-9)
    assert mun.loc["Brasil", "gap_abs"] > 0.3


def _base_swing(sw: float) -> pd.DataFrame:
    share_c = np.array([0.40, 0.55, 0.30, 0.60])
    share_u = np.array([0.45, 0.50, 0.35, 0.70])
    n_c = np.array([100.0, 200.0, 150.0, 80.0])
    n_u = np.array([300.0, 100.0, 250.0, 120.0])
    y_c = (share_c + sw) * n_c
    return pd.DataFrame(
        {
            "uf": ["AA", "AA", "BB", "BB"],
            "regiao": ["N", "N", "S", "S"],
            "share_c": share_c,
            "share_u": share_u,
            "n_c": n_c,
            "n_u": n_u,
            "y_c": y_c,
            "y_final": y_c + (share_u + sw) * n_u,
            "n_final": n_c + n_u,
        },
        index=np.arange(1, 5),
    )


def test_swing_1t_reproduz_o_final_se_o_swing_e_uniforme() -> None:
    b = _base_swing(0.03)
    est = AD.estimativa_swing_1t(b)
    real = A.alvo(b)
    assert (est - real).abs().max() < 1e-12


def test_swing_1t_entra_nos_metodos() -> None:
    assert "swing_1t" in A.METODOS
    assert len(A.METODOS) == 4
