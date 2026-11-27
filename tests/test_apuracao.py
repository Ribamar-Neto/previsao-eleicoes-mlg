"""Testes das funções puras da apuração parcial (dados sintéticos)."""

import numpy as np
import pandas as pd
import pytest

from src import apuracao as A

T = pd.Timestamp("2022-10-30 17:00:00", tz="America/Sao_Paulo")


def _secoes() -> pd.DataFrame:
    # válidos 10, 10, 20, 20, 40 (total 100); as duas últimas seções chegam no mesmo instante
    linhas = [
        (1, "AA", 0, 10, 6, 12),
        (1, "AA", 1, 10, 5, 12),
        (2, "AA", 2, 20, 8, 25),
        (3, "BB", 3, 20, 4, 30),
        (3, "BB", 3, 40, 10, 50),
    ]
    df = pd.DataFrame(linhas, columns=["cd_municipio_tse", "uf", "min", "validos", "votos_lula", "aptos"])
    df["dt_bu_recebido"] = [T + pd.Timedelta(minutes=int(m)) for m in df["min"]]
    df.loc[3, "dt_bu_recebido"] = T + pd.Timedelta(minutes=3)
    df.loc[4, "dt_bu_recebido"] = T + pd.Timedelta(minutes=3)
    return df


def _dfm() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_municipio_tse": [1, 2],
            "sigla_uf": ["AA", "AA"],
            "regiao": ["Norte", "Norte"],
            "validos_1t_2022": [80.0, 160.0],
            "aptos_1t_2022": [100.0, 200.0],
            "prop_pt_2t_2018": [0.5, 0.4],
        }
    )


@pytest.mark.parametrize("pct,minuto", [(5, 0), (10, 0), (11, 1), (20, 1), (21, 2), (40, 2), (41, 3), (100, 3)])
def test_instante_do_corte(pct: float, minuto: int) -> None:
    assert A.instante_do_corte(_secoes(), pct) == T + pd.Timedelta(minutes=minuto)


def test_cortes_juntam_secoes_do_mesmo_instante() -> None:
    s = _secoes()
    rec = A.recebidas_ate(s, 50)
    assert rec.tolist() == [True, True, True, True, True]  # 41 a 100 % chegam juntos em t+3
    assert A.pct_efetivo(s, A.recebidas_ate(s, 25)) == pytest.approx(40.0)


def test_votos_restantes_e_base() -> None:
    s = _secoes()
    rec = A.recebidas_ate(s, 5)  # só a primeira seção
    base = A.montar_base(s, _dfm(), 2022, rec)
    assert base.loc[1, ["y_c", "n_c", "aptos_pend"]].tolist() == [6.0, 10.0, 12.0]
    assert base.loc[1, "n_u"] == round(12 * 0.8)  # taxa 80/100
    assert base.loc[2, "n_u"] == round(25 * 0.8)
    assert base.loc[2, "n_c"] == 0
    assert not base.loc[3, "no_modelo"]


def test_fora_do_modelo_usa_taxa_e_regiao_da_uf() -> None:
    s = _secoes()
    s.loc[3:4, "uf"] = "AA"
    base = A.montar_base(s, _dfm(), 2022, A.recebidas_ate(s, 5))
    assert base.loc[3, "regiao"] == "Norte"
    assert base.loc[3, "taxa"] == pytest.approx(240 / 300)  # válidos/aptos do 1º turno da UF
    assert np.isnan(base.loc[3, "prev"])


def test_swing_uniforme() -> None:
    y, n, prev = np.array([60.0, 30.0]), np.array([100.0, 100.0]), np.array([0.5, 0.5])
    assert A.swing_uniforme(y, n, prev) == pytest.approx(-0.05)
    # prev NaN fica fora da média
    assert A.swing_uniforme(np.append(y, 1.0), np.append(n, 5.0), np.append(prev, np.nan)) == pytest.approx(-0.05)


def test_estimativa_swing_aplica_ao_nao_apurado() -> None:
    base = pd.DataFrame(
        {
            "uf": ["AA", "AA"],
            "regiao": ["Norte", "Norte"],
            "y_c": [60.0, 0.0],
            "n_c": [100.0, 0.0],
            "n_u": [0.0, 100.0],
            "prev": [0.5, 0.4],
        }
    )
    est = A.estimativa_swing(base)
    # swing = +0,10; município 2 vira 0,50 -> (60 + 50) / 200
    assert est["Brasil"] == pytest.approx(0.55)
    assert A.estimativa_bruta(base)["Brasil"] == pytest.approx(0.6)


def test_completar_por_uf_e_modelo() -> None:
    n_u = pd.Series([100.0, 100.0, 50.0], index=[1, 2, 3])
    uf = pd.Series(["AA", "AA", "AA"], index=[1, 2, 3])
    pi = pd.Series([0.4, 0.6, np.nan], index=[1, 2, 3])
    assert A.completar_por_uf(pi, n_u, uf)[3] == pytest.approx(0.5)


def test_erros_e_regra() -> None:
    est = pd.Series({"Norte": 0.52, "Sul": 0.46, "Brasil": 0.50})
    real = pd.Series({"Norte": 0.50, "Sul": 0.50, "Brasil": 0.51})
    br, reg = A.erros(est, real)
    assert br == pytest.approx(1.0) and reg == pytest.approx(3.0)
    linhas = []
    for c, (b, m, mo) in enumerate([(1, 2, 0.5), (1, 2, 0.5), (1, 2, 3), (1, 2, 0.5)]):
        for met, e in (("bruta", b), ("swing", b), ("modelo", mo)):
            linhas.append({"ano": 2022, "corte": c, "metodo": met, "erro_brasil_pp": e, "mae_regional_pp": m * e})
    r = A.aplicar_regra(pd.DataFrame(linhas), minimo=3)
    assert r.loc[0, "vitorias_modelo"] == 3 and bool(r.loc[0, "util"])
    assert not bool(A.aplicar_regra(pd.DataFrame(linhas), minimo=4).loc[0, "util"])


def test_swing_1t_que_vence_o_modelo_no_brasil_tira_o_corte() -> None:
    linhas = []
    for c in range(4):
        for met, br, mae in (("bruta", 2, 4), ("swing", 2, 4), ("swing_1t", 0.4, 4), ("modelo", 0.5, 1)):
            linhas.append({"ano": 2022, "corte": c, "metodo": met, "erro_brasil_pp": br, "mae_regional_pp": mae})
    assert A.aplicar_regra(pd.DataFrame(linhas), minimo=1).loc[0, "vitorias_modelo"] == 0
    sem = pd.DataFrame(linhas)
    sem = sem[sem["metodo"] != "swing_1t"]
    assert A.aplicar_regra(sem, minimo=1).loc[0, "vitorias_modelo"] == 4


def test_representatividade_separa_maiores() -> None:
    base = pd.DataFrame(
        {"y_c": [60.0, 10.0], "n_c": [100.0, 20.0], "y_final": [300.0, 40.0], "n_final": [1000.0, 100.0]}
    )
    r = A.representatividade(base, None, n_maiores=1).set_index("grupo")
    # maior: apurado 0,60; restante 240/900 = 0,2667
    assert r.loc["maiores", "lacuna_apurado_pp"] == pytest.approx((0.6 - 240 / 900) * 100)
    assert r.loc["demais", "n_municipios"] == 1


def test_taxa_literal_usa_so_os_dois_candidatos() -> None:
    dfm = _dfm().assign(votos_pt_1t_2022=[30.0, 60.0], votos_bolsonaro_1t_2022=[20.0, 40.0])
    assert A.taxa_validos_1t(dfm, 2022).tolist() == [0.8, 0.8]
    assert A.taxa_validos_1t(dfm, 2022, "dois_candidatos").tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        A.taxa_validos_1t(dfm, 2022, "outra")


def test_montar_base_taxa_literal_reduz_n_u() -> None:
    s = _secoes()
    dfm = _dfm().assign(votos_pt_1t_2022=[30.0, 60.0], votos_bolsonaro_1t_2022=[20.0, 40.0])
    rec = A.recebidas_ate(s, 10)
    b_val = A.montar_base(s, dfm, 2022, rec)
    b_lit = A.montar_base(s, dfm, 2022, rec, "dois_candidatos")
    assert b_val.loc[2, "n_u"] == 20.0  # 25 aptos x 0,8
    assert b_lit.loc[2, "n_u"] == 12.0  # 25 x 0,5 = 12,5 arredondado
    s.loc[3:4, "uf"] = "AA"  # município fora do parquet usa a taxa da UF na mesma definição
    assert A.montar_base(s, dfm, 2022, rec, "dois_candidatos").loc[3, "taxa"] == pytest.approx(150 / 300)


def _base_nu() -> pd.DataFrame:
    # 3 UFs de 6 municípios; r_i = n_c / (aptos_rec x taxa) varia por município e por UF
    linhas = []
    for j, (uf, f) in enumerate([("AA", 1.0), ("BB", 1.2), ("CC", 0.8)]):
        for i in range(6):
            linhas.append(
                {
                    "cd": 10 * j + i,
                    "uf": uf,
                    "n_c": 100.0 * f * (1 + 0.1 * i),
                    "aptos_rec": 200.0,
                    "taxa": 0.5,
                    "n_u": 50.0,
                }
            )
    return pd.DataFrame(linhas).set_index("cd")


def test_fatores_nu_centrados_em_1_e_reproduziveis() -> None:
    base = _base_nu()
    k = A.fatores_nu(base, 4000, 3)
    assert k.shape == (4000, len(base))
    assert np.allclose(k, A.fatores_nu(base, 4000, 3))
    assert np.median(np.log(k)) == pytest.approx(0.0, abs=0.05)
    assert np.log(k).std() > 0


def test_fatores_nu_sem_dispersao_vira_um() -> None:
    base = _base_nu().assign(n_c=100.0)
    assert np.allclose(A.fatores_nu(base, 50, 1), 1.0)


def test_fatores_nu_uf_pequena_usa_pool_nacional_e_invalidos_recebem_fator() -> None:
    base = _base_nu()
    base.loc[[0, 1, 2, 3], "n_c"] = 0.0  # AA fica com 2 válidos (< 5) e 4 inválidos
    k = A.fatores_nu(base, 100, 2)
    assert np.isfinite(k).all() and (k > 0).all()


def test_montar_base_aptos_rec() -> None:
    base = A.montar_base(_secoes(), _dfm(), 2022, A.recebidas_ate(_secoes(), 5))
    assert base.loc[1, "aptos_rec"] == 12.0 and base.loc[2, "aptos_rec"] == 0.0
    assert base.loc[1, "aptos_rec"] + base.loc[1, "aptos_pend"] == 24.0
