"""Lag1t e outros_1t por parte (apurada x restante) na apuração parcial (dados sintéticos)."""

import numpy as np
import pandas as pd
import pytest

from src import apuracao as A

CHAVE = ["uf", "cd_municipio_tse", "zona", "secao"]


def _secoes(mun: int = 10, n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uf": "AA",
            "cd_municipio_tse": mun,
            "zona": 1,
            "secao": range(1, n + 1),
            "aptos": 100,
            "validos": 80,
            "votos_lula": 40,
        }
    )


def _s1t(mun: int = 10, ano: int = 2022) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ano": ano,
            "uf": "AA",
            "cd_municipio_tse": mun,
            "zona": 1,
            "secao": [1, 2, 3, 4],
            "votos_lula": [30, 20, 5, 16],
            "votos_bolsonaro": [10, 20, 15, 5],
            "validos_todos": [50, 50, 40, 60],
        }
    )


def _dfm(ids: tuple[int, ...] = (10,)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_municipio_tse": list(ids),
            "logit_1t_2022": 0.3,
            "outros_1t_2022": 0.1,
            "sigla_uf": "AA",
            "regiao": "Norte",
            "validos_1t_2022": 80.0,
            "aptos_1t_2022": 100.0,
            "prop_pt_2t_2018": 0.5,
        }
    )


def _rec(*v: bool) -> pd.Series:
    return pd.Series(list(v))


def test_partes_a_mao() -> None:
    p = A.partes_1t(_secoes(), _s1t(), _rec(True, True, False, False), _dfm(), 2022).loc[10]
    assert p["lag1t_c"] == pytest.approx(np.log(50.5 / 30.5))
    assert p["outros_1t_c"] == pytest.approx(0.2)
    assert p["lag1t_u"] == pytest.approx(np.log(21.5 / 20.5))
    assert p["outros_1t_u"] == pytest.approx(1 - 41 / 100)
    assert p["share_c"] == pytest.approx(50 / 80)
    assert p["share_u"] == pytest.approx(21 / 41)
    assert (p["fonte_c"], p["fonte_u"]) == ("secoes", "secoes")


def test_tudo_apurado_vira_municipal() -> None:
    p = A.partes_1t(_secoes(), _s1t(), _rec(True, True, True, True), _dfm(), 2022).loc[10]
    assert p["lag1t_c"] == 0.3 and p["outros_1t_c"] == 0.1  # exatamente o municipal
    assert p["lag1t_u"] == 0.3 and p["fonte_u"] == "municipal" and p["fonte_c"] == "municipal"


def test_nada_apurado_resto_municipal() -> None:
    p = A.partes_1t(_secoes(), _s1t(), _rec(False, False, False, False), _dfm(), 2022).loc[10]
    assert p["fonte_c"] == "municipal" and p["fonte_u"] == "municipal"
    assert p["outros_1t_u"] == 0.1


def test_secao_sem_par_fica_de_fora() -> None:
    s = _secoes(n=5)  # a seção 5 não existe no 1º turno
    p = A.partes_1t(s, _s1t(), _rec(True, True, False, False, False), _dfm(), 2022).loc[10]
    assert p["lag1t_u"] == pytest.approx(np.log(21.5 / 20.5))


def test_parte_sem_secao_casada_cai_no_municipal() -> None:
    s1 = _s1t()
    s1 = s1[s1["secao"].isin([1, 2])]  # as seções restantes (3, 4) não casam
    p = A.partes_1t(_secoes(), s1, _rec(True, True, False, False), _dfm(), 2022).loc[10]
    assert p["fonte_c"] == "secoes" and p["fonte_u"] == "municipal"
    assert p["lag1t_u"] == 0.3


def test_denominador_zero_cai_no_municipal() -> None:
    s1 = _s1t().assign(validos_todos=0, votos_lula=0, votos_bolsonaro=0)
    p = A.partes_1t(_secoes(), s1, _rec(True, True, False, False), _dfm(), 2022).loc[10]
    assert p["fonte_c"] == "municipal" and p["outros_1t_c"] == 0.1 and p["lag1t_c"] == 0.3


def test_zero_voto_de_um_finalista_segue_finito() -> None:
    s1 = _s1t().assign(votos_bolsonaro=0)
    p = A.partes_1t(_secoes(), s1, _rec(True, True, False, False), _dfm(), 2022).loc[10]
    assert p["lag1t_c"] == pytest.approx(np.log(50.5 / 0.5))


def test_sem_vazamento_do_2t_nem_de_outro_ano() -> None:
    rec = _rec(True, True, False, False)
    base = A.partes_1t(_secoes(), _s1t(), rec, _dfm(), 2022)
    s_outra = _secoes().assign(votos_lula=999, validos=1)  # 2º turno do ano alvo muda
    assert A.partes_1t(s_outra, _s1t(), rec, _dfm(), 2022).equals(base)
    ruido = _s1t(ano=2018).assign(votos_lula=1000)  # outro ano é ignorado
    assert A.partes_1t(_secoes(), pd.concat([_s1t(), ruido]), rec, _dfm(), 2022).equals(base)


def test_montar_base_junta_partes_e_default_inalterado() -> None:
    rec = _rec(True, True, False, False)
    sem = A.montar_base(_secoes(), _dfm(), 2022, rec)
    com = A.montar_base(_secoes(), _dfm(), 2022, rec, secoes_1t=_s1t())
    assert list(com.columns[: len(sem.columns)]) == list(sem.columns)
    assert com[sem.columns].equals(sem)
    assert com.loc[10, "outros_1t_c"] == pytest.approx(0.2)
    assert "lag1t_c" not in sem.columns


def test_sobrescritas_so_para_chaves_do_mapa() -> None:
    base = A.montar_base(_secoes(), _dfm(), 2022, _rec(True, True, False, False), secoes_1t=_s1t())
    mapa = {"lag1t": "logit_1t_2022", "outros_1t": "outros_1t_2022", "lag": "x"}
    treino, resto = A.sobrescritas_1t(base, mapa)
    assert set(treino) == set(resto) == {"logit_1t_2022", "outros_1t_2022"}
    assert treino["outros_1t_2022"][0] == pytest.approx(0.2) and resto["outros_1t_2022"][0] == pytest.approx(0.59)
    t2, r2 = A.sobrescritas_1t(base, {"lag1t": "logit_1t_2022"})
    assert set(t2) == set(r2) == {"logit_1t_2022"}
    assert A.sobrescritas_1t(A.montar_base(_secoes(), _dfm(), 2022, _rec(True, True, False, False)), mapa) == ({}, {})


def test_aplicar_sobrescritas_mantem_valor_quando_nan() -> None:
    d = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = A.aplicar_sobrescritas(d, {"a": np.array([9.0, np.nan, 7.0])}, np.array([True, True, False]))
    assert out["a"].tolist() == [9.0, 2.0, 3.0]
    assert d["a"].tolist() == [1.0, 2.0, 3.0]
    assert A.aplicar_sobrescritas(d, None, np.array([True] * 3)) is d
