"""Testes da normalização do boletim de urna (bweb)."""

import pandas as pd
import pytest

from src.coleta import tse_bweb as b


def _bruto(ano: int, agreg: str = "#NULO#") -> pd.DataFrame:
    data = "30/10/2022 17:34:43" if ano == 2022 else "2026-10-25 17:34:43"
    col = b.COL_AGREGADAS[ano]
    base = {
        "SG_UF": "AC",
        "CD_MUNICIPIO": "1392",
        "NR_ZONA": "1",
        "NR_SECAO": "3",
        "CD_CARGO_PERGUNTA": "1",
        "DT_BU_RECEBIDO": data,
        "QT_APTOS": "185",
        "QT_COMPARECIMENTO": "147",
        col: agreg,
    }
    linhas = [
        {**base, "NR_VOTAVEL": "13", "QT_VOTOS": "48"},
        {**base, "NR_VOTAVEL": "22", "QT_VOTOS": "90"},
        {**base, "NR_VOTAVEL": "96", "QT_VOTOS": "9"},
        {**base, "CD_CARGO_PERGUNTA": "3", "NR_VOTAVEL": "13", "QT_VOTOS": "999"},  # outro cargo
    ]
    return pd.DataFrame(linhas)


def test_data_nos_dois_leiautes() -> None:
    d22 = b.parse_data(pd.Series(["30/10/2022 17:34:43"]), 2022)[0]
    d26 = b.parse_data(pd.Series(["2026-10-25 17:34:43"]), 2026)[0]
    assert str(d22) == "2022-10-30 17:34:43-03:00"
    assert str(d26) == "2026-10-25 17:34:43-03:00"


def test_data_invalida_levanta() -> None:
    with pytest.raises(ValueError):
        b.parse_data(pd.Series(["2026-10-25 17:34:43"]), 2022)


@pytest.mark.parametrize("ano", [2022, 2026])
def test_normalizar_uma_linha_por_secao(ano: int) -> None:
    out = b.normalizar(_bruto(ano), ano)
    assert len(out) == 1
    r = out.iloc[0]
    assert (r.votos_lula, r.votos_bolsonaro, r.validos) == (48, 90, 138)  # brancos/nulos e outro cargo fora
    assert (r.aptos, r.comparecimento, r.zona, r.secao) == (185, 147, 1, 3)
    assert pd.isna(r.agregadas)


def test_secoes_agregadas_ficam_com_a_principal() -> None:
    out = b.normalizar(_bruto(2026, agreg="175"), 2026)
    assert len(out) == 1 and out.iloc[0].agregadas == "175"


def test_candidato_sem_voto_vira_zero() -> None:
    bruto = _bruto(2022)
    bruto = bruto[bruto["NR_VOTAVEL"] != "13"]
    assert b.normalizar(bruto, 2022).iloc[0].votos_lula == 0


def test_validar_rejeita_chave_duplicada() -> None:
    out = b.normalizar(_bruto(2022), 2022).assign(id_municipio=1)
    b.validar_secoes(out)
    with pytest.raises(ValueError):
        b.validar_secoes(pd.concat([out, out], ignore_index=True))


def test_virada_ultima_cruzada() -> None:
    c = pd.DataFrame({"share_lula": [40.0, 51.0, 49.0, 50.5, 52.0]})
    assert virada_idx(c) == 3


def virada_idx(c: pd.DataFrame) -> int:
    v = b.virada(c)
    assert v is not None
    return int(str(v.name))
