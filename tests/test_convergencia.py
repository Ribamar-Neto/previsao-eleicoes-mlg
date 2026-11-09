"""Regressão de convergência do BFGS do beta-binomial nos dados reais."""
import json
from pathlib import Path

import pytest

from src import modelos as m
from src.especificacoes import CONTINUAS, MAPA_2026, Padronizador, Termos, montar_X

RAIZ = Path(__file__).resolve().parents[1]
PARQUET = RAIZ / "data" / "processed" / "municipios.parquet"
SELECAO = RAIZ / "results" / "tables" / "selecao_variaveis.json"


def test_bfgs_padrao_chega_ao_otimo_no_modelo_final():
    """Regra: o gtol padrão do BFGS é absoluto (1e-3), não proporcional a N.

    A llf é SOMADA; com gtol=1e-4*N o BFGS para dezenas de unidades abaixo do
    ótimo no bb_backward real (N=5.570, 12 contínuas + bloco + região). Um
    problema sintético não reproduz o mau condicionamento, por isso o teste usa
    os dados reais. O padrão deve ficar a menos de 0,5 do ótimo polido.
    """
    if not (PARQUET.exists() and SELECAO.exists()):
        pytest.skip("dados processados ausentes")
    import pandas as pd

    df = pd.read_parquet(PARQUET)
    sel = set(json.loads(SELECAO.read_text())["backward_bic"])
    vars_ = tuple(v for v in CONTINUAS if v in sel)
    pad = Padronizador(MAPA_2026).fit(df)
    X = montar_X(df, MAPA_2026, pad, Termos(vars=vars_))
    y = df[MAPA_2026["y"]].to_numpy(float)
    n = df[MAPA_2026["n"]].to_numpy(float)
    mod = m.BetaBinomial(y, X.to_numpy(float), n)

    antigo = mod.fit(gtol=1e-4 * len(df))
    padrao = mod.fit()
    polido = mod.fit(start_params=padrao.params, method="lbfgs", maxiter=3000)
    otimo = max(padrao.llf, polido.llf)
    print("gap antigo", otimo - antigo.llf, "gap padrao", otimo - padrao.llf)

    assert otimo - antigo.llf >= 50
    assert otimo - padrao.llf <= 0.5
