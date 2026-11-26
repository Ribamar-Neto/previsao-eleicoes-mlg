"""Fumaça dos gráficos dos antigos notebooks 05 e 06 (hoje `02-MLG` e `03-apuracao-parcial`): número de traços esperado."""

import numpy as np
import pandas as pd

from src import graficos as G


def test_figuras_do_notebook_05():
    dp = pd.DataFrame({"N": [1.0, 1.1], "S": [0.9, 1.0]}, index=["const", "regiao"])
    assert len(G.dp_por_regiao(dp, ["N", "S"], ["const", "regiao"]).data) == 2
    assert len(G.moran_contra_k(pd.DataFrame({"k_esf": [0, 10], "I_moran": [0.3, 0.1]})).data) == 1
    cg = pd.DataFrame({"ini_km": [0, 10], "fim_km": [10, 25], "correlacao": [0.2, 0.01]})
    assert len(G.correlograma_residuos(cg, "m", 10.0).data) == 1
    d = pd.DataFrame(
        {
            "nome": ["N", "S"],
            "lula_pct": [40.0, 60.0],
            "q025": [39.0, 59.0],
            "q975": [41.0, 61.0],
            "real_pct": [40.5, 60.5],
        }
    )
    assert len(G.regional_base_e_final(d, pd.Series([40.2, 60.1], index=["N", "S"]), "m").data) == 3


def test_erro_por_corte_tem_tres_metodos_por_ano():
    tab = pd.DataFrame(
        [
            {"ano": a, "corte": c, "metodo": m, "erro_brasil_pp": 1.0 + c}
            for a in (2022, 2026)
            for c in (5, 10)
            for m in ("bruta", "swing", "modelo")
        ]
    )
    fig = G.erro_por_corte(tab, (2022, 2026), "erro_brasil_pp", "erro", "t")
    assert len(fig.data) == 6 and np.all([t.mode == "lines+markers" for t in fig.data])
