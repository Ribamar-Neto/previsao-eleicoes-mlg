"""Testes de fumaça dos gráficos: a figura sai com o número esperado de traços e o HTML é gravado."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import graficos as G


def test_qq_normal_um_traco_por_modelo_mais_diagonal() -> None:
    rng = np.random.default_rng(0)
    fig = G.qq_normal({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    assert len(fig.data) == 3


def test_residuos_contra_tres_tracos_por_painel() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    fig = G.residuos_contra(x, {"a": rng.normal(size=60), "b": rng.normal(size=60)}, "x", "t", faixas=5)
    assert len(fig.data) == 4


def test_caixa_regiao_e_alavanca_cook() -> None:
    rng = np.random.default_rng(1)
    regiao = np.array(["N", "S"] * 20)
    assert len(G.caixa_regiao({"a": rng.normal(size=40)}, regiao).data) == 1
    fig = G.alavanca_cook(rng.uniform(size=40), rng.uniform(size=40), [str(i) for i in range(40)], p=3)
    assert len(fig.data) == 1


def test_amostra_limita_pontos() -> None:
    assert len(G._amostra(10, 100)) == 10
    assert len(G._amostra(5000, 100)) == 100


def test_mostrar_e_salvar_grava_html(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    dados = pd.DataFrame({"nome_municipio": ["A", "B"], "sigla_uf": ["X", "Y"]})
    G.configurar(dados, tmp_path / "figs")
    G.histograma([0.1, 0.2, 0.3], "v", name="hist_teste")
    assert (tmp_path / "figs" / "hist_teste.html").exists()
    p = G.pontos([1, 2], [3, 4], [0, 1], extras={"z": [0.5, 0.6]})
    assert "<b>A</b> (X)" in p.hovertext[0]
