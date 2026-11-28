"""Testes de fumaça dos gráficos: a figura sai com o número esperado de traços e o PNG é gravado."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import graficos as G


def test_qq_normal_um_traco_por_modelo_mais_diagonal() -> None:
    rng = np.random.default_rng(0)
    fig = G.qq_normal({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    assert len(fig.data) == 3


def test_residuos_contra_tres_tracos_por_painel() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    fig = G.residuos_contra(x, {"a": rng.normal(size=60), "b": rng.normal(size=60)}, "x", "t")
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


def test_mostrar_e_salvar_grava_png(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("kaleido")
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    dados = pd.DataFrame({"nome_municipio": ["A", "B"], "sigla_uf": ["X", "Y"]})
    G.configurar(dados, tmp_path / "figs")
    G.histograma([0.1, 0.2, 0.3], "v", name="hist_teste")
    assert (tmp_path / "figs" / "hist_teste.png").exists()
    assert not (tmp_path / "figs" / "hist_teste.html").exists()
    p = G.pontos([1, 2], [3, 4], [0, 1], extras={"z": [0.5, 0.6]})
    assert "<b>A</b> (X)" in p.hovertext[0]


def test_salvar_grava_so_png(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("kaleido")
    monkeypatch.setattr(G, "output_fig_dir", tmp_path)
    G.salvar(go.Figure(go.Scatter(x=[1, 2], y=[1, 2])), tmp_path / "x.html")
    assert [p.name for p in tmp_path.iterdir()] == ["x.png"]


def test_template_usa_virgula_decimal() -> None:
    assert G.pio.templates["eleicoes"].layout.separators == ",."


def test_painel_diagnostico_quatro_paineis_e_rotulos_cook() -> None:
    rng = np.random.default_rng(0)
    n = 60
    fig = G.painel_diagnostico(
        rng.uniform(0, 0.1, n),
        rng.uniform(0, 1, n),
        rng.normal(size=n),
        rng.uniform(size=n),
        p=3,
        rotulos=[f"m{i}" for i in range(n)],
        titulo="Diagnóstico",
    )
    assert fig.layout.title.text == "Diagnóstico"
    assert len(fig.data) == 6  # 4 painéis + rótulos Cook + LOWESS
    assert len(next(t for t in fig.data if t.mode == "markers+text").text) == 5
    assert all(isinstance(t, go.Scatter) for t in fig.data)


def test_forest_or_eixo_log_ordenado() -> None:
    tab = pd.DataFrame(
        {"termo": ["a", "b", "c"], "or": [2.0, 0.5, 1.1], "ic_inf": [1.5, 0.4, 0.9], "ic_sup": [2.6, 0.6, 1.3]}
    )
    fig = G.forest_or(tab, "OR")
    assert fig.layout.xaxis.type == "log" and fig.layout.title.text == "OR"
    assert list(fig.data[0].y) == ["b", "c", "a"]


def test_moran_dispersao_reta_com_inclinacao_i() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=100)
    fig = G.moran_dispersao(z, 0.3 * z, 0.3, "Moran")
    assert [t.type for t in fig.data] == ["scatter", "scatter"]
    reta = fig.data[1]
    assert (reta.y[1] - reta.y[0]) / (reta.x[1] - reta.x[0]) == pytest.approx(0.3)
    assert {a.text for a in fig.layout.annotations} >= {"<b>AA</b>", "<b>BB</b>", "<b>AB</b>", "<b>BA</b>"}


def test_mae_por_modelo_destaca_escolhido_e_limite() -> None:
    tab = pd.DataFrame({"modelo": ["a", "b", "c"], "mae": [1.0, 1.1, 1.5], "se": [0.1, 0.1, 0.2]})
    fig = G.mae_por_modelo(tab, limite=1.1, escolhido="b", titulo="MAE")
    assert fig.layout.title.text == "MAE" and len(fig.data) == 1
    assert fig.data[0].marker.color[1] == G.COR_CURVA  # "b" é o segundo da ordem decrescente (c, b, a)
    assert fig.layout.shapes[0].x0 == 1.1


def _ic_sintetico() -> pd.DataFrame:
    linhas = [
        {
            "ano": 2026,
            "corte": c,
            "pct_efetivo": c,
            "regiao": r,
            "estimativa_pct": 50 + c / 50,
            "q025_pct": 48,
            "q975_pct": 52,
            "real_pct": 51.0,
        }
        for r in G.UNIDADES_APURACAO
        for c in (5, 25, 50)
    ]
    return pd.DataFrame(linhas)


def test_estimativa_por_corte_uma_faceta_por_unidade() -> None:
    fig = G.estimativa_por_corte(_ic_sintetico(), "Cortes", ano=2026)
    assert fig.layout.title.text == "Cortes"
    assert len([t for t in fig.data if t.error_y.array is not None]) == 6


def test_curva_apuracao_com_e_sem_estimativas() -> None:
    curva = pd.DataFrame({"pct_recebido": np.linspace(0, 100, 50), "share_lula": np.linspace(55, 51, 50)})
    ic = _ic_sintetico().query("regiao == 'Brasil'")
    assert len(G.curva_apuracao(curva, ic, 51.0, "Curva").data) == 2
    assert len(G.curva_apuracao(curva, None, 51.0, "Curva").data) == 1


def test_mapa_uf_divergente_centrado_em_zero_e_cache() -> None:
    pytest.importorskip("geopandas")
    valores = pd.Series({"SP": -1.0, "BA": 2.0, "RS": 0.5})
    fig = G.mapa_uf(valores, "Erro", divergente=True)
    assert fig.layout.title.text == "Erro"
    assert fig.data[0].type == "choropleth" and fig.data[0].zmid == 0
    assert fig.data[0].zmin == -2.0 and fig.data[0].zmax == 2.0
    assert G._geometria_uf() is G._geometria_uf()


def test_serie_regional_uma_linha_por_regiao() -> None:
    tab = pd.DataFrame([{"regiao": r, "ano": a, "prop_pt": 0.4} for r in G.CORES_REGIAO for a in (2018, 2022)])
    fig = G.serie_regional(tab, "Série")
    assert len(fig.data) == 5 and fig.layout.title.text == "Série"
