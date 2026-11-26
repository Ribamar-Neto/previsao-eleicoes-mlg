"""Gravação padronizada de saídas: bytes do CSV, do HTML e do JSON."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from src import saidas as SA


def test_pastas_cria_tables_e_figures(tmp_path: Path) -> None:
    tabelas, figuras = SA.pastas(tmp_path / "results")
    assert tabelas.is_dir() and figuras.is_dir()
    assert (tabelas.name, figuras.name) == ("tables", "figures")


def test_salvar_tabela_arredonda_e_devolve(tmp_path: Path) -> None:
    t = pd.DataFrame({"a": [1.2345, 2.0]}, index=["x", "y"])
    saida = SA.salvar_tabela(t, tmp_path / "t.csv", casas=2, index_label="id")
    assert (tmp_path / "t.csv").read_bytes() == b"id,a\nx,1.23\ny,2.0\n"
    assert saida.loc["x", "a"] == 1.23


def test_salvar_tabela_sem_arredondar(tmp_path: Path) -> None:
    SA.salvar_tabela(pd.DataFrame({"a": [1.2345]}), tmp_path / "t.csv", index=False)
    assert (tmp_path / "t.csv").read_bytes() == b"a\n1.2345\n"


def test_salvar_fig_grava_so_png(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    caminho = SA.salvar_fig(go.Figure(go.Bar(x=["a"], y=[1])), tmp_path, "fig_x")
    assert caminho == tmp_path / "fig_x.png"
    assert caminho.read_bytes()[:4] == b"\x89PNG"
    assert not (tmp_path / "fig_x.html").exists()


def test_tabela_legenda_formatos_e_indice_oculto() -> None:
    df = pd.DataFrame({"a": [0.123, np.nan], "n": [1234, 5], "p": [0.0001, 0.5], "t": ["x", "y"]})
    sty = SA.tabela(df, "Resumo", numero="3", pct=["a"], inteiros=["n"], pvalor=["p"], nota="fonte X", indice=False)
    html = sty.to_html()
    assert "Tabela 3: Resumo" in html and "fonte X" in html
    assert "12,3%" in html and "1.234" in html
    assert "&lt;0,001" in html or "<0,001" in html
    assert ">-<" in html and "0,500" in html
    assert 'class="row_heading' not in html
    assert "Resumo" in SA.tabela(df, "Resumo").to_html()
    assert "Tabela" not in SA.tabela(df, "Resumo").to_html()


def test_tabela_casas_e_rotulos() -> None:
    df = pd.DataFrame({"a": [0.123456], "b": [2.0]})
    html = SA.tabela(df, "T", casas_por_coluna={"b": 1}, rotulos={"a": "Alfa"}).to_html()
    assert "Alfa" in html and "0,12" in html and "2,0" in html


def test_indicadores_duas_colunas() -> None:
    sty = SA.indicadores({"n": 12345, "r": 0.5, "s": "ok"}, "Ind", numero="1")
    assert list(sty.data.columns) == ["Indicador", "Valor"]
    html = sty.to_html()
    assert "12.345" in html and "0,500" in html and "ok" in html


def test_lista_markdown() -> None:
    md = SA.lista("Vars", ["x", "y"]).data
    sem_codigo = SA.lista("V", ["x"], codigo=False).data
    assert isinstance(md, str) and isinstance(sem_codigo, str)
    assert "- `x`" in md and "**Vars**" in md
    assert "- x" in sem_codigo


def test_salvar_json_utf8_indentado_e_default_str(tmp_path: Path) -> None:
    destino = SA.salvar_json({"nome": "São Paulo", "p": Path("a")}, tmp_path / "d.json")
    assert destino.read_text() == '{\n  "nome": "São Paulo",\n  "p": "a"\n}'


def test_legenda_e_nota_escapam_html_e_pvalor_mostra_menor_que() -> None:
    df = pd.DataFrame({"p": [0.0001, 0.5]})
    html = SA.tabela(df, "Breslow-Day (p < 0,001)", numero="6", pvalor=["p"], nota="a<b & c").to_html()
    assert "Tabela 6: Breslow-Day (p &lt; 0,001)" in html
    assert "a&lt;b &amp; c" in html
    assert ">&lt;0,001</td>" in html and ">0,500</td>" in html
