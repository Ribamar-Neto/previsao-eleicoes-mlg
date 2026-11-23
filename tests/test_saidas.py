"""Gravação padronizada de saídas: bytes do CSV, do HTML e do JSON."""
from pathlib import Path

import pandas as pd
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


def test_salvar_fig_html_igual_ao_write_html_direto(tmp_path: Path) -> None:
    fig = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))
    caminho = SA.salvar_fig(fig, tmp_path, "fig_x")
    fig.write_html(tmp_path / "ref.html", include_plotlyjs="cdn", div_id="fig_x")
    assert caminho == tmp_path / "fig_x.html"
    assert caminho.read_bytes() == (tmp_path / "ref.html").read_bytes()
    assert b'id="fig_x"' in caminho.read_bytes()


def test_salvar_json_utf8_indentado_e_default_str(tmp_path: Path) -> None:
    destino = SA.salvar_json({"nome": "São Paulo", "p": Path("a")}, tmp_path / "d.json")
    assert destino.read_text() == '{\n  "nome": "São Paulo",\n  "p": "a"\n}'
