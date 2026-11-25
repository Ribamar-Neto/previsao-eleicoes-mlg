"""Gravação padronizada das saídas dos notebooks (tabelas CSV, figuras PNG e JSON).

Regra de negócio: os arquivos em results/ são versionados e conferidos byte a byte; por isso
a escrita fica num só lugar (mesmo `include_plotlyjs="cdn"`, `div_id` igual ao nome do arquivo,
JSON com `indent=2` e `ensure_ascii=False`), e os notebooks não repetem esses parâmetros.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import Markdown
from pandas.io.formats.style import Styler
from plotly.graph_objects import Figure


def pastas(raiz_results: str | Path) -> tuple[Path, Path]:
    """Cria (se preciso) e devolve as pastas `tables` e `figures` dentro de `raiz_results`."""
    raiz = Path(raiz_results)
    tabelas, figuras = raiz / "tables", raiz / "figures"
    tabelas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    return tabelas, figuras


def salvar_tabela(tabela: pd.DataFrame | pd.Series, caminho: str | Path, casas: int | None = None,
                  **kwargs_csv: Any) -> pd.DataFrame | pd.Series:
    """Grava o CSV (arredondado a `casas`, se dado) e devolve a tabela gravada para exibição."""
    saida = tabela if casas is None else tabela.round(casas)
    saida.to_csv(caminho, **kwargs_csv)
    return saida


def salvar_fig(fig: Figure, pasta: str | Path, nome: str) -> Path:
    """Grava `<pasta>/<nome>.png`; só PNG é versionado (sem HTML)."""
    caminho = Path(pasta) / f"{nome}.png"
    fig.write_image(caminho)
    return caminho


def salvar_json(obj: Any, caminho: str | Path) -> Path:
    """Grava `obj` em JSON legível; valores fora do padrão JSON viram texto."""
    destino = Path(caminho)
    destino.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return destino


_TRACO = "-"


def _num(v: float, casas: int) -> str:
    """Número em formato US; o Styler troca para vírgula decimal e ponto de milhar."""
    return f"{v:,.{casas}f}"


def _num_br(v: float, casas: int) -> str:
    """Número no padrão brasileiro, para células já em texto (sem passar pelo Styler)."""
    return _num(v, casas).replace(",", "§").replace(".", ",").replace("§", ".")


def _fmt(f: Any) -> Any:
    """Envolve um formatador de número para exibir NaN como traço."""
    return lambda v: _TRACO if pd.isna(v) else f(v)


def _legenda(titulo: str, numero: str | None, nota: str | None) -> str:
    # Legenda e nota são texto puro: "<" de "p < 0,001" não pode virar tag HTML.
    base = html.escape(titulo if numero is None else f"Tabela {numero}: {titulo}", quote=False)
    if nota:
        base += f'<br><span style="font-weight:normal;font-size:smaller">{html.escape(nota, quote=False)}</span>'
    return base


def _estilizar(df: pd.DataFrame, legenda: str, formatos: Mapping[str, Any], indice: bool) -> Styler:
    sty = df.style.set_caption(legenda).set_table_styles(
        [{"selector": "caption", "props": "text-align:left;font-weight:bold;"}]
    )
    sty.format(formatos, na_rep=_TRACO, decimal=",", thousands=".", escape="html")  # altera sty no lugar
    return sty if indice else sty.hide(axis="index")


def tabela(df: pd.DataFrame, titulo: str, *, numero: str | None = None, casas: int = 2,
           casas_por_coluna: Mapping[str, int] | None = None, pct: Sequence[str] = (),
           inteiros: Sequence[str] = (), pvalor: Sequence[str] = (),
           rotulos: Mapping[str, str] | None = None, indice: bool = True,
           nota: str | None = None) -> Styler:
    """Tabela formatada para exibição: legenda numerada, vírgula decimal, ponto de milhar.

    `pct` recebe frações (0 a 1); `pvalor` mostra "<0,001" abaixo de 0,001. Colunas não
    numéricas passam sem formato; `rotulos` renomeia só a exibição. O CSV segue o dado cru.
    """
    casas_col = casas_por_coluna or {}
    formatos: dict[str, Any] = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col in pct:
            formatos[col] = _fmt(lambda v: _num(100 * v, 1) + "%")
        elif col in pvalor:
            formatos[col] = _fmt(lambda v: "&lt;0.001" if v < 0.001 else _num(v, 3))
        elif col in inteiros:
            formatos[col] = _fmt(lambda v: _num(v, 0))
        else:
            formatos[col] = _fmt(lambda v, c=casas_col.get(col, casas): _num(v, c))
    saida = df.rename(columns=dict(rotulos)) if rotulos else df
    if rotulos:
        formatos = {rotulos.get(c, c): f for c, f in formatos.items()}
    return _estilizar(saida, _legenda(titulo, numero, nota), formatos, indice)


def indicadores(valores: Mapping[str, object], titulo: str, *, numero: str | None = None,
                casas: int = 3) -> Styler:
    """Tabela de duas colunas (Indicador, Valor) no lugar de dict e escalares impressos."""
    linhas = [(k, v) for k, v in valores.items()]
    df = pd.DataFrame(linhas, columns=["Indicador", "Valor"])
    df["Valor"] = [_texto_valor(v, casas) for v in df["Valor"]]
    return _estilizar(df, _legenda(titulo, numero, None), {}, False)


def _texto_valor(v: object, casas: int) -> str:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    if pd.isna(v):
        return _TRACO
    if isinstance(v, (int, np.integer)):
        return _num_br(float(v), 0)
    return _num_br(float(v), casas)


def lista(titulo: str, itens: Iterable[str], codigo: bool = True) -> Markdown:
    """Título em negrito seguido de itens "- x"; `codigo` põe nomes de variável entre crases."""
    linhas = [f"**{titulo}**", ""]
    linhas += [f"- `{i}`" if codigo else f"- {i}" for i in itens]
    return Markdown("\n".join(linhas))
