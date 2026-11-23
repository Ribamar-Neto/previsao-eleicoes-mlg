"""Gravação padronizada das saídas dos notebooks (tabelas CSV, figuras HTML e JSON).

Regra de negócio: os arquivos em results/ são versionados e conferidos byte a byte; por isso
a escrita fica num só lugar (mesmo `include_plotlyjs="cdn"`, `div_id` igual ao nome do arquivo,
JSON com `indent=2` e `ensure_ascii=False`), e os notebooks não repetem esses parâmetros.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
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
    """Grava `<pasta>/<nome>.html` com `div_id=nome`."""
    caminho = Path(pasta) / f"{nome}.html"
    fig.write_html(caminho, include_plotlyjs="cdn", div_id=nome)
    return caminho


def salvar_json(obj: Any, caminho: str | Path) -> Path:
    """Grava `obj` em JSON legível; valores fora do padrão JSON viram texto."""
    destino = Path(caminho)
    destino.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return destino
