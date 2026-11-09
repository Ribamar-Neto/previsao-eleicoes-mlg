"""Funções comuns aos scripts de coleta.

Regras:
- Cada fonte baixa para a sua pasta `data/raw/<fonte>/` e grava um único
  `data/interim/<fonte>.parquet` com uma linha por município.
- Download com cache: um arquivo que já existe não é baixado de novo.
- CSVs do TSE: latin-1, separador `;`, divididos por UF dentro do zip.
"""

from __future__ import annotations

import fnmatch
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
INTERIM = RAIZ / "data" / "interim"
REFERENCIA = RAIZ / "data" / "referencia"
UA = {"User-Agent": "previsao-eleicoes/0.1 (projeto academico)"}


def baixar(url: str, destino: Path, timeout: int = 600) -> Path:
    """Baixa `url` para `destino`, se ainda não existir. Devolve o caminho."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    parcial = destino.with_suffix(destino.suffix + ".part")
    with requests.get(url, headers=UA, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with parcial.open("wb") as f:
            for bloco in r.iter_content(chunk_size=1 << 20):
                f.write(bloco)
    parcial.rename(destino)
    return destino


def ler_csv_zip(
    caminho_zip: Path,
    padrao: str,
    usecols: list[str] | None = None,
    filtro: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    chunksize: int = 500_000,
    **kwargs,
) -> pd.DataFrame:
    """Lê e concatena os membros do zip cujo nome casa com `padrao` (glob).

    Lê em blocos e aplica `filtro` a cada bloco, para não carregar o arquivo
    inteiro na memória.
    """
    kwargs.setdefault("sep", ";")
    kwargs.setdefault("encoding", "latin-1")
    kwargs.setdefault("dtype", str)
    partes = []
    with zipfile.ZipFile(caminho_zip) as z:
        membros = [m for m in z.namelist() if fnmatch.fnmatch(Path(m).name, padrao)]
        if not membros:
            raise FileNotFoundError(f"nenhum membro casa com {padrao!r} em {caminho_zip}")
        for m in membros:
            with z.open(m) as f:
                for bloco in pd.read_csv(f, usecols=usecols, chunksize=chunksize, **kwargs):
                    partes.append(filtro(bloco) if filtro else bloco)
    return pd.concat(partes, ignore_index=True)


def salvar_interim(df: pd.DataFrame, nome: str) -> Path:
    """Grava `data/interim/<nome>.parquet`."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    destino = INTERIM / f"{nome}.parquet"
    df.to_parquet(destino, index=False)
    return destino
