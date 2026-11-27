"""Carga comum dos notebooks `02-MLG` e `03-apuracao-parcial`: municípios, geometria e pesos de vizinhança."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src import espacial as S
from src.backtest import preparar


@dataclass(frozen=True, eq=False)
class Base:
    """Tabela de municípios, seus ids, a geometria na mesma ordem e os pesos Queen."""

    df: pd.DataFrame
    ids: pd.Series
    gdf: gpd.GeoDataFrame
    W: Any


def carregar_base(raiz: Path) -> Base:
    """Lê `data/processed`; a geometria é reordenada para a ordem de `df` (os pesos dependem dela)."""
    # `lag1t` e `outros_1t` (1º turno do próprio ciclo) são derivados aqui, não gravados no parquet
    df = preparar(pd.read_parquet(raiz / "data/processed/municipios.parquet")).reset_index(drop=True)
    ids = df["id_municipio"].astype(str)
    geo = pd.read_parquet(raiz / "data/processed/municipios_geo.parquet")
    geo["id_municipio"] = geo["id_municipio"].astype(str)
    geo = geo.set_index("id_municipio").loc[ids].reset_index()
    gdf = gpd.GeoDataFrame(geo, geometry=gpd.GeoSeries.from_wkb(geo["geometry"]), crs="EPSG:4674")
    return Base(df=df, ids=ids, gdf=gdf, W=S.pesos_queen(gdf))
