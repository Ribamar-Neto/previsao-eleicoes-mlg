"""`carregar_base`: ids como texto, geometria na ordem de `df` e pesos com uma linha por município."""

import pandas as pd
from shapely.geometry import box

from src import dados as D


def test_carregar_base_reordena_geometria(tmp_path):
    pasta = tmp_path / "data/processed"
    pasta.mkdir(parents=True)
    pd.DataFrame({"id_municipio": [2, 1, 3], "v": [1.0, 2.0, 3.0]}).to_parquet(pasta / "municipios.parquet")
    geo = pd.DataFrame({"id_municipio": [1, 2, 3], "geometry": [box(0, 0, 1, 1).wkb, box(1, 0, 2, 1).wkb, box(2, 0, 3, 1).wkb]})
    geo.to_parquet(pasta / "municipios_geo.parquet")
    b = D.carregar_base(tmp_path)
    assert list(b.ids) == ["2", "1", "3"]
    assert list(b.gdf["id_municipio"]) == ["2", "1", "3"]
    assert b.gdf.geometry.iloc[0].equals(box(1, 0, 2, 1))
    assert b.W.n == 3
    assert b.gdf.crs.to_epsg() == 4674
