"""Perfil do eleitorado do TSE por município, para 2022 e 2026.

Regras de negócio:
- Cada ano usa o perfil da própria eleição (arquivo perfil_eleitorado_{ano}.zip),
  não uma atualização mensal posterior.
- Percentuais usam como denominador apenas quem informou o atributo: a categoria
  'NÃO INFORMADO' sai do denominador de cada atributo (e do numerador, claro).
- Eleitorado do exterior (SG_UF 'ZZ') é descartado: não há município.
- Escolaridade 'até fundamental incompleto' = analfabeto + lê e escreve +
  ensino fundamental incompleto.
- Os percentuais ficam em [0, 1], não em 0-100.
"""

import pandas as pd

from src.coleta.comum import INTERIM, RAW, baixar, ler_csv_zip, salvar_interim

URL = "https://cdn.tse.jus.br/estatistica/sead/odsele/perfil_eleitorado/perfil_eleitorado_{ano}.zip"
ANOS = (2022, 2026)
NAO_INFORMADO = "NÃO INFORMADO"
# Faixa etária sem idade válida conta como não informada.
INVALIDOS = {NAO_INFORMADO, "Inválida"}
CHAVE = ["SG_UF", "CD_MUNICIPIO"]
ATRIBUTOS = ["DS_GENERO", "DS_FAIXA_ETARIA", "DS_GRAU_ESCOLARIDADE"]
COLUNAS = [*CHAVE, "NR_ZONA", *ATRIBUTOS, "QT_ELEITORES"]

# Faixas etárias do TSE: 16 anos, 17 anos, 18 anos, 19 anos, 20 anos, 21 a 24 anos, ...
FAIXAS_16_24 = {"16 anos", "17 anos", "18 anos", "19 anos", "20 anos", "21 a 24 anos"}
FAIXAS_60_MAIS = {
    "60 a 64 anos", "65 a 69 anos", "70 a 74 anos", "75 a 79 anos",
    "80 a 84 anos", "85 a 89 anos", "90 a 94 anos", "95 a 99 anos",
    "100 anos ou mais",
}
ESC_ATE_FUND_INCOMPLETO = {"ANALFABETO", "LÊ E ESCREVE", "ENSINO FUNDAMENTAL INCOMPLETO"}


def _agregar_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["QT_ELEITORES"] = df["QT_ELEITORES"].astype("int64")
    # Zona não entra: o grão final é município. O exterior (ZZ) segue até
    # perfil_ano só para conferir o total nacional.
    return df.groupby(CHAVE + ATRIBUTOS, as_index=False)["QT_ELEITORES"].sum()


def _share(base: pd.DataFrame, coluna: str, categorias: set[str]) -> pd.Series:
    """Fração do eleitorado em `categorias`, sem 'NÃO INFORMADO' no denominador."""
    informado = base[~base[coluna].isin(INVALIDOS)]
    total = informado.groupby(CHAVE)["QT_ELEITORES"].sum()
    parte = informado[informado[coluna].isin(categorias)].groupby(CHAVE)["QT_ELEITORES"].sum()
    return (parte.reindex(total.index, fill_value=0) / total)


def perfil_ano(ano: int) -> tuple[pd.DataFrame, dict]:
    zip_path = baixar(URL.format(ano=ano), RAW / "tse_perfil" / f"perfil_eleitorado_{ano}.zip")
    partes = ler_csv_zip(
        zip_path, f"perfil_eleitorado_{ano}_BRASIL.csv",
        usecols=COLUNAS, filtro=_agregar_chunk,
    )
    base = partes.groupby(CHAVE + ATRIBUTOS, as_index=False)["QT_ELEITORES"].sum()
    base["CD_MUNICIPIO"] = base["CD_MUNICIPIO"].astype(int)
    print(f"{ano}: total nacional com exterior={base['QT_ELEITORES'].sum():,}")
    base = base[base["SG_UF"] != "ZZ"]

    for col in ATRIBUTOS:
        print(ano, col, sorted(base[col].unique()))
    nao_inf = {
        col: base.loc[base[col].isin(INVALIDOS), "QT_ELEITORES"].sum() / base["QT_ELEITORES"].sum()
        for col in ATRIBUTOS
    }

    out = pd.DataFrame({f"eleitorado_{ano}": base.groupby(CHAVE)["QT_ELEITORES"].sum()})
    out[f"pct_mulheres_{ano}"] = _share(base, "DS_GENERO", {"FEMININO"})
    out[f"pct_16_24_{ano}"] = _share(base, "DS_FAIXA_ETARIA", FAIXAS_16_24)
    out[f"pct_60_mais_{ano}"] = _share(base, "DS_FAIXA_ETARIA", FAIXAS_60_MAIS)
    out[f"pct_analfabeto_{ano}"] = _share(base, "DS_GRAU_ESCOLARIDADE", {"ANALFABETO"})
    out[f"pct_ate_fund_incompleto_{ano}"] = _share(
        base, "DS_GRAU_ESCOLARIDADE", ESC_ATE_FUND_INCOMPLETO
    )
    out[f"pct_superior_completo_{ano}"] = _share(
        base, "DS_GRAU_ESCOLARIDADE", {"SUPERIOR COMPLETO"}
    )
    return out.reset_index(), nao_inf


def main() -> pd.DataFrame:
    quadros = []
    for ano in ANOS:
        df, nao_inf = perfil_ano(ano)
        print(f"{ano}: municípios={len(df)} eleitorado doméstico={df[f'eleitorado_{ano}'].sum():,}")
        print(f"{ano}: share NÃO INFORMADO nacional={ {k: round(v, 4) for k, v in nao_inf.items()} }")
        quadros.append(df)
    perfil = quadros[0].merge(quadros[1], on=CHAVE, how="outer")
    perfil["id_municipio_tse"] = perfil["CD_MUNICIPIO"]
    perfil = perfil.rename(columns={"SG_UF": "sigla_uf"}).drop(columns="CD_MUNICIPIO")
    perfil = perfil[["id_municipio_tse", "sigla_uf", *[c for c in perfil if c not in ("id_municipio_tse", "sigla_uf")]]]
    assert perfil["id_municipio_tse"].is_unique
    pcts = [c for c in perfil if c.startswith("pct_")]
    valores = perfil[pcts].to_numpy(dtype=float)
    assert ((valores >= 0) | pd.isna(valores)).all() and ((valores <= 1) | pd.isna(valores)).all()
    salvar_interim(perfil, "tse_perfil")
    return perfil


if __name__ == "__main__":
    main()
