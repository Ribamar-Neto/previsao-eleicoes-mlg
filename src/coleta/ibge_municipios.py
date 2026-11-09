"""Municípios IBGE, tabela de equivalência TSE-IBGE e malha municipal.

Regras de negócio:
- O TSE usa códigos próprios de município; o IBGE usa 7 dígitos. A tabela de
  equivalência precisa ser 1-para-1 nos dois sentidos.
- A tabela pública (betafcc) é só ponto de partida: cada par é validado contra
  a lista real de municípios do TSE (2022 e 2026) e contra o nome/UF do IBGE.
  Pares que não passam, ou lacunas, são resolvidos por nome dentro da UF
  (`metodo` registra como cada par foi obtido).
- Voto no exterior (UF ZZ) não tem município IBGE e fica fora.
- Boa Esperança do Norte (MT) foi criado depois do Censo 2022: existe nas
  listas do IBGE (API atual) e do TSE, mas não na malha de 2022 nem no Censo.
  Fica marcado em `fora_censo_2022`; a etapa de junção o exclui.
- Área em CRS equivalente (Albers) e lat/lon de ponto representativo, que
  sempre cai dentro do polígono (o centróide pode cair fora em municípios
  côncavos).
"""

from __future__ import annotations

import re
import unicodedata

import geopandas as gpd
import pandas as pd

from src.coleta.comum import (
    RAIZ,
    RAW,
    REFERENCIA,
    baixar,
    ler_csv_zip,
    salvar_interim,
)

PASTA = RAW / "ibge_municipios"
URL_IBGE = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios?view=nivelado"
URL_BETAFCC = (
    "https://raw.githubusercontent.com/betafcc/Municipios-Brasileiros-TSE/"
    "master/municipios_brasileiros_tse.csv"
)
URL_TSE = (
    "https://cdn.tse.jus.br/estatistica/sead/odsele/detalhe_votacao_munzona/"
    "detalhe_votacao_munzona_{ano}.zip"
)
URL_MALHA = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
    "?intrarregiao=municipio&formato=application/vnd.geo+json&qualidade=intermediaria"
)
ANOS_TSE = (2022, 2026)
CRS_AREA = "ESRI:102033"
TOLERANCIA_SIMPLIFICACAO = 0.002  # graus; mantém o GeoParquet abaixo de ~30 MB


def normalizar(nome: str) -> str:
    """Maiúsculas, sem acento, hífen/apóstrofo/aspas tratados como espaço."""
    s = unicodedata.normalize("NFKD", nome)
    s = "".join(c for c in s if not unicodedata.combining(c)).upper()
    s = re.sub(r"[-'`´’]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ler_ibge() -> pd.DataFrame:
    destino = baixar(URL_IBGE, PASTA / "mun.json")
    df = pd.read_json(destino, dtype=False)  # ty: ignore[invalid-argument-type]  # pandas omite bool na anotação
    df = df.rename(
        columns={
            "municipio-id": "id_municipio",
            "municipio-nome": "nome_municipio",
            "UF-id": "id_uf",
            "UF-sigla": "sigla_uf",
            "UF-nome": "nome_uf",
            "regiao-nome": "regiao",
        }
    )
    df = df[["id_municipio", "nome_municipio", "id_uf", "sigla_uf", "nome_uf", "regiao"]]
    df["id_municipio"] = df["id_municipio"].astype(str)
    df["id_uf"] = df["id_uf"].astype(str)
    return df


def ler_tse() -> pd.DataFrame:
    """Municípios do TSE presentes nas votações de 2022 e 2026 (sem exterior)."""
    partes = []
    for ano in ANOS_TSE:
        zip_ = baixar(URL_TSE.format(ano=ano), PASTA / f"detalhe_votacao_munzona_{ano}.zip")
        df = ler_csv_zip(
            zip_,
            "*.csv",
            usecols=["SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO"],
            filtro=lambda d: d.drop_duplicates(),
        )
        df = df.drop_duplicates().rename(
            columns={
                "SG_UF": "sigla_uf",
                "CD_MUNICIPIO": "id_municipio_tse",
                "NM_MUNICIPIO": "nome_tse",
            }
        )
        df["ano"] = ano
        partes.append(df)
    tse = pd.concat(partes, ignore_index=True)
    tse = tse[tse["sigla_uf"] != "ZZ"].copy()
    tse["id_municipio_tse"] = tse["id_municipio_tse"].str.strip().str.lstrip("0")
    return tse


def montar_equivalencia(ibge: pd.DataFrame, tse: pd.DataFrame) -> pd.DataFrame:
    destino = baixar(URL_BETAFCC, PASTA / "tse_ref.csv")
    ref = pd.read_csv(destino, dtype=str)
    ref["codigo_tse"] = ref["codigo_tse"].str.lstrip("0")

    tse_u = (
        tse.sort_values("ano")
        .groupby(["sigla_uf", "id_municipio_tse"], as_index=False)
        .agg(nome_tse=("nome_tse", "last"))
    )
    ibge = ibge.assign(chave=ibge["nome_municipio"].map(normalizar))
    tse_u = tse_u.assign(chave=tse_u["nome_tse"].map(normalizar))

    # 1) Pares da tabela pública que IBGE e TSE confirmam (UF e nome iguais).
    base = tse_u.merge(
        ref[["codigo_tse", "codigo_ibge"]],
        left_on="id_municipio_tse",
        right_on="codigo_tse",
        how="left",
    ).merge(
        ibge[["id_municipio", "sigla_uf", "chave"]].rename(
            columns={"sigla_uf": "uf_ibge", "chave": "chave_ibge"}
        ),
        left_on="codigo_ibge",
        right_on="id_municipio",
        how="left",
    )
    ok = (base["uf_ibge"] == base["sigla_uf"]) & (base["chave_ibge"] == base["chave"])
    pares = base[ok].assign(metodo="betafcc_validado")
    sobra_tse = base[~ok][["sigla_uf", "id_municipio_tse", "nome_tse", "chave"]]

    # 2) Resto por nome dentro da UF, só entre os códigos IBGE ainda livres.
    livres = ibge[~ibge["id_municipio"].isin(pares["id_municipio"])]
    por_nome = sobra_tse.merge(
        livres[["id_municipio", "sigla_uf", "chave"]], on=["sigla_uf", "chave"], how="left"
    )
    por_nome["metodo"] = "nome_na_uf"
    pares = pd.concat([pares, por_nome[por_nome["id_municipio"].notna()]], ignore_index=True)
    sem_par = por_nome[por_nome["id_municipio"].isna()]
    if len(sem_par):
        print("TSE sem par IBGE por nome:")
        print(sem_par[["sigla_uf", "id_municipio_tse", "nome_tse"]].head(40).to_string())

    # 3) Código TSE da tabela pública existe na votação, mas o nome diverge do
    #    IBGE (grafia): aceita o par se o código ainda está livre.
    faltam = ibge[~ibge["id_municipio"].isin(pares["id_municipio"])].merge(
        ref[["codigo_tse", "codigo_ibge"]],
        left_on="id_municipio",
        right_on="codigo_ibge",
        how="left",
    )
    faltam = faltam[
        faltam["codigo_tse"].notna() & ~faltam["codigo_tse"].isin(pares["id_municipio_tse"])
    ]
    nomes_tse = tse_u.set_index("id_municipio_tse")["nome_tse"]
    extra = faltam.assign(
        id_municipio_tse=faltam["codigo_tse"],
        nome_tse=faltam["codigo_tse"].map(nomes_tse),
        metodo=faltam["codigo_tse"].map(
            lambda c: "betafcc_nome_divergente" if c in nomes_tse.index else "betafcc_sem_votacao"
        ),
    )
    pares = pd.concat([pares, extra], ignore_index=True)

    out = pares[["id_municipio", "id_municipio_tse", "nome_tse", "metodo"]].merge(
        ibge[["id_municipio", "nome_municipio", "sigla_uf"]].rename(
            columns={"nome_municipio": "nome_ibge"}
        ),
        on="id_municipio",
    )
    out = out[["id_municipio", "id_municipio_tse", "sigla_uf", "nome_ibge", "nome_tse", "metodo"]]
    return out.sort_values("id_municipio").reset_index(drop=True)


def checar(eq: pd.DataFrame, ibge: pd.DataFrame, tse: pd.DataFrame) -> None:
    print(f"IBGE: {len(ibge)} municípios; equivalência: {len(eq)} linhas")
    print(
        "duplicados IBGE:", eq["id_municipio"].duplicated().sum(),
        "| duplicados TSE:", eq["id_municipio_tse"].duplicated().sum(),
    )
    print("IBGE sem par:", sorted(set(ibge["id_municipio"]) - set(eq["id_municipio"]))[:20])
    print(eq["metodo"].value_counts().to_string())
    for ano in ANOS_TSE:
        cods = tse.loc[tse["ano"] == ano, ["sigla_uf", "id_municipio_tse", "nome_tse"]]
        cods = cods.drop_duplicates()
        faltam = cods[~cods["id_municipio_tse"].isin(eq["id_municipio_tse"])]
        print(f"TSE {ano}: {len(cods)} municípios, sem IBGE: {len(faltam)}")
        if len(faltam):
            print(faltam.head(20).to_string())
    manuais = eq[eq["metodo"] != "betafcc_validado"]
    print(f"pares fora de 'betafcc_validado': {len(manuais)}")
    print(manuais.head(40).to_string())


def ler_malha() -> gpd.GeoDataFrame:
    """Malha municipal; geobr primeiro, API de malhas do IBGE como alternativa.

    A qualidade intermediária basta para área (erro de fronteira pequeno) e
    para vizinhança; a versão mínima distorce a área.
    """
    try:
        import geobr

        malha = geobr.read_municipality(code_muni="all", year=2022)
        malha["id_municipio"] = malha["code_muni"].astype(int).astype(str)
        print(f"malha geobr 2022: {len(malha)} polígonos")
    except Exception as e:  # noqa: BLE001 - qualquer falha do geobr cai na API
        print(f"geobr falhou ({type(e).__name__}); usando API de malhas do IBGE")
        arq = baixar(URL_MALHA, PASTA / "malha_int.json")
        malha = gpd.read_file(arq)
        malha["id_municipio"] = malha["codarea"].astype(str)
        print(f"malha API IBGE: {len(malha)} polígonos")
    return malha[["id_municipio", "geometry"]].set_crs("EPSG:4674", allow_override=True)


def main() -> None:
    PASTA.mkdir(parents=True, exist_ok=True)
    ibge = ler_ibge()
    tse = ler_tse()
    eq = montar_equivalencia(ibge, tse)
    checar(eq, ibge, tse)
    REFERENCIA.mkdir(parents=True, exist_ok=True)
    eq.to_csv(REFERENCIA / "municipios_tse_ibge.csv", index=False)

    malha = ler_malha()
    ibge["fora_censo_2022"] = ~ibge["id_municipio"].isin(set(malha["id_municipio"]))
    fora = ibge.loc[ibge["fora_censo_2022"], ["id_municipio", "nome_municipio"]]
    print("fora da malha 2022:")
    print(fora.to_string())

    ponto = malha.geometry.representative_point()
    medidas = pd.DataFrame(
        {
            "id_municipio": malha["id_municipio"].values,
            "area_km2": (malha.to_crs(CRS_AREA).geometry.area / 1e6).values,
            "lat": ponto.y.values,
            "lon": ponto.x.values,
        }
    )
    interim = ibge.merge(
        eq[["id_municipio", "id_municipio_tse"]], on="id_municipio", how="left"
    ).merge(medidas, on="id_municipio", how="left")
    interim = interim[
        ["id_municipio", "id_municipio_tse", "nome_municipio", "sigla_uf", "id_uf",
         "nome_uf", "regiao", "fora_censo_2022", "area_km2", "lat", "lon"]
    ]
    salvar_interim(interim, "ibge_municipios")
    print(interim.groupby("regiao").size().to_string())

    processed = RAIZ / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    simples = malha.copy()
    simples["geometry"] = simples.geometry.simplify(
        TOLERANCIA_SIMPLIFICACAO, preserve_topology=True
    )
    destino = processed / "municipios_geo.parquet"
    simples.to_parquet(destino)
    print(f"geo: {len(simples)} linhas, {destino.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
