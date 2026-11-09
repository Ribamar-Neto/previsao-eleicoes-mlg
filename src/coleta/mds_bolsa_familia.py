"""Famílias beneficiárias do Auxílio Brasil (2022) / Bolsa Família (2026) por município.

Medida única para os dois pleitos: nº de famílias pagas no mês anterior ao 1º turno.
Mês de referência: 202209 (Auxílio Brasil) e 202609 (Novo Bolsa Familia), AAAAMM.

Fonte: Portal da Transparência (CGU), arquivos de pagamento individual. A API
MI Social do MDS redireciona para uma página de avisos (indisponível), então os
agregados municipais do MDS não puderam ser usados.

Regras:
- Família = NIS distinto do responsável familiar que recebeu pagamento no mês de
  competência. Parcelas retroativas (MÊS REFERÊNCIA anterior) repetem o NIS e
  não contam duas vezes.
- Município = município de pagamento, não necessariamente o de residência.
- O arquivo traz código SIAFI, não IBGE. O cruzamento com o código IBGE de 7
  dígitos é por UF + nome normalizado (a API do IBGE), com aproximação textual
  (>= 0,9) só dentro da mesma UF para grafias divergentes.
- Cadastro Único não existe nessa fonte: as colunas familias_cadunico_* ficam
  fora do resultado (nunca preenchidas com dado inventado).
"""

import difflib
import re
import unicodedata
import zipfile

import pandas as pd
import requests

from src.coleta.comum import INTERIM, RAW, UA, baixar, salvar_interim

PASTA = RAW / "mds_bolsa_familia"
MESES = {
    2022: ("202209", "auxilio-brasil-saques/202209_AuxilioBrasil.zip"),
    2026: ("202609", "novo-bolsa-familia/202609_NovoBolsaFamilia.zip"),
}
BASE = "https://dadosabertos-download.cgu.gov.br/PortalDaTransparencia/saida/"
URL_IBGE = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios?view=nivelado"
LIMITE_APROXIMACAO = 0.9

# Nomes SIAFI/antigos sem equivalente textual no IBGE (renomeações e grafias),
# chaveados por (UF, SIAFI) e conferidos manualmente.
ALIAS_SIAFI = {
    ("PB", "0522"): 2513968,  # São Domingos de Pombal -> São Domingos
    ("PB", "2209"): 2515401,  # Seridó -> São Vicente do Seridó
    ("PE", "3001"): 2605459,  # Distrito Estadual de Fernando de Noronha
    ("RJ", "5875"): 3303807,  # Parati -> Paraty
    ("RN", "1603"): 2400208,  # Açu -> Assú
    ("RN", "1623"): 2401206,  # Ares -> Arez
    ("RN", "1625"): 2401305,  # Campo Grande (ex-Augusto Severo)
    ("RN", "1805"): 2410306,  # Serra Caiada (ex-Presidente Juscelino)
    ("RR", "0315"): 1400605,  # São Luiz -> São Luiz do Anauá
    ("SP", "6401"): 3515004,  # Embu -> Embu das Artes
    ("TO", "0345"): 1708254,  # Fortaleza do Tabocão -> Tabocão
    ("TO", "9691"): 1720499,  # São Valério da Natividade -> São Valério
}


def _norm(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", sem_acento.upper())


def _municipios_ibge() -> pd.DataFrame:
    cache = PASTA / "ibge_municipios.json"
    if not cache.exists():
        resp = requests.get(URL_IBGE, headers=UA, timeout=60)
        resp.raise_for_status()
        PASTA.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
    df = pd.read_json(cache, dtype=False)
    df = df.rename(columns={"municipio-id": "id_municipio", "municipio-nome": "nome", "UF-sigla": "uf"})
    df["id_municipio"] = df["id_municipio"].astype(int)
    df["chave"] = df["uf"] + "|" + df["nome"].map(_norm)
    return df[["id_municipio", "uf", "nome", "chave"]]


def _familias_por_municipio(zip_path) -> pd.DataFrame:
    """Conta NIS distintos por (UF, SIAFI, nome) lendo o CSV em blocos."""
    partes = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            blocos = pd.read_csv(
                f, sep=";", encoding="latin-1", dtype=str, chunksize=2_000_000,
                usecols=["UF", "CÓDIGO MUNICÍPIO SIAFI", "NOME MUNICÍPIO", "NIS FAVORECIDO"],
            )
            for b in blocos:
                b.columns = ["uf", "siafi", "nome", "nis"]
                partes.append(b.drop_duplicates())
    tudo = pd.concat(partes, ignore_index=True).drop_duplicates()
    return tudo.groupby(["uf", "siafi", "nome"], as_index=False).agg(familias=("nis", "nunique"))


def _casar_ibge(contagem: pd.DataFrame, ibge: pd.DataFrame) -> pd.DataFrame:
    contagem = contagem.copy()
    contagem["chave"] = contagem["uf"] + "|" + contagem["nome"].map(_norm)
    por_chave = dict(zip(ibge["chave"], ibge["id_municipio"]))
    contagem["id_municipio"] = contagem["chave"].map(por_chave)
    for i in contagem.index[contagem["id_municipio"].isna()]:
        alias = ALIAS_SIAFI.get((contagem.at[i, "uf"], contagem.at[i, "siafi"]))
        if alias:
            contagem.at[i, "id_municipio"] = alias
            continue
        uf = contagem.at[i, "uf"]
        candidatas = [c for c in por_chave if c.startswith(uf + "|")]
        achada = difflib.get_close_matches(contagem.at[i, "chave"], candidatas, n=1, cutoff=LIMITE_APROXIMACAO)
        if achada:
            contagem.at[i, "id_municipio"] = por_chave[achada[0]]
    sem_codigo = contagem[contagem["id_municipio"].isna()]
    if len(sem_codigo):
        print(f"Sem código IBGE ({len(sem_codigo)} municípios, {sem_codigo['familias'].sum()} famílias):")
        print(sem_codigo[["uf", "siafi", "nome", "familias"]].head(20).to_string())
    contagem = contagem.dropna(subset=["id_municipio"])
    contagem["id_municipio"] = contagem["id_municipio"].astype(int)
    return contagem.groupby("id_municipio", as_index=False)["familias"].sum()


def main() -> dict:
    ibge = _municipios_ibge()
    resultado = ibge[["id_municipio"]].drop_duplicates().sort_values("id_municipio").reset_index(drop=True)
    for ano, (mes, caminho) in MESES.items():
        zip_path = PASTA / caminho.split("/")[-1]
        baixar(BASE + caminho, zip_path)
        contagem = _casar_ibge(_familias_por_municipio(zip_path), ibge)
        resultado = resultado.merge(
            contagem.rename(columns={"familias": f"familias_pbf_{ano}"}), on="id_municipio", how="left"
        )
    # Município sem pagamento no mês = 0 famílias
    colunas = [c for c in resultado.columns if c.startswith("familias_pbf_")]
    resultado[colunas] = resultado[colunas].fillna(0).astype("int64")
    assert resultado["id_municipio"].is_unique
    salvar_interim(resultado, "mds_bolsa_familia")
    totais = {c: int(resultado[c].sum()) for c in colunas}
    print(len(resultado), "linhas;", totais)
    return {"meses_referencia": {a: m for a, (m, _) in MESES.items()}, "totais": totais, "linhas": len(resultado)}


if __name__ == "__main__":
    main()
