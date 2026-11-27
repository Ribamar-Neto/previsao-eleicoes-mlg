"""Boletim de urna (bweb) do 2º turno de 2022 e 2026, cargo Presidente, por seção.

Regras de negócio:
- Linha = seção eleitoral; seções agregadas ficam com a seção principal (a fonte
  traz só a principal, e `agregadas` guarda a lista da fonte).
- `dt_bu_recebido` é o recebimento do BU na totalização, em horário de Brasília
  (conferido: em todas as UFs o primeiro recebimento vem depois do fechamento
  das 17h de Brasília, mesmo no fuso local de 2 h atrás; `DT_ENCERRAMENTO` é local).
  Guardado com fuso America/Sao_Paulo; em outubro não há horário de verão desde 2019.
- Leiautes: 2022 traz data `dd/mm/aaaa hh:mm:ss` e `DS_AGREGADAS`; 2026 traz ISO e
  `DS_SECOES_AGREGADAS`. Os dois entram no mesmo esquema.
- Candidatos: Lula (13) e o bolsonarista (22 em 2022 e em 2026); `validos` = soma
  dos dois, único par de votáveis nominais do 2º turno. Brancos e nulos ficam fora.
- Só as 27 UFs (sem exterior). Não é execução de código dos zips: só leitura de CSV.
- Esquema validado por `validar_secoes` (Pandera não está nas dependências do projeto).
"""

from __future__ import annotations

import sys

import pandas as pd

from src.coleta.comum import RAIZ, RAW, baixar, ler_csv_zip, salvar_interim

DIR = RAW / "tse_bweb"
# O pandas não declara colunas: o nome diz o que o quadro carrega.
type BrutoBwebFrame = pd.DataFrame  # CSV do TSE como veio, só Presidente
type Secoes1TFrame = pd.DataFrame  # seção do 1º turno (COLUNAS_1T)
ANOS = (2022, 2026)
CARIMBO = {2022: "311020221535", 2026: "261020260742"}
CARIMBO_1T = {2022: "051020221321", 2026: "051020261403"}
UFS = [
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
]
BASE = "https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes{ano}/buweb/bweb_2t_{uf}_{carimbo}.zip"
BASE_1T = "https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes{ano}/buweb/bweb_1t_{uf}_{carimbo}.zip"
COLUNAS_1T = [
    "ano",
    "uf",
    "cd_municipio_tse",
    "zona",
    "secao",
    "votos_lula",
    "votos_bolsonaro",
    "validos",
    "validos_todos",
    "aptos",
    "comparecimento",
]
FUSO = "America/Sao_Paulo"
NR_LULA = "13"
NR_BOLSONARO = "22"
NAO_VALIDOS = ("95", "96", "97")  # branco, nulo, anulado e apurado em separado
FORMATO_DATA = {2022: "%d/%m/%Y %H:%M:%S", 2026: "%Y-%m-%d %H:%M:%S"}
COL_AGREGADAS = {2022: "DS_AGREGADAS", 2026: "DS_SECOES_AGREGADAS"}
CHAVE = ["ano", "uf", "cd_municipio_tse", "zona", "secao"]
COLUNAS = [
    "ano",
    "uf",
    "cd_municipio_tse",
    "id_municipio",
    "zona",
    "secao",
    "dt_bu_recebido",
    "aptos",
    "comparecimento",
    "votos_lula",
    "votos_bolsonaro",
    "validos",
    "agregadas",
]
NUMERICAS = ["aptos", "comparecimento", "votos_lula", "votos_bolsonaro", "validos"]
USECOLS_BASE = [
    "SG_UF",
    "CD_MUNICIPIO",
    "NR_ZONA",
    "NR_SECAO",
    "CD_CARGO_PERGUNTA",
    "DT_BU_RECEBIDO",
    "QT_APTOS",
    "QT_COMPARECIMENTO",
    "NR_VOTAVEL",
    "QT_VOTOS",
]


def parse_data(serie: pd.Series, ano: int) -> pd.Series:
    """Converte `DT_BU_RECEBIDO` do leiaute do `ano` em timestamp com fuso de Brasília."""
    dt = pd.to_datetime(serie, format=FORMATO_DATA[ano], errors="raise")
    return dt.dt.tz_localize(FUSO)


def _so_presidente(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["CD_CARGO_PERGUNTA"] == "1"]


def normalizar(bruto: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Do CSV bruto (linhas votável x seção) para uma linha por seção."""
    p = _so_presidente(bruto).copy()
    p["QT_VOTOS"] = pd.to_numeric(p["QT_VOTOS"]).astype("int64")
    chave = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
    atrib = chave + ["DT_BU_RECEBIDO", "QT_APTOS", "QT_COMPARECIMENTO", COL_AGREGADAS[ano]]
    secoes = p[atrib].drop_duplicates()
    if secoes.duplicated(chave).any():
        raise ValueError("seção com mais de um BU/atributo divergente")
    nominais = p[p["NR_VOTAVEL"].isin([NR_LULA, NR_BOLSONARO])]
    votos = nominais.pivot_table(
        index=chave, columns="NR_VOTAVEL", values="QT_VOTOS", aggfunc="sum", fill_value=0
    ).reindex(columns=[NR_LULA, NR_BOLSONARO], fill_value=0)
    votos.columns = ["votos_lula", "votos_bolsonaro"]
    s = secoes.join(votos, on=chave)
    s[["votos_lula", "votos_bolsonaro"]] = s[["votos_lula", "votos_bolsonaro"]].fillna(0)
    out = pd.DataFrame(
        {
            "ano": ano,
            "uf": s["SG_UF"].to_numpy(),
            "cd_municipio_tse": s["CD_MUNICIPIO"].astype("int64").to_numpy(),
            "zona": s["NR_ZONA"].astype("int64").to_numpy(),
            "secao": s["NR_SECAO"].astype("int64").to_numpy(),
            "dt_bu_recebido": parse_data(s["DT_BU_RECEBIDO"], ano).to_numpy(),
            "aptos": pd.to_numeric(s["QT_APTOS"]).astype("int64").to_numpy(),
            "comparecimento": pd.to_numeric(s["QT_COMPARECIMENTO"]).astype("int64").to_numpy(),
            "votos_lula": s["votos_lula"].astype("int64").to_numpy(),
            "votos_bolsonaro": s["votos_bolsonaro"].astype("int64").to_numpy(),
            "agregadas": s[COL_AGREGADAS[ano]].where(s[COL_AGREGADAS[ano]] != "#NULO#").to_numpy(),
        }
    )
    out["validos"] = out["votos_lula"] + out["votos_bolsonaro"]
    return out.reset_index(drop=True)


def _ler_uf(uf: str, ano: int) -> pd.DataFrame:
    destino = DIR / f"bweb_2t_{uf}_{ano}.zip"
    baixar(BASE.format(ano=ano, uf=uf, carimbo=CARIMBO[ano]), destino)
    cols = USECOLS_BASE + [COL_AGREGADAS[ano]]
    # O filtro do cargo entra em cada bloco; a seção pode cruzar blocos, então normaliza depois.
    bruto = ler_csv_zip(destino, "bweb_*.csv", usecols=cols, filtro=_so_presidente)
    return normalizar(bruto, ano)


def validar_secoes(df: pd.DataFrame) -> pd.DataFrame:
    """Esquema da seção: colunas, tipos, chave única e identidades. Levanta ValueError."""
    faltam = set(COLUNAS) - set(df.columns)
    if faltam:
        raise ValueError(f"colunas ausentes: {sorted(faltam)}")
    if str(df["dt_bu_recebido"].dtype) != f"datetime64[ns, {FUSO}]" and "datetime64" not in str(
        df["dt_bu_recebido"].dtype
    ):
        raise ValueError("dt_bu_recebido deve ser datetime com fuso")
    if df["dt_bu_recebido"].isna().any():
        raise ValueError("dt_bu_recebido nulo")
    if df.duplicated(CHAVE).any():
        raise ValueError("chave (ano, uf, município, zona, seção) duplicada")
    if (df[NUMERICAS] < 0).any().any():
        raise ValueError("contagem negativa")
    if (df["validos"] != df["votos_lula"] + df["votos_bolsonaro"]).any():
        raise ValueError("validos != lula + bolsonaro")
    if (df["validos"] > df["comparecimento"]).any() or (df["comparecimento"] > df["aptos"]).any():
        raise ValueError("validos <= comparecimento <= aptos violado")
    if df["id_municipio"].isna().any():
        raise ValueError("município sem código IBGE")
    return df


def coletar() -> pd.DataFrame:
    """Baixa (com cache), normaliza e valida as seções dos dois anos."""
    ident = pd.read_parquet(RAIZ / "data" / "interim" / "ibge_municipios.parquet")
    de_tse = ident.assign(cd_municipio_tse=ident["id_municipio_tse"].astype("int64"))
    de_tse = de_tse.set_index("cd_municipio_tse")["id_municipio"].astype("int64")
    partes = [_ler_uf(uf, ano) for ano in ANOS for uf in UFS]
    df = pd.concat(partes, ignore_index=True)
    # Município novo (Boa Esperança do Norte, MT) não está na base do Censo: fica sem IBGE.
    df["id_municipio"] = df["cd_municipio_tse"].map(de_tse).astype("Int64")
    return df


def normalizar_1t(bruto: BrutoBwebFrame, ano: int) -> Secoes1TFrame:
    """Seção do 1º turno: só Presidente, `validos` = Lula + Bolsonaro como no 2º turno.

    Atenção: no 1º turno há outros candidatos, então `validos` NÃO é o total de válidos
    do município; serve para medir o share Lula x Bolsonaro, não o comparecimento válido.
    `validos_todos` é o total de válidos (todos os candidatos), denominador de `outros_1t`.
    """
    out = normalizar(bruto, ano)
    p = _so_presidente(bruto)
    p = p[~p["NR_VOTAVEL"].isin(NAO_VALIDOS)]
    chave = {"SG_UF": "uf", "CD_MUNICIPIO": "cd_municipio_tse", "NR_ZONA": "zona", "NR_SECAO": "secao"}
    tot = (
        p[list(chave)]
        .rename(columns=chave)
        .astype({"cd_municipio_tse": "int64", "zona": "int64", "secao": "int64"})
        .assign(validos_todos=pd.to_numeric(p["QT_VOTOS"]).astype("int64").to_numpy())
        .groupby(list(chave.values()), as_index=False)["validos_todos"]
        .sum()
    )
    out = out.merge(tot, on=list(chave.values()), how="left", validate="one_to_one")
    out["validos_todos"] = out["validos_todos"].fillna(0).astype("int64")
    return out[COLUNAS_1T]


def coletar_1t() -> Secoes1TFrame:
    """Baixa (com cache) e normaliza as seções do 1º turno de 2022 e 2026 (só Presidente)."""
    partes = [normalizar_1t(_ler_uf_bruto_1t(uf, ano), ano) for ano in ANOS for uf in UFS]
    return pd.concat(partes, ignore_index=True).sort_values(CHAVE).reset_index(drop=True)


def _ler_uf_bruto_1t(uf: str, ano: int) -> BrutoBwebFrame:
    destino = DIR / f"bweb_1t_{uf}_{ano}.zip"
    baixar(BASE_1T.format(ano=ano, uf=uf, carimbo=CARIMBO_1T[ano]), destino)
    cols = USECOLS_BASE + [COL_AGREGADAS[ano]]
    return ler_csv_zip(destino, "bweb_*.csv", usecols=cols, filtro=_so_presidente)


def validar_secoes_1t(df: Secoes1TFrame) -> Secoes1TFrame:
    """Esquema do 1º turno: colunas, chave única e contagens coerentes. Levanta ValueError."""
    faltam = set(COLUNAS_1T) - set(df.columns)
    if faltam:
        raise ValueError(f"colunas ausentes: {sorted(faltam)}")
    duplicadas = int(df.duplicated(CHAVE).sum())
    if duplicadas:
        raise ValueError(f"chave (ano, uf, município, zona, seção) duplicada em {duplicadas} linhas")
    negativas = int((df[NUMERICAS] < 0).any(axis=1).sum())
    if negativas:
        raise ValueError(f"contagem negativa em {negativas} linhas")
    soma_errada = int((df["validos"] != df["votos_lula"] + df["votos_bolsonaro"]).sum())
    if soma_errada:
        raise ValueError(f"validos != lula + bolsonaro em {soma_errada} linhas")
    fora_da_ordem = int(((df["validos"] > df["comparecimento"]) | (df["comparecimento"] > df["aptos"])).sum())
    if fora_da_ordem:
        raise ValueError(f"validos <= comparecimento <= aptos violado em {fora_da_ordem} linhas")
    fora_dos_todos = int(((df["validos_todos"] < df["validos"]) | (df["validos_todos"] > df["comparecimento"])).sum())
    if fora_dos_todos:
        raise ValueError(f"validos <= validos_todos <= comparecimento violado em {fora_dos_todos} linhas")
    return df


def curva(secoes: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Curva nacional de apuração (domestica): acumulado por instante de recebimento.

    `pct_recebido` = % dos votos válidos (dois candidatos) já recebidos; `share_lula`
    = Lula / (Lula + Bolsonaro) no acumulado. O denominador não inclui o exterior.
    """
    s = secoes[secoes["ano"] == ano].sort_values("dt_bu_recebido")
    por_instante = s.groupby("dt_bu_recebido")[["votos_lula", "validos"]].sum().cumsum()
    out = pd.DataFrame(
        {
            "dt_bu_recebido": por_instante.index,
            "pct_recebido": 100 * por_instante["validos"].to_numpy() / s["validos"].sum(),
            "share_lula": 100 * por_instante["votos_lula"].to_numpy() / por_instante["validos"].to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def virada(c: pd.DataFrame) -> pd.Series | None:
    """Último instante em que Lula passa de <= 50% a > 50% (None se nunca)."""
    acima = c["share_lula"] > 50
    cruz = acima & ~acima.shift(fill_value=False)
    return c[cruz].iloc[-1] if cruz.any() else None


def checar_totais(secoes: pd.DataFrame) -> pd.DataFrame:
    """Compara totais por município com `votos_pt_2t_<ano>` e `n_2t_<ano>` do parquet do modelo."""
    modelo = pd.read_parquet(RAIZ / "data" / "processed" / "municipios.parquet")
    linhas = []
    for ano in ANOS:
        g = secoes[secoes["ano"] == ano].groupby("cd_municipio_tse")[["votos_lula", "validos"]].sum()
        m = modelo.set_index(modelo["id_municipio_tse"].astype("int64"))
        cmp = g.join(m[[f"votos_pt_2t_{ano}", f"n_2t_{ano}", "sigla_uf", "nome_municipio"]], how="outer")
        cmp["ano"] = ano
        cmp["dif_votos"] = cmp["votos_lula"].fillna(0) - cmp[f"votos_pt_2t_{ano}"].fillna(0)
        cmp["dif_n"] = cmp["validos"].fillna(0) - cmp[f"n_2t_{ano}"].fillna(0)
        linhas.append(cmp[(cmp["dif_votos"] != 0) | (cmp["dif_n"] != 0)].reset_index())
    return pd.concat(linhas, ignore_index=True)


def main_1t() -> None:
    df = validar_secoes_1t(coletar_1t())
    salvar_interim(df, "tse_bweb_1t")
    print(df.groupby("ano").size())


def main() -> None:
    if "--turno" in sys.argv and sys.argv[sys.argv.index("--turno") + 1] == "1":
        main_1t()
        return
    df = coletar()
    sem_ibge = df[df["id_municipio"].isna()][["uf", "cd_municipio_tse"]].drop_duplicates()
    print("municípios TSE sem IBGE:\n", sem_ibge.to_string())
    ok = df.dropna(subset=["id_municipio"]).copy()
    validar_secoes(ok)
    salvar_interim(df.sort_values(CHAVE).reset_index(drop=True), "tse_bweb")
    print(df.groupby("ano").size())
    saida = RAIZ / "results" / "tables"
    for ano in ANOS:
        c = curva(df, ano)
        c.to_csv(saida / f"apuracao_curva_{ano}.csv", index=False)
        v = virada(c)
        print(ano, "virada:", None if v is None else v.to_dict(), "| final:", c.iloc[-1].to_dict())
    print("divergências de total:\n", checar_totais(df).to_string())


if __name__ == "__main__":
    main()
