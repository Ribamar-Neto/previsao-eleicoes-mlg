"""Ideologia por município: voto em deputado federal por bloco e partido do prefeito.

Regras de negócio:
- Bloco do partido vem de data/referencia/ideologia_partidos.csv (Bolognesi, Ribeiro e
  Codato 2023), colapsado em esquerda/centro/direita.
- Votos de partido = nominais + legenda válidos (TSE registra por partido-membro, então
  federações não precisam de tratamento especial). Voto em trânsito (ZZ/BR/VT) fica fora.
- Prefeito = eleito da eleição ordinária (ou da suplementar, se a ordinária foi anulada); em municípios com 2º turno vale o 2º turno
  (o eleito é o único com DS_SIT_TOT_TURNO 'ELEITO' na linha do turno decisivo).
- Município sem prefeito eleito (Brasília, Noronha) recebe 'sem_prefeito'.
"""

import pandas as pd

from src.coleta.comum import RAW, REFERENCIA, baixar, ler_csv_zip, salvar_interim

DIR = RAW / "tse_ideologia"
URL_VOT = "https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_partido_munzona/votacao_partido_munzona_{ano}.zip"
URL_CAND = "https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand/consulta_cand_{ano}.zip"
FORA = {"ZZ", "BR", "VT"}
BLOCOS = {"esquerda": "esq", "centro": "centro", "direita": "dir"}
CLASSES = {"extrema-esquerda": "ext_esq", "esquerda": "esq", "centro-esquerda": "centro_esq",
           "centro": "centro", "centro-direita": "centro_dir", "direita": "dir",
           "extrema-direita": "ext_dir"}
# Cortes do artigo (limite superior de cada classe), usados para fusões sem classe própria.
CORTES = [(1.5, "extrema-esquerda"), (3.0, "esquerda"), (4.49, "centro-esquerda"),
          (5.5, "centro"), (7.0, "centro-direita"), (8.5, "direita"), (10.0, "extrema-direita")]
# Sem nota no artigo: classe atribuída pelo projeto.
CLASSE_MANUAL = {"UP": "esquerda", "MISSÃO": "direita"}


def _classe(linha: pd.Series) -> str:
    if linha["sigla_partido"] in CLASSE_MANUAL:
        return CLASSE_MANUAL[linha["sigla_partido"]]
    if pd.notna(linha["classe_brc"]):
        return linha["classe_brc"]
    return next(c for lim, c in CORTES if float(linha["nota_brc"]) <= lim)


def carregar_notas_classes() -> tuple[dict[str, float], dict[str, str]]:
    ref = pd.read_csv(REFERENCIA / "ideologia_partidos.csv", dtype=str)
    classes = {r["sigla_partido"]: CLASSES[_classe(r)] for _, r in ref.iterrows()}
    notas = ref.dropna(subset=["nota_brc"]).set_index("sigla_partido")["nota_brc"].astype(float).to_dict()
    return notas, classes


def carregar_blocos() -> dict[str, str]:
    ref = pd.read_csv(REFERENCIA / "ideologia_partidos.csv", dtype=str)
    return dict(zip(ref["sigla_partido"], ref["bloco"]))


def votos_deputado(ano: int, blocos: dict[str, str]) -> pd.DataFrame:
    zip_path = baixar(URL_VOT.format(ano=ano), DIR / f"vot_{ano}.zip")
    df = ler_csv_zip(
        zip_path,
        f"votacao_partido_munzona_{ano}_BRASIL.csv",
        usecols=["SG_UF", "CD_MUNICIPIO", "DS_CARGO", "NR_TURNO", "SG_PARTIDO",
                 "QT_VOTOS_LEGENDA_VALIDOS", "QT_VOTOS_NOMINAIS_VALIDOS"],
        filtro=lambda d: d[(d["DS_CARGO"] == "Deputado Federal") & (d["NR_TURNO"] == "1")],
    )
    df = df[~df["SG_UF"].isin(FORA)].copy()
    df["votos"] = (pd.to_numeric(df["QT_VOTOS_LEGENDA_VALIDOS"])
                   + pd.to_numeric(df["QT_VOTOS_NOMINAIS_VALIDOS"]))
    df["bloco"] = df["SG_PARTIDO"].map(blocos)
    df["id_municipio_tse"] = df["CD_MUNICIPIO"].astype(int)
    notas, classes = carregar_notas_classes()
    df["classe"] = df["SG_PARTIDO"].map(classes)
    df["nota"] = df["SG_PARTIDO"].map(notas)
    idx = ["id_municipio_tse", "SG_UF"]
    cl = df.pivot_table(index=idx, columns="classe", values="votos", aggfunc="sum", fill_value=0)
    cl = cl.reindex(columns=list(CLASSES.values()), fill_value=0)
    com_nota = df.dropna(subset=["nota"])
    soma_nota = (com_nota["votos"] * com_nota["nota"]).groupby([com_nota[c] for c in idx]).sum()
    votos_nota = com_nota.groupby(idx)["votos"].sum()
    print(f"{ano}: votos nacionais {df['votos'].sum():,.0f}; sem nota {df.loc[df['nota'].isna(), 'votos'].sum() / df['votos'].sum():.4f}; "
          f"nota media nacional {soma_nota.sum() / votos_nota.sum():.3f}")
    tab = df.pivot_table(index=idx, columns="bloco",
                         values="votos", aggfunc="sum", fill_value=0)
    for b in BLOCOS:
        if b not in tab:
            tab[b] = 0
    tab = tab[list(BLOCOS)].rename(columns={b: f"votos_dep_{s}_{ano}" for b, s in BLOCOS.items()})
    tab[f"votos_dep_validos_{ano}"] = tab.sum(axis=1)
    for s in BLOCOS.values():
        tab[f"pct_dep_{s}_{ano}"] = tab[f"votos_dep_{s}_{ano}"] / tab[f"votos_dep_validos_{ano}"]
    tab[f"nota_media_dep_{ano}"] = soma_nota / votos_nota
    for c in CLASSES.values():
        tab[f"pct_dep_{c}_{ano}"] = cl[c] / tab[f"votos_dep_validos_{ano}"]
    tab.attrs["nao_mapeados"] = sorted(df.loc[df["bloco"].isna(), "SG_PARTIDO"].unique())
    return tab.reset_index().rename(columns={"SG_UF": "sigla_uf"})


def prefeitos(ano: int, blocos: dict[str, str]) -> pd.DataFrame:
    zip_path = baixar(URL_CAND.format(ano=ano), DIR / f"cand_{ano}.zip")
    df = ler_csv_zip(
        zip_path,
        f"consulta_cand_{ano}_BRASIL.csv",
        usecols=["SG_UF", "SG_UE", "NM_TIPO_ELEICAO", "NR_TURNO", "DS_CARGO",
                 "SG_PARTIDO", "DS_SIT_TOT_TURNO"],
        filtro=lambda d: d[(d["DS_CARGO"].str.upper() == "PREFEITO")
                           & (d["DS_SIT_TOT_TURNO"].str.startswith("ELEITO", na=False))],
    )
    df = df[~df["SG_UF"].isin(FORA)].copy()
    df["id_municipio_tse"] = df["SG_UE"].astype(int)
    df["suplementar"] = ~df["NM_TIPO_ELEICAO"].str.upper().str.contains("ORDIN", na=False)
    # Eleição ordinária anulada: o prefeito em exercício é o eleito na suplementar.
    df = df.sort_values(["suplementar", "NR_TURNO"], ascending=[False, True])
    df = df.drop_duplicates("id_municipio_tse", keep="last")
    df[f"prefeito_suplementar_{ano}"] = df["suplementar"]
    df[f"partido_prefeito_{ano}"] = df["SG_PARTIDO"]
    df[f"bloco_prefeito_{ano}"] = df["SG_PARTIDO"].map(blocos)
    out = df[["id_municipio_tse", f"partido_prefeito_{ano}", f"bloco_prefeito_{ano}",
              f"prefeito_suplementar_{ano}"]].copy()
    out.attrs["nao_mapeados"] = sorted(df.loc[df[f"bloco_prefeito_{ano}"].isna(), "SG_PARTIDO"].unique())
    return out


def main() -> None:
    blocos = carregar_blocos()
    base = None
    nao_map = set()
    for ano in (2018, 2022, 2026):
        t = votos_deputado(ano, blocos)
        nao_map |= set(t.attrs["nao_mapeados"])
        base = t if base is None else base.merge(t, on=["id_municipio_tse", "sigla_uf"], how="outer")
    for ano in (2020, 2024):
        p = prefeitos(ano, blocos)
        nao_map |= set(p.attrs["nao_mapeados"])
        base = base.merge(p, on="id_municipio_tse", how="outer")
        base[f"bloco_prefeito_{ano}"] = base[f"bloco_prefeito_{ano}"].fillna("sem_prefeito")
    print("partidos sem bloco:", sorted(nao_map))
    salvar_interim(base, "tse_ideologia")
    print(base.shape)


if __name__ == "__main__":
    main()
