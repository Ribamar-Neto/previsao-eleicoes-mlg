"""Votação presidencial por município (TSE, 1º e 2º turnos de 2018, 2022 e 2026).

Regras de negócio:
- Os zips trazem um arquivo `_BRASIL.csv` (consolidado) além dos arquivos por UF;
  só os por UF são lidos, para não duplicar votos. O exterior (SG_UF == 'ZZ')
  entra na checagem nacional, mas é removido da saída municipal.
- `validos` = soma dos votos nominais válidos de todos os candidatos a presidente
  (no 1º turno, só PT e Bolsonaro são mantidos individualmente).
- Candidato do bolsonarismo: 17 em 2018, 22 em 2022 e 2026; PT é sempre 13.
- Zonas são somadas por município (a fonte é município x zona).
"""
import pandas as pd

from src.coleta.comum import RAIZ, RAW, baixar, ler_csv_zip, salvar_interim

BASE = "https://cdn.tse.jus.br/estatistica/sead/odsele"
DIR = RAW / "tse_presidente"
ANOS = (2018, 2022, 2026)
NR_BOLSONARO = {2018: "17", 2022: "22", 2026: "22"}
OFICIAL = {  # 2º turno, votos válidos incluindo exterior
    2022: (60_345_999, 58_206_354),
    2018: (47_040_906, 57_797_847),
}
CHAVE = ["CD_MUNICIPIO", "SG_UF", "NM_MUNICIPIO"]


def _so_presidente(df):
    return df[df["CD_CARGO"] == "1"]


def _ler(tipo, ano, usecols):
    destino = DIR / f"{tipo}_{ano}.zip"
    baixar(f"{BASE}/{tipo}/{tipo}_{ano}.zip", destino)
    df = ler_csv_zip(destino, f"{tipo}_{ano}_[A-Z][A-Z].csv", usecols=usecols, filtro=_so_presidente)
    # Alguns anos preenchem o código com zeros à esquerda; normaliza para casar as fontes.
    df["CD_MUNICIPIO"] = df["CD_MUNICIPIO"].astype(int)
    return df


def _candidatos(ano):
    cols = ["NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO", "CD_CARGO", "NR_CANDIDATO",
            "QT_VOTOS_NOMINAIS_VALIDOS"]
    df = _ler("votacao_candidato_munzona", ano, cols)
    df["votos"] = pd.to_numeric(df["QT_VOTOS_NOMINAIS_VALIDOS"]).astype("int64")
    df["nr"] = df["NR_CANDIDATO"]
    df["grupo"] = df["nr"].map({"13": "votos_pt", NR_BOLSONARO[ano]: "votos_bolsonaro"})
    return df


def _detalhe(ano):
    cols = ["NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO", "CD_CARGO", "QT_APTOS",
            "QT_COMPARECIMENTO", "QT_ABSTENCOES", "QT_VOTOS_BRANCOS", "QT_TOTAL_VOTOS_NULOS"]
    df = _ler("detalhe_votacao_munzona", ano, cols)
    ren = {"QT_APTOS": "aptos", "QT_COMPARECIMENTO": "comparecimento", "QT_ABSTENCOES": "abstencoes",
           "QT_VOTOS_BRANCOS": "brancos", "QT_TOTAL_VOTOS_NULOS": "nulos"}
    df = df.rename(columns=ren)
    for c in ren.values():
        df[c] = pd.to_numeric(df[c]).astype("int64")
    return df


def _municipal(ano):
    """Uma linha por município x turno, com as métricas do ano."""
    cand = _candidatos(ano)
    por_grupo = (cand.dropna(subset=["grupo"])
                 .pivot_table(index=CHAVE + ["NR_TURNO"], columns="grupo", values="votos",
                              aggfunc="sum", fill_value=0))
    validos = cand.groupby(CHAVE + ["NR_TURNO"])["votos"].sum().rename("validos")
    det = _detalhe(ano).groupby(CHAVE + ["NR_TURNO"])[
        ["aptos", "comparecimento", "abstencoes", "brancos", "nulos"]].sum()
    m = por_grupo.join(validos, how="outer").join(det, how="outer").reset_index()
    m["ano"] = ano
    return m


def _checagem(mun):
    """Totais nacionais (com e sem exterior) e por UF, com % dos votos válidos."""
    por_uf = mun.groupby(["ano", "NR_TURNO", "SG_UF"])[["votos_pt", "votos_bolsonaro", "validos"]].sum()
    por_uf = por_uf.reset_index()
    nac = por_uf.groupby(["ano", "NR_TURNO"])[["votos_pt", "votos_bolsonaro", "validos"]].sum().reset_index()
    nac["SG_UF"] = "BRASIL_COM_EXTERIOR"
    dom = (por_uf[por_uf["SG_UF"] != "ZZ"].groupby(["ano", "NR_TURNO"])
           [["votos_pt", "votos_bolsonaro", "validos"]].sum().reset_index())
    dom["SG_UF"] = "BRASIL_DOMESTICO"
    out = pd.concat([nac, dom, por_uf], ignore_index=True)
    out["pct_pt"] = 100 * out["votos_pt"] / out["validos"]
    out["pct_bolsonaro"] = 100 * out["votos_bolsonaro"] / out["validos"]
    out = out.rename(columns={"NR_TURNO": "turno", "SG_UF": "unidade"})
    return out.sort_values(["ano", "turno", "unidade"]).reset_index(drop=True)


def _largo(mun):
    dom = mun[mun["SG_UF"] != "ZZ"].copy()
    # Nome/UF variam entre anos para o mesmo código; vale o do ano mais recente.
    ident = (dom.sort_values("ano").drop_duplicates("CD_MUNICIPIO", keep="last")
             .set_index("CD_MUNICIPIO")[["SG_UF", "NM_MUNICIPIO"]])
    dom["CD_MUNICIPIO"] = dom["CD_MUNICIPIO"].astype(int)
    ident.index = ident.index.astype(int)
    metricas = ["votos_pt", "votos_bolsonaro", "validos", "aptos", "comparecimento",
                "abstencoes", "brancos", "nulos"]
    partes = []
    for (ano, turno), g in dom.groupby(["ano", "NR_TURNO"]):
        g = g.groupby("CD_MUNICIPIO")[metricas].sum(min_count=1)
        g.columns = [f"{c}_{turno}t_{ano}" for c in metricas]
        partes.append(g)
    largo = pd.concat(partes, axis=1).join(ident).reset_index()
    largo = largo.rename(columns={"CD_MUNICIPIO": "id_municipio_tse", "SG_UF": "sigla_uf",
                                  "NM_MUNICIPIO": "nome_municipio"})
    largo.insert(1, "sigla_uf", largo.pop("sigla_uf"))
    largo.insert(2, "nome_municipio", largo.pop("nome_municipio"))
    for c in largo.columns[3:]:
        largo[c] = largo[c].astype("Int64")
    return largo


def main():
    mun = pd.concat([_municipal(a) for a in ANOS], ignore_index=True)
    mun["NR_TURNO"] = mun["NR_TURNO"].astype(int)
    for c in ("votos_pt", "votos_bolsonaro"):
        mun[c] = mun[c].fillna(0)

    chk = _checagem(mun)
    destino = RAIZ / "results" / "tables" / "checagem_totais_presidente.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    chk.to_csv(destino, index=False)

    for ano, (pt, bol) in OFICIAL.items():
        r = chk[(chk.ano == ano) & (chk.turno == 2) & (chk.unidade == "BRASIL_COM_EXTERIOR")].iloc[0]
        assert (r.votos_pt, r.votos_bolsonaro) == (pt, bol), (ano, r.votos_pt, r.votos_bolsonaro)

    largo = _largo(mun)
    assert not largo["id_municipio_tse"].duplicated().any()
    salvar_interim(largo, "tse_presidente")

    print(chk[(chk.turno == 2) & chk.unidade.str.startswith("BRASIL")].to_string())
    print(largo.shape)
    for ano in ANOS:
        print(ano, "municípios no 2º turno:", largo[f"validos_2t_{ano}"].notna().sum())


if __name__ == "__main__":
    main()
