"""Consolida as fontes interim numa tabela de modelagem (uma linha por município).

Regras de negócio: ver docs/dados/consolidacao.md e docs/dados/dicionario.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.coleta.comum import INTERIM, RAIZ

ANOS = (2018, 2022, 2026)
TOTAL_PT_2026 = 64_056_078
TOTAL_BOLSONARO_2026 = 59_654_544
ID_TSE_FORA_CENSO = 73709
COLUNAS_PCT_CENSO = [
    "pct_pretos_pardos_2022",
    "pct_indigenas_2022",
    "taxa_alfabetizacao_15mais_2022",
    "pct_evangelicos_2022",
    "pct_catolicos_2022",
    "pct_sem_religiao_2022",
    "pct_urbana_2022",
    "pct_vab_agro_2021",
]


def _logit(p: pd.Series) -> pd.Series:
    return pd.Series(np.log(p / (1 - p)), index=p.index)


def _juntar(base: pd.DataFrame, nome: str, chave: str) -> pd.DataFrame:
    fonte = pd.read_parquet(INTERIM / f"{nome}.parquet")
    fonte[chave] = fonte[chave].astype("int64")
    n0 = len(base)
    # Colunas de identificação repetidas na fonte: a base manda; confere antes de descartar.
    comuns = [c for c in fonte.columns if c in base.columns and c != chave]
    if "sigla_uf" in comuns:
        uf = base[[chave, "sigla_uf"]].merge(fonte[[chave, "sigla_uf"]], on=chave)
        assert (uf["sigla_uf_x"] == uf["sigla_uf_y"]).all(), f"UF diverge em {nome}"
    fonte = fonte.drop(columns=comuns)
    out = base.merge(fonte, on=chave, how="left", validate="one_to_one", indicator="_m")
    assert len(out) == n0 and (out["_m"] == "both").all(), f"chave sem par em {nome}"
    return out.drop(columns="_m")


def _derivar(df: pd.DataFrame) -> pd.DataFrame:
    novas: dict[str, pd.Series] = {}
    for a in ANOS:
        n = df[f"votos_pt_2t_{a}"] + df[f"votos_bolsonaro_2t_{a}"]
        p = df[f"votos_pt_2t_{a}"] / n
        novas[f"n_2t_{a}"] = n
        novas[f"prop_pt_2t_{a}"] = p
        novas[f"logit_pt_2t_{a}"] = _logit(p)
        novas[f"prop_pt_1t_{a}"] = df[f"votos_pt_1t_{a}"] / df[f"validos_1t_{a}"]
        novas[f"prop_bolsonaro_1t_{a}"] = df[f"votos_bolsonaro_1t_{a}"] / df[f"validos_1t_{a}"]
        for t in (1, 2):
            novas[f"taxa_comparecimento_{t}t_{a}"] = (
                df[f"comparecimento_{t}t_{a}"] / df[f"aptos_{t}t_{a}"]
            )
    df = pd.concat([df, pd.DataFrame(novas)], axis=1)
    df["delta_comparecimento_2t_2022"] = (
        df["taxa_comparecimento_2t_2022"] - df["taxa_comparecimento_2t_2018"]
    )
    df["delta_comparecimento_2t_2026"] = (
        df["taxa_comparecimento_2t_2026"] - df["taxa_comparecimento_2t_2022"]
    )
    df["swing_logit_2022"] = df["logit_pt_2t_2022"] - df["logit_pt_2t_2018"]
    df["swing_logit_2026"] = df["logit_pt_2t_2026"] - df["logit_pt_2t_2022"]
    df["log_populacao_2022"] = np.log(df["populacao_2022"])
    df["log_eleitorado_2022"] = np.log(df["eleitorado_2022"])
    df["log_eleitorado_2026"] = np.log(df["eleitorado_2026"])
    df["densidade_2022"] = df["populacao_2022"] / df["area_km2"]
    df["log_densidade_2022"] = np.log(df["densidade_2022"])
    for a in (2022, 2026):
        df[f"cobertura_pbf_{a}"] = df[f"familias_pbf_{a}"] / df["domicilios_2022"]
    return df


def _resultado(df: pd.DataFrame, por: str) -> pd.DataFrame:
    partes = []
    for a in ANOS:
        g = df.groupby(por, observed=True).agg(
            votos_pt=(f"votos_pt_2t_{a}", "sum"),
            votos_bolsonaro=(f"votos_bolsonaro_2t_{a}", "sum"),
        )
        g["n"] = g["votos_pt"] + g["votos_bolsonaro"]
        g["prop_pt"] = g["votos_pt"] / g["n"]
        g.insert(0, "ano", a)
        partes.append(g.reset_index())
    return pd.concat(partes, ignore_index=True)


def main() -> None:
    base = pd.read_parquet(INTERIM / "ibge_municipios.parquet")
    base["id_municipio"] = base["id_municipio"].astype("int64")
    base["id_municipio_tse"] = base["id_municipio_tse"].astype("int64")
    assert len(base) == 5571 and base["fora_censo_2022"].sum() == 1

    # Votos 2026 do município fora do Censo (reportados, não modelados).
    pres = pd.read_parquet(INTERIM / "tse_presidente.parquet")
    fora = pres[pres["id_municipio_tse"] == ID_TSE_FORA_CENSO].iloc[0]
    pt_fora, bol_fora = int(fora["votos_pt_2t_2026"]), int(fora["votos_bolsonaro_2t_2026"])
    print(f"Fora do Censo (TSE {ID_TSE_FORA_CENSO}) 2026 2t: PT={pt_fora} Bolsonaro={bol_fora}")

    df = base[~base["fora_censo_2022"]].drop(columns="fora_censo_2022").reset_index(drop=True)
    df = _juntar(df, "tse_presidente", "id_municipio_tse")
    df = _juntar(df, "tse_ideologia", "id_municipio_tse")
    df = _juntar(df, "tse_perfil", "id_municipio_tse")
    df = _juntar(df, "ibge_censo_pib", "id_municipio")
    df = _juntar(df, "mds_bolsa_familia", "id_municipio")

    df[COLUNAS_PCT_CENSO] = df[COLUNAS_PCT_CENSO] / 100  # IBGE publica em 0-100

    df = _derivar(df)
    df["regiao"] = df["regiao"].astype("category")
    df["sigla_uf"] = df["sigla_uf"].astype("category")

    assert len(df) == 5570 and df["id_municipio"].is_unique
    assert df["id_municipio_tse"].is_unique
    assert df["votos_pt_2t_2026"].sum() == TOTAL_PT_2026 - pt_fora
    assert df["votos_bolsonaro_2t_2026"].sum() == TOTAL_BOLSONARO_2026 - bol_fora
    pcts = [c for c in df.columns if c.startswith(("pct_", "prop_", "taxa_"))]
    pcts = [c for c in pcts if not c.startswith("pct_dep_") or True]
    fora_faixa = {c: (df[c].min(), df[c].max()) for c in pcts if df[c].min() < 0 or df[c].max() > 1}
    print("Colunas de share fora de [0,1]:", fora_faixa)

    saida = RAIZ / "data" / "processed"
    saida.mkdir(parents=True, exist_ok=True)
    df.to_parquet(saida / "municipios.parquet", index=False)
    tabelas = RAIZ / "results" / "tables"
    tabelas.mkdir(parents=True, exist_ok=True)
    reg = _resultado(df, "regiao")
    reg.to_csv(tabelas / "resultado_regional.csv", index=False)
    _resultado(df, "sigla_uf").to_csv(tabelas / "resultado_uf.csv", index=False)

    print(f"municipios.parquet: {df.shape[0]} linhas x {df.shape[1]} colunas")
    nan = df.isna().sum()
    print("NaN > 0:\n", nan[nan > 0].to_string()[:3000])
    print(reg[reg["ano"] == 2026][["regiao", "votos_pt", "votos_bolsonaro", "prop_pt"]].to_string())
    for a in (2022, 2026):
        print(f"cobertura_pbf_{a} > 1:", int((df[f"cobertura_pbf_{a}"] > 1).sum()))
    print("inf:", int(np.isinf(df.select_dtypes("number")).sum().sum()))


if __name__ == "__main__":
    main()
