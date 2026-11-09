"""Backtest 2022 -> 2026: ajusta no 2º turno de 2022 só com o que se sabia antes e prevê 2026.

Regra de negócio central: o intercepto e os efeitos de região absorvem o swing nacional do
período de ajuste (2018 -> 2022, cerca de +6 p.p. para o PT). Prevendo 2026 com eles, o modelo
repete esse swing; `recalibrar` o remove, fixando a média nacional prevista no valor de 2022.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from src import modelos as M
from src.especificacoes import CANDIDATOS, CONTINUAS, MAPA_2026

MODELOS = ["quasi_logit", "quasi_reglag", "bb_logit", "bb_reglag", "glmm_uf"]
# Fora do ajuste ex-ante: comparecimento do próprio turno e sua variação não são conhecidos
# antes da eleição. O papel `delta_comp` só existe porque o Padronizador o exige.
VARS_EXANTE = tuple(v for v in CONTINUAS if v != "delta_comp")

# --- Ajuste no 2º turno de 2022 com informação anterior a 2 out 2022 ------------------
MAPA_FIT_2022 = {
    **MAPA_2026,
    "y": "votos_pt_2t_2022", "n": "n_2t_2022", "lag": "logit_pt_2t_2018",
    "nota_dep": "nota_media_dep_2018", "pct_dep_esq": "pct_dep_esq_2018",
    "bloco_pref": "bloco_prefeito_2020",
    "comparecimento": "taxa_comparecimento_2t_2018",
    "delta_comp": "delta_comparecimento_2t_2022",   # não usado no preditor
    "pct_mulheres": "pct_mulheres_2022", "pct_16_24": "pct_16_24_2022",
    "pct_60_mais": "pct_60_mais_2022", "pct_superior": "pct_superior_completo_2022",
    "pct_analfabeto": "pct_analfabeto_2022", "cobertura_pbf": "cobertura_pbf_2022",
    "pib_pc": "pib_per_capita_2021",
}
# --- Previsão de 2026: o mesmo mapa deslocado 4 anos ----------------------------------
MAPA_PREV_2026 = {
    **MAPA_FIT_2022,
    "y": "votos_pt_2t_2026", "n": "n_2t_2026", "lag": "logit_pt_2t_2022",
    "nota_dep": "nota_media_dep_2022", "pct_dep_esq": "pct_dep_esq_2022",
    "bloco_pref": "bloco_prefeito_2024",
    "comparecimento": "taxa_comparecimento_2t_2022",
    "pct_mulheres": "pct_mulheres_2026", "pct_16_24": "pct_16_24_2026",
    "pct_60_mais": "pct_60_mais_2026", "pct_superior": "pct_superior_completo_2026",
    "pct_analfabeto": "pct_analfabeto_2026", "cobertura_pbf": "cobertura_pbf_2026",
    "pib_pc": "pib_per_capita_2023",
}


def mapas_entre_turnos(ano_fit=2022, ano_prev=2026):
    """Variante (c): 1º turno do mesmo ciclo como preditor do 2º turno.

    Papéis reaproveitados: lag = logit PT/(PT+Bolsonaro) no 1º turno; delta_comp = parcela
    dos demais candidatos no 1º turno; comparecimento = comparecimento do 1º turno.
    """
    def um(ano, base):
        return {**base,
                "lag": f"logit_1t_{ano}", "delta_comp": f"outros_1t_{ano}",
                "nota_dep": f"nota_media_dep_{ano}", "pct_dep_esq": f"pct_dep_esq_{ano}",
                "comparecimento": f"taxa_comparecimento_1t_{ano}"}
    return um(ano_fit, MAPA_FIT_2022), um(ano_prev, MAPA_PREV_2026)


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as colunas derivadas do 1º turno (logit PT x Bolsonaro e parcela de outros)."""
    d = df.copy()
    for a in (2022, 2026):
        pt, bo, va = (d[f"votos_pt_1t_{a}"].astype(float), d[f"votos_bolsonaro_1t_{a}"].astype(float),
                      d[f"validos_1t_{a}"].astype(float))
        d[f"logit_1t_{a}"] = logit(np.clip(pt / (pt + bo), 1e-6, 1 - 1e-6))
        d[f"outros_1t_{a}"] = 1 - (pt + bo) / va
    return d


def linhas_modeladas(df, *mapas) -> pd.Series:
    """Municípios com todas as colunas dos mapas finitas (exclui Boa Esperança do Norte)."""
    cols = sorted({c for m in mapas for k, c in m.items() if k not in ("regiao", "uf", "bloco_pref")})
    ok = np.isfinite(df[cols].to_numpy(float)).all(axis=1)
    return pd.Series(ok, index=df.index)


def candidato_exante(nome, vars_=VARS_EXANTE):
    c = CANDIDATOS[nome]
    return replace(c, termos=replace(c.termos, vars=tuple(vars_)))


def ajustar_e_prever(nome, df, mapa_fit, mapa_prev, vars_=VARS_EXANTE):
    """Ajusta em `df` com `mapa_fit` e devolve (ajuste, pi_2026). Padronização do treino."""
    cand = candidato_exante(nome, vars_)
    aj = cand.ajustar(df, mapa_fit)
    aj.pad.mapa = mapa_prev        # médias e desvios ficam os do treino; só as colunas mudam
    return aj, np.asarray(cand.prever(aj, df, mapa_prev), float)


def coef_lag(aj) -> float:
    """Coeficiente do lag na escala bruta (log-odds por log-odds)."""
    p = aj.modelo.params
    v = p["lag"] if hasattr(p, "index") else np.asarray(p)[aj.nomes.index("lag")]
    return float(v / aj.pad.desvio["lag"])


def recalibrar(pi, n, alvo) -> tuple[np.ndarray, float]:
    """Desloca o logit de pi por delta, de modo que a média ponderada por n seja `alvo`."""
    pi, n = np.asarray(pi, float), np.asarray(n, float)
    z = logit(np.clip(pi, 1e-9, 1 - 1e-9))
    f = lambda d: float(np.sum(expit(z + d) * n) / n.sum() - alvo)  # noqa: E731
    d = brentq(f, -3, 3)
    return expit(z + d), float(d)


def n_previsto(df, mapa_prev=MAPA_PREV_2026, ano_ref=2022, ano_alvo=2026):
    """n̂ = eleitorado do ano-alvo x (n_2t / aptos_2t do ano de referência)."""
    return (df[f"eleitorado_{ano_alvo}"] * df[f"n_2t_{ano_ref}"] / df[f"aptos_2t_{ano_ref}"]).to_numpy(float)


def erros_grupo(pi, n_peso, y_real, n_real, grupos) -> pd.DataFrame:
    """Share previsto (peso n_peso) x real (y/n) por grupo e Brasil; erro em p.p."""
    prev = M.agregar(pi, n_peso, grupos)
    g = pd.DataFrame({"g": np.asarray(grupos), "y": np.asarray(y_real, float),
                      "n": np.asarray(n_real, float)}).groupby("g")[["y", "n"]].sum()
    real = g["y"] / g["n"]
    real.loc["Brasil"] = g["y"].sum() / g["n"].sum()
    out = pd.DataFrame({"previsto": prev, "real": real})
    out["erro_pp"] = (out["previsto"] - out["real"]) * 100
    return out


def resumo_erros(pi, n_peso, y_real, n_real, regiao, uf) -> dict:
    """Erro nacional, MAE por região e por UF (simples) e MAE municipal ponderado (p.p.)."""
    er, eu = erros_grupo(pi, n_peso, y_real, n_real, regiao), erros_grupo(pi, n_peso, y_real, n_real, uf)
    m = M.metricas(y_real, n_real, pi)
    return {"erro_nacional_pp": float(er.loc["Brasil", "erro_pp"]),
            "mae_regiao_pp": float(er.drop("Brasil")["erro_pp"].abs().mean()),
            "mae_uf_pp": float(eu.drop("Brasil")["erro_pp"].abs().mean()),
            "mae_mun_pond_pp": m["mae_pond_pp"]}
