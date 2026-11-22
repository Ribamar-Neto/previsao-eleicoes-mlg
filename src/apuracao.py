"""Apuração parcial na ordem real do TSE (Parte 18, cenário A do doc 14).

Regras de negócio:
- O corte de x % é o primeiro instante de recebimento em que os votos válidos (Lula +
  adversário) das seções recebidas chegam a x % do total nacional doméstico. Seções com o
  mesmo instante entram juntas, então o % efetivo é um pouco maior que x.
- n restante (`n_u`) = aptos das seções ainda não recebidas x taxa de válidos do 1º turno do
  mesmo ano no município (válidos / aptos). É uma estimativa nos três métodos; o alvo é o
  resultado final real.
- Município sem linha no parquet do modelo (Boa Esperança do Norte, MT) entra com os votos
  já apurados em todos os métodos; o restante dele usa o share do método na UF e a taxa de
  válidos da UF.
- Contagem bruta: share apurado da região. Swing uniforme: swing nacional médio dos apurados
  contra o 2º turno anterior, somado ao share anterior de cada município não apurado.
- Regra de utilidade (fixada antes): o modelo vence um corte se o erro absoluto do Brasil E o
  MAE regional forem menores que os da contagem bruta E do swing; é útil com >= 4 de 6 cortes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src import espacial as S
from src import especificacoes as E
from src import modelos as M

CORTES: tuple[int, ...] = (5, 10, 25, 50, 67, 90)
MINIMO_VITORIAS = 4
METODOS: tuple[str, ...] = ("bruta", "swing", "modelo")
PREVIO: dict[int, int] = {2022: 2018, 2026: 2022}
# Definição da taxa de n_u: todos os válidos do 1º turno (adotada) ou só os dois finalistas (literal).
DEFINICAO_TAXA = "todos_validos"
DEFINICOES_TAXA: tuple[str, ...] = ("todos_validos", "dois_candidatos")


# --------------------------------------------------------------------------
# Cortes na ordem real
# --------------------------------------------------------------------------
def instante_do_corte(secoes: pd.DataFrame, pct: float) -> pd.Timestamp:
    """Primeiro instante em que os válidos recebidos chegam a `pct` % do total."""
    por_instante = secoes.groupby("dt_bu_recebido")["validos"].sum().sort_index().cumsum()
    frac = 100.0 * por_instante.to_numpy(float) / float(por_instante.iloc[-1])
    return cast(pd.Timestamp, por_instante.index[int(np.argmax(frac >= pct))])


def recebidas_ate(secoes: pd.DataFrame, pct: float) -> pd.Series:
    """Máscara das seções recebidas até o corte de `pct` %."""
    return secoes["dt_bu_recebido"] <= instante_do_corte(secoes, pct)


def pct_efetivo(secoes: pd.DataFrame, recebida: pd.Series) -> float:
    """% dos válidos nacionais efetivamente recebidos na máscara."""
    return 100.0 * float(secoes.loc[recebida, "validos"].sum()) / float(secoes["validos"].sum())


# --------------------------------------------------------------------------
# Base por município no corte
# --------------------------------------------------------------------------
def votos_taxa_1t(dfm: pd.DataFrame, ano: int, definicao: str = DEFINICAO_TAXA) -> pd.Series:
    """Numerador da taxa: válidos do 1º turno ou só os votos dos dois finalistas."""
    if definicao == "todos_validos":
        return dfm[f"validos_1t_{ano}"].astype(float)
    if definicao == "dois_candidatos":
        return (dfm[f"votos_pt_1t_{ano}"] + dfm[f"votos_bolsonaro_1t_{ano}"]).astype(float)
    raise ValueError(f"definição de taxa desconhecida: {definicao}")


def taxa_validos_1t(dfm: pd.DataFrame, ano: int, definicao: str = DEFINICAO_TAXA) -> pd.Series:
    """Votos da definição / aptos do 1º turno, por município do parquet do modelo."""
    return votos_taxa_1t(dfm, ano, definicao) / dfm[f"aptos_1t_{ano}"].astype(float)


def votos_restantes(aptos_pendentes: pd.Series, taxa: pd.Series) -> pd.Series:
    """n_u inteiro: aptos das seções não recebidas x taxa de válidos."""
    return (aptos_pendentes.astype(float) * taxa.astype(float)).round()


def montar_base(secoes: pd.DataFrame, dfm: pd.DataFrame, ano: int, recebida: pd.Series,
                definicao: str = DEFINICAO_TAXA) -> pd.DataFrame:
    """Uma linha por município (código TSE) das seções do `ano`, no corte `recebida`.

    Colunas: uf, regiao, y_c, n_c (apurados), aptos_pend, taxa, n_u, y_final, n_final, prev
    (share do 2º turno anterior; NaN sem linha no parquet) e `no_modelo`. `definicao` escolhe a taxa
    de n_u (`DEFINICOES_TAXA`); a contagem bruta não usa n_u.
    """
    s = secoes.assign(_rec=recebida.to_numpy())
    ap = s[~s["_rec"]].groupby("cd_municipio_tse")["aptos"].sum()
    rec = s[s["_rec"]].groupby("cd_municipio_tse")[["votos_lula", "validos"]].sum()
    fin = s.groupby("cd_municipio_tse")[["votos_lula", "validos"]].sum()
    base = pd.DataFrame({"uf": s.groupby("cd_municipio_tse")["uf"].first()})
    base["y_c"] = rec["votos_lula"].reindex(base.index).fillna(0).astype(float)
    base["n_c"] = rec["validos"].reindex(base.index).fillna(0).astype(float)
    base["aptos_pend"] = ap.reindex(base.index).fillna(0).astype(float)
    base["y_final"] = fin["votos_lula"].astype(float)
    base["n_final"] = fin["validos"].astype(float)
    m = dfm.assign(cd=dfm["id_municipio_tse"].astype("int64")).set_index("cd")
    taxa_m = taxa_validos_1t(dfm, ano, definicao).set_axis(m.index)
    base["regiao"] = m["regiao"].astype(str).reindex(base.index)
    base["prev"] = m[f"prop_pt_2t_{PREVIO[ano]}"].astype(float).reindex(base.index)
    base["no_modelo"] = base.index.isin(m.index)
    base["taxa"] = taxa_m.reindex(base.index)
    # fora do parquet: região e taxa da UF
    uf_reg = m.groupby("sigla_uf")["regiao"].first().astype(str)
    base["regiao"] = base["regiao"].fillna(base["uf"].map(uf_reg))
    num = votos_taxa_1t(dfm, ano, definicao).groupby(dfm["sigla_uf"]).sum()
    den = dfm[f"aptos_1t_{ano}"].astype(float).groupby(dfm["sigla_uf"]).sum()
    base["taxa"] = base["taxa"].fillna(base["uf"].map(num / den))
    base["n_u"] = votos_restantes(base["aptos_pend"], base["taxa"])
    return base


# --------------------------------------------------------------------------
# Métodos
# --------------------------------------------------------------------------
def swing_uniforme(y_c: np.ndarray, n_c: np.ndarray, prev: np.ndarray) -> float:
    """Swing nacional médio ponderado por n_c: (Σ y_c − Σ prev·n_c) / Σ n_c (prev finito)."""
    ok = np.isfinite(prev) & (n_c > 0)
    return float((y_c[ok].sum() - (prev[ok] * n_c[ok]).sum()) / n_c[ok].sum())


def completar_por_uf(pi: pd.Series, n_u: pd.Series, uf: pd.Series) -> pd.Series:
    """Troca π NaN pelo share médio (ponderado por n_u) dos demais municípios da UF."""
    ok = pi.notna()
    ponderado = (pi[ok] * n_u[ok]).groupby(uf[ok]).sum() / n_u[ok].groupby(uf[ok]).sum()
    return pi.fillna(uf.map(ponderado))


def agregar_share(y_hat: pd.Series, n_hat: pd.Series, grupos: pd.Series) -> pd.Series:
    """Share (fração) por grupo mais o total 'Brasil'."""
    y = y_hat.groupby(grupos).sum()
    n = n_hat.groupby(grupos).sum()
    out = y / n
    out["Brasil"] = float(y.sum() / n.sum())
    return out


def estimativa_bruta(base: pd.DataFrame) -> pd.Series:
    """Share apurado por região e Brasil."""
    return agregar_share(base["y_c"], base["n_c"], base["regiao"])


def estimativa_swing(base: pd.DataFrame) -> pd.Series:
    """Swing uniforme sobre o share anterior; UF para município sem anterior."""
    sw = swing_uniforme(base["y_c"].to_numpy(), base["n_c"].to_numpy(), base["prev"].to_numpy())
    pi = (base["prev"] + sw).clip(0.0, 1.0)
    pi = completar_por_uf(pi, base["n_u"], base["uf"])
    return agregar_share(base["y_c"] + pi * base["n_u"], base["n_c"] + base["n_u"], base["regiao"])


def estimativa_modelo(base: pd.DataFrame, pi: pd.Series) -> pd.Series:
    """Σ y_c + Σ π̂·n_u por região e Brasil; π̂ ausente vira o da UF."""
    pi = completar_por_uf(pi.reindex(base.index), base["n_u"], base["uf"])
    return agregar_share(base["y_c"] + pi * base["n_u"], base["n_c"] + base["n_u"], base["regiao"])


def alvo(base: pd.DataFrame) -> pd.Series:
    """Resultado final real (share de Lula) por região e Brasil."""
    return agregar_share(base["y_final"], base["n_final"], base["regiao"])


def erros(est: pd.Series, real: pd.Series) -> tuple[float, float]:
    """(erro absoluto do Brasil, MAE entre as regiões), em p.p."""
    d = (est - real).abs() * 100.0
    return float(d["Brasil"]), float(d.drop("Brasil").mean())


# --------------------------------------------------------------------------
# Modelo ajustado no apurado
# --------------------------------------------------------------------------
@dataclass
class ParcialModelo:
    """π̂ de todos os municípios do parquet e insumos da simulação de intervalos."""
    pi: np.ndarray
    beta: np.ndarray
    cov: np.ndarray
    X: np.ndarray
    theta: np.ndarray
    n_treino: int


def ajustar_parcial(cand: E.Candidato, dfm: pd.DataFrame, mapa: dict[str, str], y_c: np.ndarray,
                    n_c: np.ndarray, n_u: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    corte_km: float) -> ParcialModelo:
    """Ajusta `cand` nos (y_c, n_c) com n_c > 0 e prevê π̂ do restante em todos os municípios.

    A padronização vem só das linhas de treino. O n da linha de previsão é o n_u (dispersão
    por log n: o número de ensaios previstos), com piso 1.
    """
    d = dfm.copy()
    d[mapa["y"]] = y_c
    d[mapa["n"]] = n_c
    ok = n_c > 0
    aj = cand.ajustar(d[ok], mapa)
    dte = d.copy()
    dte[mapa["n"]] = np.maximum(n_u, 1.0)
    nb = int(aj.extra["n_beta"])
    X = E.montar_X(dte, mapa, aj.pad, cand.termos)
    cov = np.asarray(S.cov_conley(aj.modelo, lat[ok], lon[ok], corte_km), float)[:nb, :nb]
    return ParcialModelo(
        pi=np.asarray(cand.prever(aj, dte, mapa), float),
        beta=np.asarray(aj.modelo.params, float)[:nb],
        cov=cov, X=X.to_numpy(float),
        theta=E.prever_theta(aj, dte, mapa), n_treino=int(ok.sum()),
    )


def intervalos(pm: ParcialModelo, base_m: pd.DataFrame, base: pd.DataFrame, pi_uf: pd.Series,
               ligacao: str, sigma_uf: float, B: int, semente: int) -> pd.DataFrame:
    """IC 95 % do share por região e Brasil: votos já apurados + restante simulado.

    `base_m` = linhas do parquet (ordem de `pm`); município fora do modelo entra fixo no
    share de UF `pi_uf`. Choque de UF com `sigma_uf`, β com Conley, beta-binomial com θᵢ.
    """
    n_u = base_m["n_u"].to_numpy(float)
    sims = M.simular_votos_grupo(pm.beta, pm.cov, pm.X, n_u, base_m["uf"].to_numpy(),
                                 link=E.LIGACOES[ligacao], theta=pm.theta, sigma=sigma_uf,
                                 B=B, rng=semente)
    ufs = np.unique(base_m["uf"].to_numpy())
    reg_da_uf = base_m.groupby("uf")["regiao"].first().reindex(ufs).astype(str)
    regs = sorted(set(base["regiao"]))
    mat = np.array([[1.0 if reg_da_uf[u] == r else 0.0 for r in regs] for u in ufs])
    votos = sims @ mat
    fora = base[~base["no_modelo"]]
    fixo = (fora["y_c"] + fora["uf"].map(pi_uf).astype(float) * fora["n_u"]).groupby(fora["regiao"]).sum()
    apurado = base["y_c"].groupby(base["regiao"]).sum()
    den = (base["n_c"] + base["n_u"]).groupby(base["regiao"]).sum()
    votos = votos + (apurado.reindex(regs) + fixo.reindex(regs).fillna(0.0)).to_numpy()[None, :]
    share = np.column_stack([votos / den.reindex(regs).to_numpy()[None, :],
                             votos.sum(1) / float(den.sum())])
    q = np.quantile(share, [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame({"q025": q[0], "q50": q[1], "q975": q[2]}, index=regs + ["Brasil"])


def share_uf_modelo(base: pd.DataFrame, pi: pd.Series) -> pd.Series:
    """π̂ médio por UF (ponderado por n_u) sobre os municípios com modelo."""
    ok = base["no_modelo"]
    p = pi.reindex(base.index)[ok]
    w = base.loc[ok, "n_u"]
    return (p * w).groupby(base.loc[ok, "uf"]).sum() / w.groupby(base.loc[ok, "uf"]).sum()


# --------------------------------------------------------------------------
# Representatividade, regra e cobertura
# --------------------------------------------------------------------------
def representatividade(base: pd.DataFrame, pi: pd.Series | None, n_maiores: int = 20) -> pd.DataFrame:
    """Lacuna entre o share apurado (ou π̂) e o share real do restante, maiores x demais.

    Só municípios com n_c > 0 e restante real > 0; média ponderada pelo restante real, em p.p.
    `lacuna_apurado` = apurado − restante real (viés com sinal); `abs_*` em módulo.
    """
    top = base["n_final"].rank(ascending=False, method="first") <= n_maiores
    resto = base["n_final"] - base["n_c"]
    ok = (base["n_c"] > 0) & (resto > 0)
    sh_c = base["y_c"] / base["n_c"].where(base["n_c"] > 0)
    sh_r = (base["y_final"] - base["y_c"]) / resto.where(resto > 0)
    linhas = []
    for nome, sel in (("maiores", top), ("demais", ~top)):
        m = ok & sel
        w = resto[m]
        lin = {"grupo": nome, "n_municipios": int(m.sum()),
               "pct_restante_real": 100.0 * float(w.sum() / resto[ok].sum()) if ok.any() else np.nan,
               "lacuna_apurado_pp": 100.0 * float(((sh_c - sh_r)[m] * w).sum() / w.sum()) if m.any() else np.nan,
               "abs_apurado_pp": 100.0 * float(((sh_c - sh_r).abs()[m] * w).sum() / w.sum()) if m.any() else np.nan}
        if pi is not None:
            p = pi.reindex(base.index)
            lin["lacuna_modelo_pp"] = 100.0 * float(((p - sh_r)[m] * w).sum() / w.sum()) if m.any() else np.nan
            lin["abs_modelo_pp"] = 100.0 * float(((p - sh_r).abs()[m] * w).sum() / w.sum()) if m.any() else np.nan
        linhas.append(lin)
    return pd.DataFrame(linhas)


def aplicar_regra(tab: pd.DataFrame, minimo: int = MINIMO_VITORIAS) -> pd.DataFrame:
    """Vitórias do modelo por ano: Brasil E MAE regional menores que bruta E swing.

    `tab` tem ano, corte, metodo, erro_brasil_pp, mae_regional_pp. Corte sem linha de modelo
    (ajuste falhou) não vence.
    """
    linhas = []
    for ano, t in tab.groupby("ano"):
        vit, cortes = 0, sorted(t["corte"].unique())
        for c in cortes:
            p = t[t["corte"] == c].set_index("metodo")
            if "modelo" not in p.index:
                continue
            m = p.loc["modelo"]
            outros = p.drop(index="modelo")
            if (m["erro_brasil_pp"] < outros["erro_brasil_pp"]).all() and (
                    m["mae_regional_pp"] < outros["mae_regional_pp"]).all():
                vit += 1
        linhas.append({"ano": ano, "cortes": len(cortes), "vitorias_modelo": vit,
                       "minimo": minimo, "util": vit >= minimo})
    return pd.DataFrame(linhas)
