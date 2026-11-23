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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from statsmodels.genmod.families.links import Link

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

    Colunas: uf, regiao, y_c, n_c (apurados), aptos_pend, aptos_rec, taxa, n_u, y_final, n_final, prev
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
    base["aptos_rec"] = s[s["_rec"]].groupby("cd_municipio_tse")["aptos"].sum().reindex(base.index).fillna(0).astype(float)
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


MIN_MUN_UF = 5   # abaixo disso a UF sorteia o resíduo municipal do pool nacional


def _residuos_nu(base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """(válidos, e_i, sd_UF) da razão r_i = n_c / (aptos_rec x taxa), só com seções recebidas.

    Válido: n_c > 0 e aptos_rec x taxa > 0. R_UF = log(Σ n_c / Σ aptos_rec x taxa) por UF e R_BR no
    país; sd_UF = desvio (ddof 1) de R_UF - R_BR entre UFs com válidos; e_i = log r_i - R_UF.
    """
    esperado = base["aptos_rec"] * base["taxa"]
    ok = ((base["n_c"] > 0) & (esperado > 0)).to_numpy()
    uf = base["uf"].to_numpy()
    nc, es = base["n_c"].to_numpy(float), esperado.to_numpy(float)
    r_uf = {u: np.log(nc[ok & (uf == u)].sum() / es[ok & (uf == u)].sum()) for u in np.unique(uf[ok])}
    r_br = np.log(nc[ok].sum() / es[ok].sum())
    sd_uf = float(np.std([v - r_br for v in r_uf.values()], ddof=1)) if len(r_uf) > 1 else 0.0
    e = np.full(len(base), np.nan)
    e[ok] = np.log(nc[ok] / es[ok]) - np.array([r_uf[u] for u in uf[ok]])
    return ok, e, sd_uf


def sd_uf_nu(base: pd.DataFrame) -> float:
    """Desvio entre UFs da razão n_c / (aptos_rec x taxa), em log, contra o Brasil."""
    return _residuos_nu(base)[2]


def fatores_nu(base: pd.DataFrame, B: int, semente: int) -> np.ndarray:
    """Fatores k (B x municípios de `base`) da incerteza de n_u, só com seções já recebidas.

    Por simulação, choque de UF ~ N(0, sd_UF²) mais resíduo municipal e_i sorteado com
    reposição da UF (ou do pool nacional com menos de `MIN_MUN_UF` válidos), centrado pela
    mediana do pool: k tem centro 1 em log. Ver `_residuos_nu`.
    """
    ok, e, sd_uf = _residuos_nu(base)
    uf = base["uf"].to_numpy()
    pool_br = e[ok]
    rng = np.random.default_rng(semente)
    ufs = np.unique(uf)
    u = rng.normal(0.0, sd_uf, size=(B, len(ufs)))
    k = np.empty((B, len(base)))
    for j, g in enumerate(ufs):
        idx = np.flatnonzero(uf == g)
        pool = e[ok & (uf == g)]
        if len(pool) < MIN_MUN_UF:
            pool = pool_br
        k[:, idx] = np.exp(u[:, [j]] + rng.choice(pool, size=(B, len(idx))) - np.median(pool))
    return k


def intervalos(pm: ParcialModelo, base_m: pd.DataFrame, base: pd.DataFrame, pi_uf: pd.Series,
               ligacao: str, sigma_uf: float, B: int, semente: int,
               fatores: np.ndarray | None = None) -> pd.DataFrame:
    """IC 95 % do share por região e Brasil: votos já apurados + restante simulado.

    `base_m` = linhas do parquet (ordem de `pm`); município fora do modelo entra fixo no
    share de UF `pi_uf`. Choque de UF com `sigma_uf`, β com Conley, beta-binomial com θᵢ.
    `fatores` (de `fatores_nu`, B x linhas de `base`) simula n_u por município, nos votos e no
    denominador; sem ele, n_u é fixo.
    """
    n_u = base_m["n_u"].to_numpy(float)
    nu_b = None
    if fatores is not None:
        nu_b = np.rint(base["n_u"].to_numpy(float)[None, :] * fatores)
        pos = base.index.get_indexer(base_m.index)
        n_u = np.where(pos[None, :] >= 0, nu_b[:, np.maximum(pos, 0)], n_u[None, :])
    sims = M.simular_votos_grupo(pm.beta, pm.cov, pm.X, n_u, base_m["uf"].to_numpy(),
                                 link=E.LIGACOES[ligacao], theta=pm.theta, sigma=sigma_uf,
                                 B=B, rng=semente)
    ufs = np.unique(base_m["uf"].to_numpy())
    reg_da_uf = base_m.groupby("uf")["regiao"].first().reindex(ufs).astype(str)
    regs = sorted(set(base["regiao"]))
    mat = np.array([[1.0 if reg_da_uf[u] == r else 0.0 for r in regs] for u in ufs])
    votos = sims @ mat
    fora = base[~base["no_modelo"]]
    # y_c de `fora` já está em `apurado`; aqui entra só o restante projetado com π da UF
    fixo = (fora["uf"].map(pi_uf).astype(float) * fora["n_u"]).groupby(fora["regiao"]).sum()
    apurado = base["y_c"].groupby(base["regiao"]).sum()
    den = (base["n_c"] + base["n_u"]).groupby(base["regiao"]).sum()
    fixo_r = fixo.reindex(regs).fillna(0.0).to_numpy()[None, :]
    den_r = den.reindex(regs).to_numpy()[None, :]
    if nu_b is not None:
        reg_mat = (base["regiao"].to_numpy()[:, None] == np.array(regs)[None, :]).astype(float)
        peso_fora = (~base["no_modelo"].to_numpy()) * base["uf"].map(pi_uf).astype(float).fillna(0.0).to_numpy()
        fixo_r = fixo_r + ((nu_b - base["n_u"].to_numpy(float)[None, :]) * peso_fora[None, :]) @ reg_mat
        den_r = den_r + (nu_b - base["n_u"].to_numpy(float)[None, :]) @ reg_mat
    votos = votos + (apurado.reindex(regs).to_numpy()[None, :] + fixo_r)
    share = np.column_stack([votos / den_r, votos.sum(1) / den_r.sum(1)])
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


# --------------------------------------------------------------------------
# Estimativas do modelo final por UF e região (notebook 05, Parte 16)
# --------------------------------------------------------------------------

def simular_f(partes: Mapping[str, Mapping[str, Any]], chave: str, sigma: float, *, nhat: np.ndarray,
              link: type[Link], B: int = 1000, seed: int = 7) -> np.ndarray:
    """Votos simulados da UF retida, uma coluna por UF (ordem de `partes`), com o n̂ do LOSO.

    `chave` escolhe a covariância de β (`cov_conley` ou `cov_modelo`); `sigma` é o choque de UF.
    O gerador é um só para todas as UFs: a mesma semente reproduz os mesmos sorteios.
    """
    rng = np.random.default_rng(seed)
    return np.column_stack([
        M.simular_votos_grupo(v["beta"], v[chave], v["X"], nhat[v["te"]], np.zeros(len(v["te"])), link=link,
                              theta=v["theta"], sigma=sigma, B=B, rng=rng)[:, 0]
        for v in partes.values()])


def resumo_f(votos: np.ndarray, *, ufs: list[str], regs: list[str], n_uf: np.ndarray, uf_reg: np.ndarray,
             obs_uf: pd.Series, obs_reg: pd.Series, obs_brasil: float, prev_uf: pd.Series) -> pd.DataFrame:
    """Ponto (previsão LOSO agregada com n̂) e quantis dos votos simulados, por UF, região e Brasil."""
    linhas: dict[tuple[str, str], tuple[np.ndarray, float, float]] = {}
    for i, u in enumerate(ufs):
        linhas[("uf", u)] = (votos[:, i], n_uf[i], obs_uf[u])
    for r in regs:
        m = uf_reg == r
        linhas[("regiao", r)] = (votos[:, m].sum(1), n_uf[m].sum(), obs_reg[r])
    linhas[("regiao", "Brasil")] = (votos.sum(1), n_uf.sum(), obs_brasil)
    prev_n = prev_uf[ufs].to_numpy() * n_uf
    out = []
    for (nivel, nome), (v, nn, real) in linhas.items():
        q = np.quantile(v / nn, [0.025, 0.05, 0.5, 0.95, 0.975])
        if nivel == "regiao" and nome != "Brasil":
            pt = prev_n[uf_reg == nome].sum() / nn
        else:
            pt = prev_uf[nome] if nivel == "uf" else prev_n.sum() / nn
        out.append({"nivel": nivel, "nome": nome, "lula_pct": pt * 100, "q025": q[0] * 100, "q05": q[1] * 100, "q50": q[2] * 100,
                    "q95": q[3] * 100, "q975": q[4] * 100, "real_pct": real * 100})
    d = pd.DataFrame(out)
    d["erro_pp"] = d.lula_pct - d.real_pct
    d["dentro_ic95"] = (d.real_pct >= d.q025) & (d.real_pct <= d.q975)
    d["dentro_ic90"] = (d.real_pct >= d.q05) & (d.real_pct <= d.q95)
    d["largura_pp"] = d.q975 - d.q025
    d["largura90_pp"] = d.q95 - d.q05
    return d
