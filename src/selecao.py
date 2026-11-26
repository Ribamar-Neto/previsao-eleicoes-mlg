"""Seleção de variáveis: passo a passo (AIC/BIC), Hosmer-Lemeshow e todos os subconjuntos.

Regras:
- `lag`, dummies de região, `bloco_pref` e theta nunca são candidatos: só as
  `candidatas` entram e saem. Os retornos listam apenas candidatas (mais as
  interações do estágio 2), sem as `fixas`.
- Todo ajuste passa por `Candidato.ajustar`; a triagem de todos os subconjuntos usa
  um IRLS logit próprio sobre a matriz de desenho cheia (padronização ajustada uma vez).
- Quase-binomial: phi de Pearson do modelo cheio, fixo para todos os subconjuntos.
- Interações (estágio 2 de Hosmer-Lemeshow) entram como colunas-produto das
  variáveis padronizadas, registradas no mapa com prefixo `w_` (o Padronizador as
  padroniza sem mudar `especificacoes.py`); o nome do papel é `w_<a>__x__<b>`.
- Hosmer-Lemeshow segue Paula (Modelos de regressão, pp. 358-371). Passo 4 (p. 360):
  antes de abrir as interações, o modelo do estágio 1 é ajustado e cada coeficiente
  testado por Wald (erro padrão do modelo, mesma verossimilhança do LRT); remove-se uma
  variável por vez, a de maior p acima de alfa_wald (0,05), reajustando, até todas
  ficarem significativas. O estágio 2 parte dos sobreviventes. `alfa_wald=None` desliga
  o passo e reproduz a seleção sem Wald.
- A tabela `niveis` guarda o nível descritivo pi de toda candidata em cada iteração
  (entrada para quem está fora, saída para quem está dentro), como nas Tabelas 4.8 e 4.9
  do livro; `escolhida` marca o termo que se moveu.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy.special import expit, gammaln
from threadpoolctl import threadpool_limits

from src import especificacoes as E
from src import modelos as M
from src.especificacoes import (
    _COL_OFFSET,
    BLOCOS,
    REGIAO_REF,
    REGIOES,
    CONTINUAS,
    LIGACOES,
    Candidato,
    Padronizador,
    Termos,
    montar_X,
)

Metodo = Literal["backward", "forward", "both"]
Criterio = Literal["bic", "aic"]

CANDIDATAS_PADRAO: tuple[str, ...] = tuple(c for c in CONTINUAS if c != "lag")
TERMOS_PADRAO: Termos = Termos()
_CTX: dict[str, Any] = {}  # contexto herdado pelos filhos do fork


# --------------------------------------------------------------------------
# Infra: ajuste por conjunto de variáveis e mapa paralelo
# --------------------------------------------------------------------------
def _mapear(func: Callable[[Any], Any], itens: Sequence[Any], n_jobs: int) -> list[Any]:
    itens = list(itens)
    if n_jobs <= 1 or len(itens) <= 1:
        return [func(i) for i in itens]
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(itens)), mp_context=get_context("fork")) as ex:
        return list(ex.map(func, itens))


def _ajustar_vars(variaveis: tuple[str, ...]) -> tuple[float, int]:
    """(loglik, k) do modelo com `variaveis` candidatas (contexto em `_CTX`)."""
    c = _CTX
    termos = replace(c["termos"], vars=c["base"] + tuple(variaveis))
    aj = Candidato("sel", c["familia"], "logit", termos).ajustar(c["df"], c["mapa"])
    return float(aj.loglik), int(aj.k)


def _base(termos: Termos, candidatas: Sequence[str], fixas: Sequence[str]) -> tuple[str, ...]:
    """Papéis do modelo que não são candidatos (fixas e extras como w_*)."""
    resto = tuple(v for v in termos.vars if v not in candidatas and v not in fixas)
    return tuple(fixas) + resto


def _valor(ll: float, k: int, n: int, criterio: Criterio) -> float:
    return -2.0 * ll + (k * np.log(n) if criterio == "bic" else 2.0 * k)


def _preparar(
    df: pd.DataFrame,
    mapa: dict[str, str],
    termos: Termos,
    candidatas: Sequence[str],
    fixas: Sequence[str],
    familia: str,
) -> None:
    _CTX.clear()
    _CTX.update(df=df, mapa=mapa, termos=termos, familia=familia, base=_base(termos, candidatas, fixas))


def _ordenar(vs: Sequence[str], candidatas: Sequence[str]) -> tuple[str, ...]:
    pos = {v: i for i, v in enumerate(candidatas)}
    return tuple(sorted(vs, key=lambda v: pos.get(v, len(pos))))


# --------------------------------------------------------------------------
# 1. Passo a passo por AIC/BIC
# --------------------------------------------------------------------------
def passo_a_passo(
    df: pd.DataFrame,
    mapa: dict[str, str],
    termos: Termos,
    metodo: Metodo,
    criterio: Criterio = "bic",
    candidatas: Sequence[str] = CANDIDATAS_PADRAO,
    fixas: Sequence[str] = ("lag",),
    familia: str = "betabin",
    n_jobs: int = 14,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Busca gulosa completa: aceita o melhor movimento só se reduz o critério.

    backward parte de todas as candidatas e avalia cada remoção; forward parte só das
    fixas e avalia cada entrada; both parte das fixas e avalia entradas e remoções
    em cada passo. Devolve (candidatas escolhidas, trilha por passo).
    """
    candidatas = tuple(candidatas)
    n = len(df)
    _preparar(df, mapa, termos, candidatas, fixas, familia)
    atual = candidatas if metodo == "backward" else ()
    ll, k = _ajustar_vars(atual)
    val = _valor(ll, k, n, criterio)
    linhas = [{"passo": 0, "acao": "inicio", "variavel": "", "valor_criterio": val, "n_vars": len(atual)}]
    passo = 0
    while True:
        movimentos: list[tuple[str, str, tuple[str, ...]]] = []
        if metodo in ("forward", "both"):
            movimentos += [("entra", v, _ordenar(atual + (v,), candidatas)) for v in candidatas if v not in atual]
        if metodo in ("backward", "both"):
            movimentos += [("sai", v, tuple(x for x in atual if x != v)) for v in atual]
        if not movimentos:
            break
        res = _mapear(_ajustar_vars, [m[2] for m in movimentos], n_jobs)
        vals = [_valor(r[0], r[1], n, criterio) for r in res]
        i = int(np.argmin(vals))
        if not vals[i] < val:
            break
        passo += 1
        acao, var, atual = movimentos[i]
        val = vals[i]
        linhas.append({"passo": passo, "acao": acao, "variavel": var, "valor_criterio": val, "n_vars": len(atual)})
    return atual, pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# 2. Stepwise de Hosmer-Lemeshow (Paula, sec. 4.6.9)
# --------------------------------------------------------------------------
def _interacoes(
    df: pd.DataFrame, mapa: dict[str, str], termos: Termos, principais: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Copia de df com colunas-produto z_a*z_b e novos papéis w_<a>__x__<b> no mapa."""
    pad = Padronizador(mapa).fit(df)
    novo = df.copy()
    mapa2 = dict(mapa)
    nomes: dict[str, str] = {}
    for a, b in combinations(principais, 2):
        papel = f"w_{a}__x__{b}"
        col = f"_int_{a}__x__{b}"
        novo[col] = pad.z(df, a) * pad.z(df, b)
        mapa2[papel] = col
        nomes[papel] = f"{a}:{b}"
    return novo, mapa2, nomes


def _lrt_etapa(
    candidatas: tuple[str, ...],
    dentro: tuple[str, ...],
    ll_atual: float,
    k_atual: int,
    n_jobs: int,
    p_entrada: float,
    p_saida: float,
    n: int,
    etapa: int,
    linhas: list[dict[str, Any]],
    niveis: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Itera entrada/saída de HL sobre `candidatas`; devolve o conjunto final."""
    it = 0
    while it < 4 * max(len(candidatas), 1) + 4:
        it += 1
        mudou = False
        fora = [v for v in candidatas if v not in dentro]
        if fora:
            res = _mapear(_ajustar_vars, [_ordenar(dentro + (v,), candidatas) for v in fora], n_jobs)
            delt = [max(2.0 * (r[0] - ll_atual), 0.0) for r in res]
            gl = [r[1] - k_atual for r in res]
            pis = [float(stats.chi2.sf(d, g)) for d, g in zip(delt, gl, strict=True)]
            i = int(np.argmin(pis))
            ok = pis[i] < p_entrada
            niveis += [
                {
                    "etapa": etapa,
                    "iteracao": it,
                    "fase": "entrada",
                    "variavel": v,
                    "delta_deviance": d,
                    "gl": g,
                    "pi": p,
                    "escolhida": ok and j == i,
                }
                for j, (v, d, g, p) in enumerate(zip(fora, delt, gl, pis, strict=True))
            ]
            if ok:
                dentro = _ordenar(dentro + (fora[i],), candidatas)
                ll_atual, k_atual = res[i]
                linhas.append(
                    {
                        "etapa": etapa,
                        "iteracao": it,
                        "acao": "entra",
                        "variavel": fora[i],
                        "delta_deviance": delt[i],
                        "gl": gl[i],
                        "pi": pis[i],
                    }
                )
                mudou = True
        if dentro:
            res = _mapear(_ajustar_vars, [tuple(x for x in dentro if x != v) for v in dentro], n_jobs)
            delt = [max(2.0 * (ll_atual - r[0]), 0.0) for r in res]
            gl = [k_atual - r[1] for r in res]
            pis = [float(stats.chi2.sf(d, g)) for d, g in zip(delt, gl, strict=True)]
            i = int(np.argmax(pis))
            ok = pis[i] > p_saida
            niveis += [
                {
                    "etapa": etapa,
                    "iteracao": it,
                    "fase": "saida",
                    "variavel": v,
                    "delta_deviance": d,
                    "gl": g,
                    "pi": p,
                    "escolhida": ok and j == i,
                }
                for j, (v, d, g, p) in enumerate(zip(dentro, delt, gl, pis, strict=True))
            ]
            if ok:
                saiu = dentro[i]
                dentro = tuple(x for x in dentro if x != saiu)
                ll_atual, k_atual = res[i]
                linhas.append(
                    {
                        "etapa": etapa,
                        "iteracao": it,
                        "acao": "sai",
                        "variavel": saiu,
                        "delta_deviance": delt[i],
                        "gl": gl[i],
                        "pi": pis[i],
                    }
                )
                mudou = True
        if not mudou:
            break
    return dentro


_COLS_NIVEIS = ["etapa", "iteracao", "fase", "variavel", "delta_deviance", "gl", "pi", "escolhida"]


@dataclass(frozen=True)
class ResultadoHL:
    variaveis: tuple[str, ...]
    passos: pd.DataFrame  # trilha dos movimentos (entra/sai), com `rotulo`
    niveis: pd.DataFrame  # pi de toda candidata em cada iteração (Paula, Tabelas 4.8/4.9)
    wald: pd.DataFrame  # passo 4: rodada, variavel, coef, ep, z, p, acao


class WaldSemErroPadrao(ValueError):
    """Hessiana singular: termos colineares não têm erro-padrão, e o passo 4 não decide."""

    def __init__(self, termos: list[str]) -> None:
        super().__init__(f"Wald sem erro-padrão para {termos}: termos colineares")


def _passo_wald(dentro: tuple[str, ...], alfa: float) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Passo 4 de Paula: Wald de cada candidata em `dentro`; remove a de maior p > alfa, repete."""
    c = _CTX
    linhas: list[dict[str, Any]] = []
    rodada = 0
    while dentro:
        rodada += 1
        termos = replace(c["termos"], vars=c["base"] + dentro)
        aj = Candidato("sel", c["familia"], "logit", termos).ajustar(c["df"], c["mapa"])
        params = pd.Series(np.asarray(aj.modelo.params, float), index=aj.nomes)
        ep = pd.Series(np.asarray(aj.modelo.bse, float), index=aj.nomes)
        z = params / ep
        p = pd.Series(2 * stats.norm.sf(np.abs(z)), index=aj.nomes)
        # Hessiana singular deixa p NaN; parar em silêncio manteria um termo não identificável
        sem_p = [v for v in dentro if not np.isfinite(p[v])]
        if sem_p:
            raise WaldSemErroPadrao(sem_p)
        pior = max(dentro, key=lambda v: p[v])
        sai = p[pior] > alfa
        linhas += [
            {
                "rodada": rodada,
                "variavel": v,
                "coef": float(params[v]),
                "ep": float(ep[v]),
                "z": float(z[v]),
                "p": float(p[v]),
                "acao": "removida" if sai and v == pior else "mantida",
            }
            for v in dentro
        ]
        if not sai:
            break
        dentro = tuple(v for v in dentro if v != pior)
    tabela = pd.DataFrame(linhas, columns=["rodada", "variavel", "coef", "ep", "z", "p", "acao"])
    return dentro, tabela


def niveis_largos(niveis: pd.DataFrame, etapa: int) -> pd.DataFrame:
    """Layout de Paula (Tabelas 4.8/4.9): linhas = iteração, colunas = termo, valores = pi.

    Entrada e saída da mesma iteração ficam em linhas separadas (`fase` no índice);
    a coluna de cada termo traz o pi da fase em que ele é avaliado.
    """
    nv = niveis[niveis["etapa"] == etapa]
    col = "rotulo" if "rotulo" in nv.columns else "variavel"
    larga = nv.pivot_table(index=["iteracao", "fase"], columns=col, values="pi", aggfunc="first")
    larga = larga[list(dict.fromkeys(nv[col]))]
    larga = larga.sort_index(
        level=["iteracao", "fase"], key=lambda ix: ix.map(lambda x: {"entrada": 0, "saida": 1}.get(x, x))
    )
    return larga.reset_index(level="fase").rename_axis("iteracao")


def stepwise_hosmer_lemeshow(
    df: pd.DataFrame,
    mapa: dict[str, str],
    termos: Termos,
    *,
    candidatas: Sequence[str] = CANDIDATAS_PADRAO,
    fixas: Sequence[str] = ("lag",),
    p_entrada: float = 0.20,
    p_saida: float = 0.25,
    familia: str = "betabin",
    interacoes: bool = False,
    n_jobs: int = 14,
    alfa_wald: float | None = 0.05,
) -> ResultadoHL:
    """Stepwise de Hosmer-Lemeshow com teste da razão de verossimilhanças.

    Delta = 2(l_com - l_sem), gl = parâmetros extras, pi = P(chi2_gl > Delta). Entra
    a de menor pi se pi < p_entrada; depois sai a de maior pi se pi > p_saida.
    Passo 4 (`alfa_wald`): Wald nos sobreviventes do estágio 1, removendo uma por vez.
    Estágio 2 (`interacoes=True`): interações de pares entre os sobreviventes (e as
    fixas), mesma regra, com os efeitos principais mantidos (hierarquia). Nomes de
    interação saem como papéis `w_<a>__x__<b>`; a tabela traz o nome `a:b` em `rotulo`.
    """
    candidatas = tuple(candidatas)
    n = len(df)
    _preparar(df, mapa, termos, candidatas, fixas, familia)
    ll0, k0 = _ajustar_vars(())
    linhas: list[dict[str, Any]] = []
    niveis: list[dict[str, Any]] = []
    dentro = _lrt_etapa(candidatas, (), ll0, k0, n_jobs, p_entrada, p_saida, n, 1, linhas, niveis)
    wald = pd.DataFrame(columns=["rodada", "variavel", "coef", "ep", "z", "p", "acao"])
    if alfa_wald is not None and dentro:
        dentro, wald = _passo_wald(dentro, alfa_wald)
        dentro = _ordenar(dentro, candidatas)
    rotulos: dict[str, str] = {}
    if interacoes:
        principais = tuple(fixas) + dentro
        df2, mapa2, rotulos = _interacoes(df, mapa, termos, principais)
        cand2 = tuple(rotulos)
        _preparar(df2, mapa2, termos, cand2, fixas, familia)
        _CTX["base"] = _base(termos, candidatas, fixas) + dentro
        ll1, k1 = _ajustar_vars(())
        dentro2 = _lrt_etapa(cand2, (), ll1, k1, n_jobs, p_entrada, p_saida, n, 2, linhas, niveis)
        dentro = dentro + dentro2
    tabela = pd.DataFrame(linhas, columns=["etapa", "iteracao", "acao", "variavel", "delta_deviance", "gl", "pi"])
    tabela["rotulo"] = tabela["variavel"].map(lambda v: rotulos.get(v, v))
    tab_niveis = pd.DataFrame(niveis, columns=_COLS_NIVEIS)
    tab_niveis["rotulo"] = tab_niveis["variavel"].map(lambda v: rotulos.get(v, v))
    return ResultadoHL(dentro, tabela, tab_niveis, wald)


# --------------------------------------------------------------------------
# 3. Todos os subconjuntos (triagem quase-binomial + refit beta-binomial)
# --------------------------------------------------------------------------
def _irls_logit(
    X: np.ndarray, y: np.ndarray, n: np.ndarray, off: np.ndarray, b0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """IRLS binomial logit; devolve (beta, pi). Sem offset: off = 0."""
    b = np.zeros(X.shape[1]) if b0 is None else b0.copy()
    if b0 is None:
        p0 = np.clip(y.sum() / n.sum(), 1e-6, 1 - 1e-6)
        b[0] = np.log(p0 / (1 - p0))
    for _ in range(30):
        eta = X @ b + off
        pi = np.clip(expit(eta), 1e-10, 1 - 1e-10)
        w = n * pi * (1 - pi)
        z = eta - off + (y / n - pi) / (pi * (1 - pi))
        Xw = X * w[:, None]
        novo = np.linalg.solve(X.T @ Xw, Xw.T @ z)
        passo = float(np.max(np.abs(novo - b)))
        b = novo
        if passo < 1e-8:
            break
    pi = np.clip(expit(X @ b + off), 1e-10, 1 - 1e-10)
    return b, pi


def _triar_faixa(faixa: tuple[int, int]) -> np.ndarray:
    with threadpool_limits(limits=1):  # BLAS multithread em 14 processos é 4x mais lento
        return _triar_faixa_1t(faixa)


def _triar_faixa_1t(faixa: tuple[int, int]) -> np.ndarray:
    c = _CTX
    X, y, n, off = c["X"], c["y"], c["n"], c["off"]
    base, cols_cand, const, phi, ln_n = c["base_idx"], c["cand_idx"], c["const"], c["phi"], c["ln_n"]
    out = np.empty(faixa[1] - faixa[0])
    for j, mask in enumerate(range(*faixa)):
        cols = base + [cols_cand[i] for i in range(len(cols_cand)) if mask >> i & 1]
        Xs = X[:, cols]
        try:
            _, pi = _irls_logit(Xs, y, n, off)
            ll = float(np.sum(const + y * np.log(pi) + (n - y) * np.log1p(-pi)))
            out[j] = -2.0 * ll / phi + len(cols) * ln_n
        except np.linalg.LinAlgError:
            out[j] = np.inf
    return out


def todos_os_subconjuntos(
    df: pd.DataFrame,
    mapa: dict[str, str],
    termos: Termos,
    candidatas: Sequence[str] = CANDIDATAS_PADRAO,
    fixas: Sequence[str] = ("lag",),
    top: int = 200,
    n_jobs: int = 14,
) -> pd.DataFrame:
    """Triagem de 2^m subconjuntos por QBIC e refit beta-binomial dos `top` melhores.

    QBIC = -2 l_binomial / phi_cheio + k ln N, com phi_cheio o phi de Pearson do modelo
    com todas as candidatas. O tempo da triagem fica em `resultado.attrs["tempo_triagem_s"]`.
    """
    candidatas = tuple(candidatas)
    m = len(candidatas)
    base_vars = _base(termos, candidatas, fixas)
    termo_cheio = replace(termos, vars=base_vars + candidatas)
    pad = Padronizador(mapa).fit(df)
    Xdf = montar_X(df, mapa, pad, termo_cheio)
    off = Xdf.pop(_COL_OFFSET).to_numpy(float) if _COL_OFFSET in Xdf else np.zeros(len(df))
    y, n = df[mapa["y"]].to_numpy(float), df[mapa["n"]].to_numpy(float)
    cols = list(Xdf.columns)
    cand_idx = [cols.index(v) for v in candidatas]
    base_idx = [i for i in range(len(cols)) if i not in cand_idx]
    X = Xdf.to_numpy(float)
    _, pi = _irls_logit(X, y, n, off)
    phi = float(np.sum((y - n * pi) ** 2 / (n * pi * (1 - pi))) / (len(y) - X.shape[1]))
    const = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    _CTX.clear()
    _CTX.update(
        X=X, y=y, n=n, off=off, base_idx=base_idx, cand_idx=cand_idx, const=const, phi=phi, ln_n=float(np.log(len(df)))
    )
    total = 1 << m
    passo = max(1, total // (max(n_jobs, 1) * 16))
    faixas = [(a, min(a + passo, total)) for a in range(0, total, passo)]
    t0 = time.perf_counter()
    qbic = np.concatenate(_mapear(_triar_faixa, faixas, n_jobs))
    tempo = time.perf_counter() - t0

    melhores = np.argsort(qbic, kind="stable")[:top]
    subs = [tuple(candidatas[i] for i in range(m) if int(mk) >> i & 1) for mk in melhores]
    _preparar(df, mapa, termos, candidatas, fixas, "betabin")
    res = _mapear(_ajustar_vars, subs, n_jobs)
    ln_n = np.log(len(df))
    out = pd.DataFrame(
        {
            "variaveis": subs,
            "n_vars": [len(s) for s in subs],
            "qbic": qbic[melhores],
            "bic_bb": [-2 * ll + k * ln_n for ll, k in res],
            "aic_bb": [-2 * ll + 2 * k for ll, k in res],
        }
    )
    out["rank_qbic"] = out["qbic"].rank(method="min").astype(int)
    out["rank_bic_bb"] = out["bic_bb"].rank(method="min").astype(int)
    out.attrs["tempo_triagem_s"] = tempo
    out.attrs["n_subconjuntos"] = total
    return out.sort_values("rank_bic_bb", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Comparação de métodos
# --------------------------------------------------------------------------
def comparar_metodos(
    resultados: dict[str, tuple[Any, ...]],
    df: pd.DataFrame,
    mapa: dict[str, str],
    termos: Termos,
    referencia: str = "todos",
    fixas: Sequence[str] = ("lag",),
    familia: str = "betabin",
    n_jobs: int = 14,
) -> pd.DataFrame:
    """Tabela método x variáveis com BIC/AIC beta-binomial e diferenças vs `referencia`.

    `resultados[nome][0]` é a tupla de variáveis escolhidas (como devolvem as funções
    deste módulo). Refaz o ajuste de cada conjunto, por isso exige df, mapa e termos.
    """
    nomes = list(resultados)
    conjuntos = [tuple(resultados[k][0]) for k in nomes]
    todas = tuple(dict.fromkeys(v for c in conjuntos for v in c))
    _preparar(df, mapa, termos, todas, fixas, familia)
    res = _mapear(_ajustar_vars, conjuntos, n_jobs)
    ln_n = np.log(len(df))
    ref = set(resultados[referencia][0]) if referencia in resultados else set()
    linhas = []
    for nome, conj, (ll, k) in zip(nomes, conjuntos, res, strict=True):
        s = set(conj)
        linhas.append(
            {
                "metodo": nome,
                "n_vars": len(conj),
                "variaveis": conj,
                "BIC_bb": -2 * ll + k * ln_n,
                "AIC_bb": -2 * ll + 2 * k,
                "a_mais_vs_ref": tuple(sorted(s - ref)),
                "a_menos_vs_ref": tuple(sorted(ref - s)),
            }
        )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# 5. Testes de fusão de níveis (região e bloco) por razão de verossimilhanças
# --------------------------------------------------------------------------
def ajuste_bb(X: pd.DataFrame, y: np.ndarray, n: np.ndarray) -> Any:
    """Beta-binomial logit sobre `X` (BFGS polido por L-BFGS, fica o de maior loglik)."""
    m = M.BetaBinomial(y, X.to_numpy(float), n, link=LIGACOES["logit"])
    r = m.fit()
    p = m.fit(start_params=r.params, method="lbfgs", maxiter=3000)
    return p if p.llf >= r.llf else r


def tabela_niveis(res: Any, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coeficiente, EP de Wald, z, p e razão de chances (com IC95%) dos níveis `cols`."""
    b = pd.Series(np.asarray(res.params)[:-1], index=X.columns)[cols]
    ep = pd.Series(np.asarray(res.bse)[:-1], index=X.columns)[cols]
    z = b / ep
    return pd.DataFrame(
        {
            "beta": b,
            "EP": ep,
            "z": z,
            "p": 2 * stats.norm.sf(z.abs()),
            "OR": np.exp(b),
            "OR_inf95": np.exp(b - 1.96 * ep),
            "OR_sup95": np.exp(b + 1.96 * ep),
        }
    )


def lrt(
    res_ref: Any, X_ref: pd.DataFrame, X_red: pd.DataFrame, y: np.ndarray, n: np.ndarray
) -> tuple[float, int, float]:
    """(Delta, gl, p) do LRT do modelo de referência contra o reduzido `X_red`."""
    d = max(2 * (res_ref.llf - ajuste_bb(X_red, y, n).llf), 0.0)
    gl = X_ref.shape[1] - X_red.shape[1]
    return d, gl, float(stats.chi2.sf(d, gl))


def fundir(X: pd.DataFrame, cols: list[str], prefixo: str) -> pd.DataFrame:
    """Troca as colunas `cols` por uma só (a soma), nomeada `prefixo` + níveis unidos por '+'."""
    novo = prefixo + "+".join(c[len(prefixo) :] for c in cols)
    Xn = X.drop(columns=cols)
    Xn[novo] = X[cols].sum(axis=1)
    return Xn


def _niveis_do_fator(df: pd.DataFrame, coluna: str) -> pd.Series:
    """Níveis como texto; ausente vira "NA" (ou "False", se a coluna só tem booleanos)."""
    v = df[coluna]
    so_bool = v.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all()
    return (
        v.map(lambda x: str(bool(x)) if so_bool and pd.notna(x) else x).fillna("False" if so_bool else "NA").astype(str)
    )


def tabela_descritiva(df: pd.DataFrame, coluna: str, ref: str, y_col: str, n_col: str) -> pd.DataFrame:
    """Uma linha por nível: totais, % de votos, logit empírico, mediana/IQR municipal e psi.

    psi = razão de chances do nível contra `ref` sobre os totais de votos, com IC95% de
    Woolf. Esses IC ignoram a superdispersão entre municípios e são só descritivos
    (Paula 2.10.5, 4.4). Ausente vira o nível "NA".
    """
    g = _niveis_do_fator(df, coluna)
    y, n = df[y_col].astype(float), df[n_col].astype(float)
    pct = 100 * y / n
    por = pd.DataFrame({"g": g, "y": y, "n": n, "pct": pct}).groupby("g")
    soma_y, soma_n = por["y"].sum(), por["n"].sum()
    t = pd.DataFrame({"municipios": por.size(), "soma_y": soma_y, "soma_n": soma_n})
    t["pct_lula"] = 100 * soma_y / soma_n
    t["logit_emp"] = np.log(soma_y / (soma_n - soma_y))
    t["mediana_pct"] = por["pct"].median()
    t["iqr_pct"] = por["pct"].quantile(0.75) - por["pct"].quantile(0.25)
    ln_psi = t["logit_emp"] - t.loc[ref, "logit_emp"]
    ep = np.sqrt(1 / soma_y + 1 / (soma_n - soma_y) + 1 / soma_y[ref] + 1 / (soma_n[ref] - soma_y[ref]))
    t["psi"] = np.exp(ln_psi)
    t["psi_inf95"] = np.exp(ln_psi - 1.96 * ep)
    t["psi_sup95"] = np.exp(ln_psi + 1.96 * ep)
    t.loc[ref, ["psi", "psi_inf95", "psi_sup95"]] = [1.0, np.nan, np.nan]
    return t


def teste_univariado(df: pd.DataFrame, mapa: dict[str, str], chave: str, ref: str) -> dict[str, float]:
    """LRT do fator sozinho (const + dummies dos níveis) contra só a constante (Paula 4.6.5, passo 1)."""
    g = _niveis_do_fator(df, mapa[chave])
    y, n = df[mapa["y"]].to_numpy(float), df[mapa["n"]].to_numpy(float)
    X0 = pd.DataFrame({"const": np.ones(len(df))})
    X = X0.copy()
    for nivel in sorted(set(g) - {ref}):
        X[f"{chave}_{nivel}"] = (g == nivel).to_numpy(float)
    res = ajuste_bb(X, y, n)
    d, gl, p = lrt(res, X, X0, y, n)
    return {"delta": d, "gl": float(gl), "p": p, "loglik": float(res.llf)}


def mantel_haenszel(df: pd.DataFrame, binaria: str, estrato: str, y_col: str, n_col: str) -> dict[str, float]:
    """OR bruto (Woolf), OR de Mantel-Haenszel por `estrato` (IC de Robins-Breslow-Greenland)
    e teste de homogeneidade dos estratos (Breslow-Day), sobre totais de votos.

    Ausente (None) na binária conta como False. IC sem correção de superdispersão. Estrato
    com linha ou coluna vazia (ex.: região sem nenhum exposto) não informa sobre a OR e fica
    fora do MH e do Breslow-Day, que seria indefinido; o OR bruto usa todos os estratos.
    """
    flag = df[binaria].map(lambda x: bool(x) if pd.notna(x) else False).to_numpy()
    y, n = df[y_col].to_numpy(float), df[n_col].to_numpy(float)
    est = df[estrato].astype(str).to_numpy()
    tabelas = []
    for e in sorted(set(est)):
        m = est == e
        tabelas.append(
            np.array([[y[m & flag].sum(), (n - y)[m & flag].sum()], [y[m & ~flag].sum(), (n - y)[m & ~flag].sum()]])
        )
    bruta = np.sum(tabelas, axis=0)
    or_b = (bruta[0, 0] * bruta[1, 1]) / (bruta[0, 1] * bruta[1, 0])
    ep_b = np.sqrt((1 / bruta).sum())
    informativas = [t for t in tabelas if (t.sum(axis=0) > 0).all() and (t.sum(axis=1) > 0).all()]
    st = StratifiedTable(informativas)
    inf, sup = st.oddsratio_pooled_confint()
    return {
        "or_bruto": float(or_b),
        "or_bruto_inf95": float(or_b * np.exp(-1.96 * ep_b)),
        "or_bruto_sup95": float(or_b * np.exp(1.96 * ep_b)),
        "or_mh": float(st.oddsratio_pooled),
        "or_mh_inf95": float(inf),
        "or_mh_sup95": float(sup),
        "p_homogeneidade": float(st.test_equal_odds().pvalue),
        "estratos_usados": len(informativas),
    }


def analisar_fator(
    X: pd.DataFrame,
    y: np.ndarray,
    n: np.ndarray,
    cols: list[str],
    rotulo: str,
    prefixo: str,
    p_fusao: float = 0.05,
    verbose: bool = True,
    fundir_na_referencia: bool = True,
    ref: str = "ref",
    passos: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, Any, list[str]]:
    """Tabela por nível, LRT de retirar o fator e fusão sequencial (regra: p > p_fusao).

    Candidatos a cada passo: todo par de níveis e, com `fundir_na_referencia`, todo nível
    na referência `ref` (retira sua dummy). Funde-se o de maior p se p > p_fusao e
    reajusta-se. Devolve (X com as fusões, ajuste final, colunas do fator).
    Se `passos` é dada, recebe a tabela de cada passo (nivel_a, nivel_b, delta, gl, p, passo,
    fundido); o retorno não muda.
    """
    res, passo, ref_atual = ajuste_bb(X, y, n), 0, ref
    while True:
        if verbose:
            print(f"=== {rotulo}, passo {passo}: {len(cols)} níveis fora da referência; loglik = {res.llf:,.2f}")
            print(tabela_niveis(res, X, cols).round(4).to_string())
        if not cols:
            break
        d, gl, p = lrt(res, X, X.drop(columns=cols), y, n)
        if verbose:
            print(f"LRT (todos os níveis = 0, gl={gl}): Δ = {d:.1f}, p = {p:.2e}")
        candidatos: list[tuple[tuple[str, ...], pd.DataFrame]] = [
            (par, fundir(X, list(par), prefixo)) for par in itertools.combinations(cols, 2)
        ]
        if fundir_na_referencia:
            candidatos += [((c,), X.drop(columns=c)) for c in cols]
        if not candidatos:
            break
        linhas = []
        for alvo, X_red in candidatos:
            d2, gl2, p2 = lrt(res, X, X_red, y, n)
            linhas.append(
                {
                    "nivel_a": alvo[0][len(prefixo) :],
                    "nivel_b": alvo[1][len(prefixo) :] if len(alvo) > 1 else ref_atual,
                    "delta": d2,
                    "gl": gl2,
                    "p": p2,
                }
            )
        T = pd.DataFrame(linhas)
        if verbose:
            print("LRT de igualdade entre níveis (H0: β_a = β_b):")
            print(T.round(4).to_string(index=False))
        i = int(T.p.idxmax())
        if passos is not None:
            passos.append(T.assign(passo=passo, fundido=(T.index == i) & (T.p > p_fusao)))
        if T.p[i] <= p_fusao:
            if verbose:
                print(f"Nenhum candidato com p > {p_fusao}: nenhum nível é fundido.")
            break
        alvo, X_novo = candidatos[i]
        if verbose:
            print(f"Funde {T.nivel_a[i]} e {T.nivel_b[i]} (p = {T.p[i]:.3f} > {p_fusao}) e reajusta.")
        if len(alvo) == 1:
            ref_atual = f"{ref_atual}+{T.nivel_a[i]}"
        novas = [c for c in X_novo.columns if c not in X.columns]
        cols = [c for c in cols if c not in alvo] + novas
        X = X_novo
        res, passo = ajuste_bb(X, y, n), passo + 1
    return X, res, cols


# --------------------------------------------------------------------------
# 6. Seleção aninhada no LOSO (a seleção e as fusões só veem as UFs de treino)
# --------------------------------------------------------------------------
METODOS_BIC: tuple[Metodo, ...] = ("backward", "forward", "both")
ARQ_ANINHADA = Path(__file__).resolve().parents[1] / "results/cv/loso_bb_selecao_aninhada.parquet"
ARQ_DADOS = Path(__file__).resolve().parents[1] / "data/processed/municipios.parquet"
_NOME_ANINHADO = "bb_selecao_aninhada"


def _agora() -> str:
    return time.strftime("%H:%M:%S")


def _melhor_bic(
    df: pd.DataFrame,
    mapa: dict[str, str],
    termos: Termos,
    candidatas: Sequence[str],
    n_jobs: int,
    uf: str,
    log: Callable[[str], None],
) -> tuple[Metodo, tuple[str, ...], float]:
    """Roda backward, forward e both por BIC; devolve (método, conjunto, BIC) de menor BIC."""
    achados: list[tuple[float, Metodo, tuple[str, ...]]] = []
    for metodo in METODOS_BIC:
        t0 = time.time()
        log(f"STEP {uf}:{metodo} [{_agora()}]")
        vs, trilha = passo_a_passo(df, mapa, termos, metodo, "bic", candidatas=candidatas, n_jobs=n_jobs)
        bic = float(trilha["valor_criterio"].iloc[-1])
        achados.append((bic, metodo, vs))
        log(f"{uf} {metodo}: {len(vs)} contínuas, BIC={bic:.2f} ({time.time() - t0:.0f}s)")
    bic, metodo, vs = min(achados, key=lambda a: a[0])  # empate: a primeira da ordem fixa
    return metodo, vs, bic


def _fusoes(
    df: pd.DataFrame, mapa: dict[str, str], termos: Termos, p_fusao: float
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """(grupos de regiões, grupos de níveis do bloco) pela regra de fusão por LRT.

    Região e bloco partem dos níveis brutos; níveis sem efeito vão para a referência (bloco
    em `Termos.bloco_grupos`). Os testes usam o modelo cheio, como a Parte 8.1.
    """
    y, n = df[mapa["y"]].to_numpy(float), df[mapa["n"]].to_numpy(float)
    pad = Padronizador(mapa).fit(df)
    X = montar_X(df, mapa, pad, replace(termos, bloco_grupos=tuple((b,) for b in BLOCOS)))
    cols_reg = [c for c in X.columns if c.startswith("reg_")]
    X_reg, _, cols_reg = analisar_fator(X, y, n, cols_reg, "Região", "reg_", p_fusao, verbose=False, ref="Sudeste")
    grupos = tuple(tuple(c[4:].split("+")) for c in cols_reg if "+" in c)
    presentes = {r for c in cols_reg for r in c[4:].split("+")}
    na_ref = tuple(r for r in REGIOES if r not in presentes)
    if na_ref:
        grupos += ((*na_ref, REGIAO_REF),)
    cols_b = [f"bloco_{b}" for b in BLOCOS]
    _, _, cols_blo = analisar_fator(X_reg, y, n, cols_b, "Bloco", "bloco_", p_fusao, verbose=False, ref="direita")
    return grupos, tuple(tuple(c[6:].split("+")) for c in cols_blo)


def selecao_aninhada_loso(
    df: pd.DataFrame,
    mapa: dict[str, str],
    uf_retida: str,
    candidatas: Sequence[str] = CANDIDATAS_PADRAO,
    n_jobs: int = 14,
    p_fusao: float = 0.05,
    termos: Termos = TERMOS_PADRAO,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Um fold do LOSO com fusões e seleção refeitas só nas UFs de treino (ordem do nb02).

    (1) LRT de fusão de regiões e de blocos (p > `p_fusao`) no beta-binomial cheio;
    (2) passo a passo por BIC (backward, forward, both; `lag`, região e bloco sempre
    dentro, com as fusões de (1)) e fica o de menor BIC; (3) ajusta no treino e prevê a
    UF retida com n̂ (mesma convenção de `especificacoes.prever_fold`).
    """
    df = df.reset_index(drop=True)
    uf = df[mapa["uf"]].astype(str).to_numpy()
    treino, teste = np.flatnonzero(uf != uf_retida), np.flatnonzero(uf == uf_retida)
    if len(teste) == 0:
        raise KeyError(f"UF {uf_retida!r} ausente em df")
    t0 = time.time()
    df_tr = df.iloc[treino]
    for fase in ("fusoes", *METODOS_BIC, "previsao"):
        log(f"PLAN {uf_retida}:{fase}")
    log(f"STEP {uf_retida}:fusoes [{_agora()}]")
    grupos, grupos_bloco = _fusoes(df_tr, mapa, termos, p_fusao)
    termos_fus = replace(termos, regiao_fundir=grupos, bloco_grupos=grupos_bloco)
    metodo, vs, bic = _melhor_bic(df_tr, mapa, termos_fus, candidatas, n_jobs, uf_retida, log)
    variaveis = ("lag", *vs)
    log(f"STEP {uf_retida}:previsao [{_agora()}]")
    E.definir_dados_cv(df, mapa)
    E.EXTENSOES[_NOME_ANINHADO] = Candidato(_NOME_ANINHADO, "betabin", "logit", replace(termos_fus, vars=variaveis))
    try:
        pi = E.prever_fold(_NOME_ANINHADO, treino, teste)
    finally:
        E.EXTENSOES.pop(_NOME_ANINHADO, None)
    nh = E._DADOS_CV["n_hat"][teste]
    prev = float(M.agregar(pi, nh, uf[teste]).drop("Brasil").iloc[0])
    obs = float(df[mapa["y"]].to_numpy(float)[teste].sum() / df[mapa["n"]].to_numpy(float)[teste].sum())
    return {
        "uf": uf_retida,
        "metodo": metodo,
        "variaveis": ",".join(vs),
        "n_vars": len(vs),
        "bic_treino": bic,
        "fusao_regiao": ";".join("+".join(g) for g in grupos),
        "fusao_bloco": ";".join("+".join(g) for g in grupos_bloco),
        "prev_pct": prev * 100,
        "obs_pct": obs * 100,
        "erro_pp": (prev - obs) * 100,
        "erro_abs_pp": abs(prev - obs) * 100,
        "tempo_s": time.time() - t0,
    }


def mae_uf_pp(tabela: pd.DataFrame) -> float:
    """MAE_UF do LOSO: média do |share previsto - share real| (p.p.) sobre as UFs."""
    return float(tabela["erro_abs_pp"].mean())


def _gravar_atomico(tabela: pd.DataFrame, arq: Path) -> None:
    arq.parent.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_suffix(".tmp")
    tabela.to_parquet(tmp, index=False)
    os.replace(tmp, arq)


def _fold_com_marcador(uf: str, n_jobs: int, df: pd.DataFrame, mapa: dict[str, str]) -> dict[str, Any]:
    print(f"=== {uf} {_agora()}", flush=True)
    return selecao_aninhada_loso(df, mapa, uf, n_jobs=n_jobs, log=lambda m: print(m, flush=True))


def rodar_folds(
    ufs: Sequence[str],
    lote: str,
    status: Path,
    arq: Path = ARQ_ANINHADA,
    n_jobs: int = 14,
    folds_paralelos: int = 1,
    refazer: bool = False,
    df: pd.DataFrame | None = None,
    mapa: dict[str, str] = E.MAPA_2026,
) -> pd.DataFrame:
    """Roda os folds de `ufs` e grava cada resultado em `arq` (retomável: pula as UFs prontas).

    Escreve em `status` uma linha `<lote> <UF> OK|FAIL <detalhe> <HH:MM:SS>` por fold e
    `DONE` no fim. Com `folds_paralelos` > 1 os folds rodam em processos (fork), cada um
    com `n_jobs` processos internos no passo a passo.
    """
    dados = pd.read_parquet(ARQ_DADOS).reset_index(drop=True) if df is None else df
    feito = pd.read_parquet(arq) if arq.exists() else pd.DataFrame()
    prontas = set() if refazer or feito.empty else set(feito["uf"])
    pend = [u for u in ufs if u not in prontas]
    status.parent.mkdir(parents=True, exist_ok=True)

    def linha(texto: str) -> None:
        with status.open("a") as f:
            f.write(texto + "\n")

    for u in ufs:
        if u in prontas:
            linha(f"{lote} {u} OK ja_pronto {_agora()}")
    if not feito.empty and refazer:
        feito = feito[~feito["uf"].isin(pend)]
    resultados = feito
    with ProcessPoolExecutor(max_workers=max(folds_paralelos, 1), mp_context=get_context("fork")) as ex:
        fut = {ex.submit(_fold_com_marcador, u, n_jobs, dados, mapa): u for u in pend}
        for f in as_completed(fut):
            u = fut[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - o status registra a falha e o lote segue
                print(f"FAIL {u}: {e!r}", flush=True)
                linha(f"{lote} {u} FAIL {type(e).__name__} {_agora()}")
                continue
            resultados = pd.concat([resultados, pd.DataFrame([r])], ignore_index=True)
            _gravar_atomico(resultados.drop_duplicates("uf", keep="last"), arq)
            linha(f"{lote} {u} OK erro={r['erro_pp']:+.3f}pp t={r['tempo_s']:.0f}s {_agora()}")
    linha("DONE")
    return resultados


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LOSO com seleção de variáveis aninhada")
    ap.add_argument("--ufs", nargs="*", default=None, help="UFs (padrão: as 27)")
    ap.add_argument("--lote", default="aninhada")
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--folds-paralelos", type=int, default=1)
    ap.add_argument("--refazer", action="store_true")
    a = ap.parse_args(argv)
    dados = pd.read_parquet(ARQ_DADOS).reset_index(drop=True)
    ufs = a.ufs or sorted(dados[E.MAPA_2026["uf"]].astype(str).unique())
    r = rodar_folds(
        ufs, a.lote, a.status, n_jobs=a.n_jobs, folds_paralelos=a.folds_paralelos, refazer=a.refazer, df=dados
    )
    print(f"MAE_UF aninhado ({len(r)} UFs): {mae_uf_pp(r):.4f} p.p.", flush=True)


if __name__ == "__main__":
    main()
