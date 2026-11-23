"""Validação cruzada: previsões fora da amostra, erros por UF/região e calibração.

Uma só cópia para os notebooks 03 e 05. Regras de negócio:
- o peso das previsões agregadas é n̂ (`dados.nh`), nunca o n real da UF retida;
- o "real" de cada UF e região é o total recalculado sobre as linhas modeladas;
- as previsões em `results/cv/<esquema>_<modelo>.parquet` são o cache da validação.
"""
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest as BT
from src import especificacoes as E
from src import modelos as M

Folds = Sequence[tuple[np.ndarray, np.ndarray]]
Previsoes = dict[str, np.ndarray]
Prever = Callable[[str, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, eq=False)
class Dados:
    """Alvo, pesos e grupos da validação; `nh` é calculado uma vez (`E.n_loso(df)`)."""

    y: np.ndarray
    n: np.ndarray
    uf: np.ndarray
    reg: np.ndarray
    nh: np.ndarray

    @cached_property
    def obs_uf(self) -> pd.Series:
        return pd.Series(self.y).groupby(self.uf).sum() / pd.Series(self.n).groupby(self.uf).sum()

    @cached_property
    def obs_reg(self) -> pd.Series:
        return pd.Series(self.y).groupby(self.reg).sum() / pd.Series(self.n).groupby(self.reg).sum()

    @cached_property
    def ufs(self) -> list[str]:
        return sorted(set(self.uf))


def ler_previsoes(nome: str, esquema: str, pasta: Path) -> np.ndarray:
    return pd.read_parquet(pasta / f"{esquema}_{nome}.parquet")["pi"].to_numpy()


def _do_cache(nome: str, esquema: str, pasta: Path, recalcular: bool) -> np.ndarray | None:
    if (pasta / f"{esquema}_{nome}.parquet").exists() and not recalcular:
        return ler_previsoes(nome, esquema, pasta)
    return None


def rodar(nomes: Sequence[str], esquema: str, folds: Folds, *, n_linhas: int, pasta: Path,
          recalcular: bool, prever: Prever, max_workers: int) -> Previsoes:
    """Previsão fora da amostra de cada modelo; o que falta no cache vai para um pool de forks.

    Os workers herdam o estado de `E.definir_dados_cv` e `E.registrar_extensao`: chame-os antes.
    """
    pend = [k for k in nomes if _do_cache(k, esquema, pasta, recalcular) is None]
    saida: Previsoes = {}
    if pend:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as ex:
            fut = {(k, i): ex.submit(prever, k, tr, te) for k in pend for i, (tr, te) in enumerate(folds)}
            for k in pend:
                pi = np.full(n_linhas, np.nan)
                for i, (_, te) in enumerate(folds):
                    pi[te] = fut[(k, i)].result()
                pd.DataFrame({"pi": pi}).to_parquet(pasta / f"{esquema}_{k}.parquet")
                saida[k] = pi
    for k in nomes:
        if k not in saida:
            saida[k] = ler_previsoes(k, esquema, pasta)
    return saida


def erros(pi: np.ndarray, d: Dados, folds: Folds) -> dict[str, float]:
    """Métricas municipais, MAE do share por UF e região (p.p.) e EP do MAE ponderado entre dobras."""
    a_uf = BT.erros_grupo(pi, d.nh, d.y, d.n, d.uf).drop("Brasil")["erro_pp"].abs()
    a_rg = BT.erros_grupo(pi, d.nh, d.y, d.n, d.reg).drop("Brasil")["erro_pp"].abs()
    m = M.metricas(d.y, d.n, pi)
    por = [M.metricas(d.y[te], d.n[te], pi[te])["mae_pond_pp"] for _, te in folds]
    m.update(mae_uf_pp=a_uf.mean(), se_uf_pp=a_uf.std(ddof=1) / np.sqrt(len(a_uf)),
             mae_reg_pp=a_rg.mean(), max_reg_pp=a_rg.max(),
             se_mae_pond_pp=np.std(por, ddof=1) / np.sqrt(len(por)))
    return m


def tabela(PI: Previsoes, d: Dados, folds: Folds, k: Mapping[str, int]) -> pd.DataFrame:
    """Uma linha por modelo; `k` (nº de parâmetros) vem de `comparacao_modelos.csv` do notebook 02."""
    return pd.DataFrame([{"modelo": nome, "familia": E.CANDIDATOS[nome].familia, "k": k[nome], **erros(pi, d, folds)}
                         for nome, pi in PI.items()])


def tabela_uf(PI: Previsoes, d: Dados) -> pd.DataFrame:
    """Share observado, previsto e erro de cada UF retida, por modelo (só LOSO)."""
    ufs = d.ufs
    partes = []
    for nome, pi in PI.items():
        pu = M.agregar(pi, d.nh, d.uf).drop("Brasil")
        mu = pd.Series({u: M.metricas(d.y[d.uf == u], d.n[d.uf == u], pi[d.uf == u])["mae_pond_pp"] for u in ufs})
        partes.append(pd.DataFrame({"modelo": nome, "uf": ufs, "obs_pct": d.obs_uf[ufs].values * 100,
                                    "prev_pct": pu[ufs].values * 100,
                                    "erro_uf_pp": (pu[ufs] - d.obs_uf[ufs]).values * 100,
                                    "mae_pond_municipal_pp": mu[ufs].values}))
    return pd.concat(partes)


def calib(pi: np.ndarray, d: Dados, q: int = 20) -> tuple[pd.Series, pd.Series]:
    """π̂ médio e share observado (ponderados por votos) em q faixas de π̂."""
    t = pd.DataFrame({"pi": pi, "y": d.y, "n": d.n})
    t["b"] = pd.qcut(t.pi, q, duplicates="drop")
    g = t.assign(v=t.pi * t.n).groupby("b", observed=True)[["v", "y", "n"]].sum()
    return g.v / g.n, g.y / g.n


def linha_de_csv(arquivo: Path, esquema: str, modelo: str) -> dict[str, object]:
    """Linha de um modelo já calculado em outro notebook (cv_kfold/cv_loso), no formato de `final_cv.csv`.

    O texto dos floats lidos e regravados fica idêntico (leitura exata do pandas).
    """
    t = pd.read_csv(arquivo)
    linha = t[t.modelo == modelo].iloc[0].drop(["familia", "k"])
    return {"esquema": esquema, **linha.to_dict()}
