"""Caracterização: os helpers de src/validacao.py reproduzem o código que estava nos notebooks 03 e 05."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import modelos as M
from src import validacao as V

NOMES = ["bin_logit", "bin_probit"]
K = {"bin_logit": 20, "bin_probit": 25}


def prever_falso(nome: str, treino: np.ndarray, teste: np.ndarray) -> np.ndarray:
    """Previsão determinística que depende só do modelo e do município de teste."""
    return 0.2 + 0.5 * ((teste * 7 + len(nome)) % 10) / 10


def prever_proibido(nome: str, treino: np.ndarray, teste: np.ndarray) -> np.ndarray:
    raise AssertionError("o cache deveria bastar")


def sintetico() -> V.Dados:
    rng = np.random.default_rng(3)
    uf = np.repeat(["A", "B", "C", "D"], 15)
    reg = np.repeat(["N", "S"], 30)
    n = rng.integers(200, 900, 60).astype(float)
    y = rng.binomial(n.astype(int), 0.4 + 0.05 * rng.standard_normal(60).clip(-2, 2)).astype(float)
    nh = n * rng.uniform(0.9, 1.1, 60)
    return V.Dados(y=y, n=n, uf=uf, reg=reg, nh=nh)


def referencia(y, n, nh, uf, reg, K):
    """Cópia literal de `erros`, `tabela` e `calib` do notebook 03 (antes da extração)."""
    UFS_ = sorted(set(uf))
    obs_uf = pd.Series(y).groupby(uf).sum() / pd.Series(n).groupby(uf).sum()
    obs_reg = pd.Series(y).groupby(reg).sum() / pd.Series(n).groupby(reg).sum()
    from src import especificacoes as E

    def erros(pi, esq, folds):
        a_uf = (M.agregar(pi, nh, uf).drop("Brasil") - obs_uf).abs() * 100
        a_rg = (M.agregar(pi, nh, reg).drop("Brasil") - obs_reg).abs() * 100
        m = M.metricas(y, n, pi)
        if esq == "5-fold": por = [M.metricas(y[te], n[te], pi[te])["mae_pond_pp"] for _, te in folds]
        else: por = [M.metricas(y[te], n[te], pi[te])["mae_pond_pp"] for _, te in folds]
        m.update(mae_uf_pp=a_uf.mean(), se_uf_pp=a_uf.std(ddof=1) / np.sqrt(len(a_uf)),
                 mae_reg_pp=a_rg.mean(), max_reg_pp=a_rg.max(),
                 se_mae_pond_pp=np.std(por, ddof=1) / np.sqrt(len(por)))
        return m, a_uf, a_rg

    def tabela(PI, esq, folds):
        L, UFE = [], []
        for k, pi in PI.items():
            m, a_uf, _ = erros(pi, esq, folds)  # noqa: RUF059 (cópia literal do notebook)
            L.append({"modelo": k, "familia": E.CANDIDATOS[k].familia, "k": K[k], **m})
            if esq == "LOSO":
                pu = M.agregar(pi, nh, uf).drop("Brasil")
                mu = pd.Series({u: M.metricas(y[uf == u], n[uf == u], pi[uf == u])["mae_pond_pp"] for u in UFS_})
                UFE.append(pd.DataFrame({"modelo": k, "uf": UFS_, "obs_pct": obs_uf[UFS_].values * 100, "prev_pct": pu[UFS_].values * 100,
                                         "erro_uf_pp": (pu[UFS_] - obs_uf[UFS_]).values * 100, "mae_pond_municipal_pp": mu[UFS_].values}))
        t = pd.DataFrame(L)
        return t, (pd.concat(UFE) if UFE else None)

    def calib(pi, q=20):
        d = pd.DataFrame({"pi": pi, "y": y, "n": n}); d["b"] = pd.qcut(d.pi, q, duplicates="drop")
        g = d.assign(v=d.pi * d.n).groupby("b", observed=True)[["v", "y", "n"]].sum()
        return g.v / g.n, g.y / g.n

    return erros, tabela, calib


def referencia_rodar(tam: int, CV: Path, RECALCULAR: bool):
    """Cópia literal de `_cv` e `rodar` do notebook 03, com o prever trocado pelo falso."""
    def _cv(nome, esquema, folds):
        arq = CV / f"{esquema}_{nome}.parquet"
        if arq.exists() and not RECALCULAR:
            return pd.read_parquet(arq)["pi"].to_numpy()
        return None

    def rodar(nomes, esquema, folds):
        pend = [k for k in nomes if _cv(k, esquema, folds) is None]
        saida = {}
        if pend:
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context("fork")) as ex:
                fut = {(k, i): ex.submit(prever_falso, k, tr, te) for k in pend for i, (tr, te) in enumerate(folds)}
                for k in pend:
                    pi = np.full(tam, np.nan)
                    for i, (tr, te) in enumerate(folds): pi[te] = fut[(k, i)].result()
                    pd.DataFrame({"pi": pi}).to_parquet(CV / f"{esquema}_{k}.parquet"); saida[k] = pi
        for k in nomes:
            if k not in saida: saida[k] = _cv(k, esquema, folds)
        return saida

    return rodar


@pytest.fixture
def d() -> V.Dados:
    return sintetico()


@pytest.fixture
def previsoes(d: V.Dados) -> dict[str, np.ndarray]:
    todos = np.arange(len(d.y))
    return {k: prever_falso(k, todos, todos) for k in NOMES}


@pytest.mark.parametrize("esq", ["5-fold", "LOSO"])
def test_erros_igual_ao_notebook(d: V.Dados, previsoes: dict[str, np.ndarray], esq: str) -> None:
    folds = M.folds_kfold(len(d.y), k=5, seed=0) if esq == "5-fold" else M.folds_por_grupo(d.uf)
    ref_erros, _, _ = referencia(d.y, d.n, d.nh, d.uf, d.reg, K)
    for pi in previsoes.values():
        esperado, _, _ = ref_erros(pi, esq, folds)
        assert V.erros(pi, d, folds) == esperado


@pytest.mark.parametrize("esq", ["5-fold", "LOSO"])
def test_tabela_igual_ao_notebook(d: V.Dados, previsoes: dict[str, np.ndarray], esq: str) -> None:
    folds = M.folds_kfold(len(d.y), k=5, seed=0) if esq == "5-fold" else M.folds_por_grupo(d.uf)
    _, ref_tabela, _ = referencia(d.y, d.n, d.nh, d.uf, d.reg, K)
    esperado, esperado_uf = ref_tabela(previsoes, esq, folds)
    pd.testing.assert_frame_equal(V.tabela(previsoes, d, folds, K), esperado, check_exact=True)
    if esq == "LOSO":
        pd.testing.assert_frame_equal(V.tabela_uf(previsoes, d), esperado_uf, check_exact=True)


def test_calib_igual_ao_notebook(d: V.Dados, previsoes: dict[str, np.ndarray]) -> None:
    _, _, ref_calib = referencia(d.y, d.n, d.nh, d.uf, d.reg, K)
    for pi in previsoes.values():
        for q in (20, 5):
            p, o = V.calib(pi, d, q=q)
            pe, oe = ref_calib(pi, q=q)
            pd.testing.assert_series_equal(p, pe, check_exact=True)
            pd.testing.assert_series_equal(o, oe, check_exact=True)


def test_obs_uf_e_obs_reg_uma_vez(d: V.Dados) -> None:
    assert d.obs_uf is d.obs_uf and d.obs_reg is d.obs_reg
    assert d.obs_uf["A"] == d.y[d.uf == "A"].sum() / d.n[d.uf == "A"].sum()


@pytest.mark.parametrize("esq", ["kfold", "loso"])
def test_rodar_igual_ao_notebook_recalculando(tmp_path: Path, d: V.Dados, esq: str) -> None:
    folds = M.folds_kfold(len(d.y), k=5, seed=0) if esq == "kfold" else M.folds_por_grupo(d.uf)
    (tmp_path / "ref").mkdir(); (tmp_path / "novo").mkdir()
    esperado = referencia_rodar(len(d.y), tmp_path / "ref", True)(NOMES, esq, folds)
    obtido = V.rodar(NOMES, esq, folds, n_linhas=len(d.y), pasta=tmp_path / "novo", recalcular=True,
                     prever=prever_falso, max_workers=2)
    assert list(obtido) == list(esperado)
    for k in NOMES:
        np.testing.assert_array_equal(obtido[k], esperado[k])
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "novo" / f"{esq}_{k}.parquet"),
                                      pd.read_parquet(tmp_path / "ref" / f"{esq}_{k}.parquet"))


def test_rodar_sem_recalcular_usa_so_o_cache(tmp_path: Path, d: V.Dados) -> None:
    folds = M.folds_por_grupo(d.uf)
    V.rodar(NOMES, "loso", folds, n_linhas=len(d.y), pasta=tmp_path, recalcular=True, prever=prever_falso, max_workers=2)
    lido = V.rodar(NOMES, "loso", folds, n_linhas=len(d.y), pasta=tmp_path, recalcular=False, prever=prever_proibido, max_workers=2)
    ref = referencia_rodar(len(d.y), tmp_path, False)(NOMES, "loso", folds)
    for k in NOMES:
        np.testing.assert_array_equal(lido[k], ref[k])


def test_rodar_sem_recalcular_calcula_so_o_que_falta(tmp_path: Path, d: V.Dados) -> None:
    folds = M.folds_por_grupo(d.uf)
    V.rodar(["bin_logit"], "loso", folds, n_linhas=len(d.y), pasta=tmp_path, recalcular=True, prever=prever_falso, max_workers=2)
    antes = (tmp_path / "loso_bin_logit.parquet").stat().st_mtime_ns
    saida = V.rodar(NOMES, "loso", folds, n_linhas=len(d.y), pasta=tmp_path, recalcular=False, prever=prever_falso, max_workers=2)
    ref = referencia_rodar(len(d.y), tmp_path, False)(NOMES, "loso", folds)
    assert (tmp_path / "loso_bin_logit.parquet").stat().st_mtime_ns == antes
    assert (tmp_path / "loso_bin_probit.parquet").exists()
    for k in NOMES:
        np.testing.assert_array_equal(saida[k], ref[k])


def test_linha_de_csv_troca_so_o_esquema_e_tira_familia_e_k(tmp_path: Path) -> None:
    arq = tmp_path / "cv.csv"
    pd.DataFrame({"modelo": ["a", "b"], "familia": ["f", "g"], "k": [1, 2], "mae_uf_pp": [0.1 + 0.2, 1 / 3]}).to_csv(arq, index=False)
    linha = V.linha_de_csv(arq, "LOSO", "b")
    assert linha == {"esquema": "LOSO", "modelo": "b", "mae_uf_pp": 1 / 3}
    assert list(linha) == ["esquema", "modelo", "mae_uf_pp"]
