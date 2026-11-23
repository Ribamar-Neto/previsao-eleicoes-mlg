"""`simular_f` e `resumo_f` reproduzem o código que estava no notebook 05 (mesma semente, mesmos sorteios)."""

import numpy as np
import pandas as pd
import pytest

from src import apuracao as A
from src import especificacoes as E
from src import modelos as M

UFS = ["AA", "BB", "CC"]
REGS = ["N", "S"]


def _partes(theta):
    rng = np.random.default_rng(1)
    partes, ini = {}, 0
    for k, u in enumerate(UFS):
        te = np.arange(ini, ini + 4 + k)
        ini += 4 + k
        X = np.column_stack([np.ones(len(te)), rng.normal(size=len(te))])
        cov = np.array([[0.04, 0.0], [0.0, 0.02]])
        partes[u] = {"beta": np.array([0.1, 0.3]), "cov_conley": cov, "cov_modelo": cov / 2, "X": X, "te": te,
                     "theta": theta(len(te))}
    return partes


def _simular_antigo(partes, chave, sigma, nhat, link, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    return np.column_stack([M.simular_votos_grupo(v["beta"], v[chave], v["X"], nhat[v["te"]], np.zeros(len(v["te"])), link=link,
                            theta=v["theta"], sigma=sigma, B=B, rng=rng)[:, 0] for v in partes.values()])


def _resumo_antigo(V, ufs, regs, n_uf, uf_reg, obs_uf, obs_reg, obs_br, prev_uf_f):
    linhas = {}
    for i, u in enumerate(ufs): linhas[("uf", u)] = (V[:, i], n_uf[i], obs_uf[u])
    for r in regs:
        m = uf_reg == r; linhas[("regiao", r)] = (V[:, m].sum(1), n_uf[m].sum(), obs_reg[r])
    linhas[("regiao", "Brasil")] = (V.sum(1), n_uf.sum(), obs_br)
    out = []
    for (nivel, nome), (v, nn, real) in linhas.items():
        q = np.quantile(v / nn, [0.025, 0.05, 0.5, 0.95, 0.975])
        pt = (prev_uf_f[ufs].values * n_uf)[uf_reg == nome].sum() / nn if nivel == "regiao" and nome != "Brasil" else \
             (prev_uf_f[nome] if nivel == "uf" else (prev_uf_f[ufs].values * n_uf).sum() / nn)
        out.append({"nivel": nivel, "nome": nome, "lula_pct": pt * 100, "q025": q[0] * 100, "q05": q[1] * 100, "q50": q[2] * 100, "q95": q[3] * 100, "q975": q[4] * 100, "real_pct": real * 100})
    d = pd.DataFrame(out); d["erro_pp"] = d.lula_pct - d.real_pct
    d["dentro_ic95"] = (d.real_pct >= d.q025) & (d.real_pct <= d.q975); d["dentro_ic90"] = (d.real_pct >= d.q05) & (d.real_pct <= d.q95)
    d["largura_pp"] = d.q975 - d.q025; d["largura90_pp"] = d.q95 - d.q05
    return d


@pytest.mark.parametrize("theta", [lambda k: None, lambda k: np.full(k, 40.0)])
@pytest.mark.parametrize("chave,sigma", [("cov_conley", 0.2), ("cov_modelo", 0.0)])
def test_simular_f_igual_ao_notebook(theta, chave, sigma):
    partes = _partes(theta)
    nhat = np.random.default_rng(2).integers(500, 900, size=15).astype(float)
    link = E.LIGACOES["logit"]
    novo = A.simular_f(partes, chave, sigma, nhat=nhat, link=link, B=50)
    antigo = _simular_antigo(partes, chave, sigma, nhat, link, B=50)
    assert novo.shape == (50, 3)
    np.testing.assert_array_equal(novo, antigo)


def test_resumo_f_igual_ao_notebook():
    rng = np.random.default_rng(3)
    n_uf = np.array([3000.0, 4000.0, 5000.0])
    votos = rng.binomial(n_uf.astype(int), [0.4, 0.5, 0.6], size=(200, 3)).astype(float)
    uf_reg = np.array(["N", "N", "S"])
    obs_uf = pd.Series([0.41, 0.52, 0.58], index=UFS)
    obs_reg = pd.Series([0.47, 0.58], index=REGS)
    prev = pd.Series([0.40, 0.51, 0.61], index=UFS)
    novo = A.resumo_f(votos, ufs=UFS, regs=REGS, n_uf=n_uf, uf_reg=uf_reg, obs_uf=obs_uf, obs_reg=obs_reg,
                      obs_brasil=0.5, prev_uf=prev)
    antigo = _resumo_antigo(votos, UFS, REGS, n_uf, uf_reg, obs_uf, obs_reg, 0.5, prev)
    pd.testing.assert_frame_equal(novo, antigo, check_exact=True)
    assert list(novo.nome) == UFS + REGS + ["Brasil"]
