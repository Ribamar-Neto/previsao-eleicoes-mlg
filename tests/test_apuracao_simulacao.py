"""`simular_f` e `resumo_f` reproduzem o código que estava na Parte 16 do `02-MLG` (mesma semente, mesmos sorteios)."""

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
        partes[u] = {
            "beta": np.array([0.1, 0.3]),
            "cov_conley": cov,
            "cov_modelo": cov / 2,
            "X": X,
            "te": te,
            "theta": theta(len(te)),
        }
    return partes


def _simular_antigo(partes, chave, sigma, nhat, link, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            M.simular_votos_grupo(
                v["beta"],
                v[chave],
                v["X"],
                nhat[v["te"]],
                np.zeros(len(v["te"])),
                link=link,
                theta=v["theta"],
                sigma=sigma,
                B=B,
                rng=rng,
            )[:, 0]
            for v in partes.values()
        ]
    )


def _resumo_antigo(V, ufs, regs, n_uf, uf_reg, obs_uf, obs_reg, obs_br, prev_uf_f):
    linhas = {}
    for i, u in enumerate(ufs):
        linhas[("uf", u)] = (V[:, i], n_uf[i], obs_uf[u])
    for r in regs:
        m = uf_reg == r
        linhas[("regiao", r)] = (V[:, m].sum(1), n_uf[m].sum(), obs_reg[r])
    linhas[("regiao", "Brasil")] = (V.sum(1), n_uf.sum(), obs_br)
    out = []
    for (nivel, nome), (v, nn, real) in linhas.items():
        q = np.quantile(v / nn, [0.025, 0.05, 0.5, 0.95, 0.975])
        pt = (
            (prev_uf_f[ufs].values * n_uf)[uf_reg == nome].sum() / nn
            if nivel == "regiao" and nome != "Brasil"
            else (prev_uf_f[nome] if nivel == "uf" else (prev_uf_f[ufs].values * n_uf).sum() / nn)
        )
        out.append(
            {
                "nivel": nivel,
                "nome": nome,
                "lula_pct": pt * 100,
                "q025": q[0] * 100,
                "q05": q[1] * 100,
                "q50": q[2] * 100,
                "q95": q[3] * 100,
                "q975": q[4] * 100,
                "real_pct": real * 100,
            }
        )
    d = pd.DataFrame(out)
    d["erro_pp"] = d.lula_pct - d.real_pct
    d["dentro_ic95"] = (d.real_pct >= d.q025) & (d.real_pct <= d.q975)
    d["dentro_ic90"] = (d.real_pct >= d.q05) & (d.real_pct <= d.q95)
    d["largura_pp"] = d.q975 - d.q025
    d["largura90_pp"] = d.q95 - d.q05
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
    novo = A.resumo_f(
        votos,
        ufs=UFS,
        regs=REGS,
        n_uf=n_uf,
        uf_reg=uf_reg,
        obs_uf=obs_uf,
        obs_reg=obs_reg,
        obs_brasil=0.5,
        prev_uf=prev,
    )
    antigo = _resumo_antigo(votos, UFS, REGS, n_uf, uf_reg, obs_uf, obs_reg, 0.5, prev)
    pd.testing.assert_frame_equal(novo, antigo, check_exact=True)
    assert list(novo.nome) == UFS + REGS + ["Brasil"]


def test_sigma_sem_rms_das_outras_ufs() -> None:
    res = pd.Series([0.3, -0.4, 0.5], index=UFS)
    assert A.sigma_sem(res, ["AA"]) == pytest.approx(np.sqrt((0.16 + 0.25) / 2))
    assert A.sigma_sem(res, ["AA", "BB"]) == pytest.approx(0.5)
    assert A.sigma_sem(res, []) == pytest.approx(np.sqrt((0.09 + 0.16 + 0.25) / 3))


def test_simular_f_sigma_vetor_igual_escalar_quando_constante() -> None:
    partes, nhat, link = _partes(lambda k: None), np.full(15, 700.0), E.LIGACOES["logit"]
    esc = A.simular_f(partes, "cov_conley", 0.2, nhat=nhat, link=link, B=40)
    vet = A.simular_f(partes, "cov_conley", [0.2, 0.2, 0.2], nhat=nhat, link=link, B=40)
    np.testing.assert_array_equal(esc, vet)


def test_estimar_aninhado_usa_sigma_sem_por_uf_e_regiao() -> None:
    partes, nhat, link = _partes(lambda k: None), np.full(15, 700.0), E.LIGACOES["logit"]
    res = pd.Series([0.3, -0.4, 0.5], index=UFS)
    uf_reg = np.array(["N", "N", "S"])
    kw = {
        "ufs": UFS,
        "regs": REGS,
        "n_uf": np.array([3000.0, 4000.0, 5000.0]),
        "uf_reg": uf_reg,
        "obs_uf": pd.Series([0.4, 0.5, 0.6], index=UFS),
        "obs_reg": pd.Series([0.45, 0.6], index=REGS),
        "obs_brasil": 0.5,
        "prev_uf": pd.Series([0.4, 0.5, 0.6], index=UFS),
    }
    d = A.estimar_aninhado(partes, "cov_conley", res, kw, A.Sorteio(nhat, link, n_boot=40))
    assert list(d.nome) == UFS + REGS
    assert "Brasil" not in set(d.nome)
    sig_uf = [A.sigma_sem(res, [u]) for u in UFS]
    ref = A.resumo_f(A.simular_f(partes, "cov_conley", sig_uf, nhat=nhat, link=link, B=40), **kw)
    np.testing.assert_array_equal(d[d.nivel == "uf"].q50, ref[ref.nivel == "uf"].q50)
    ref_s = A.resumo_f(A.simular_f(partes, "cov_conley", A.sigma_sem(res, ["CC"]), nhat=nhat, link=link, B=40), **kw)
    assert d[d.nome == "S"].q50.iloc[0] == ref_s[ref_s.nome == "S"].q50.iloc[0]


def test_simular_f_com_gl_uf_muda_o_choque_e_none_nao() -> None:
    partes, nhat, link = _partes(lambda k: None), np.full(15, 700.0), E.LIGACOES["logit"]
    base = A.simular_f(partes, "cov_conley", 0.2, nhat=nhat, link=link, B=40)
    np.testing.assert_array_equal(base, A.simular_f(partes, "cov_conley", 0.2, nhat=nhat, link=link, B=40, gl_uf=None))
    com_t = A.simular_f(partes, "cov_conley", 0.2, nhat=nhat, link=link, B=40, gl_uf=5)
    assert not np.array_equal(base, com_t)
    vet = A.simular_f(partes, "cov_conley", 0.2, nhat=nhat, link=link, B=40, gl_uf=[5, 5, 5])
    np.testing.assert_array_equal(com_t, vet)


def test_estimar_aninhado_com_t_usa_k_menos_as_ufs_excluidas() -> None:
    partes, nhat, link = _partes(lambda k: None), np.full(15, 700.0), E.LIGACOES["logit"]
    res = pd.Series([0.3, -0.4, 0.5], index=UFS)
    kw = {
        "ufs": UFS,
        "regs": REGS,
        "n_uf": np.array([3000.0, 4000.0, 5000.0]),
        "uf_reg": np.array(["N", "N", "S"]),
        "obs_uf": pd.Series([0.4, 0.5, 0.6], index=UFS),
        "obs_reg": pd.Series([0.45, 0.6], index=REGS),
        "obs_brasil": 0.5,
        "prev_uf": pd.Series([0.4, 0.5, 0.6], index=UFS),
    }
    d = A.estimar_aninhado(partes, "cov_conley", res, kw, A.Sorteio(nhat, link, n_boot=40, gl_uf=7))
    sig_uf = [A.sigma_sem(res, [u]) for u in UFS]
    ref = A.resumo_f(A.simular_f(partes, "cov_conley", sig_uf, nhat=nhat, link=link, B=40, gl_uf=6), **kw)
    np.testing.assert_array_equal(d[d.nivel == "uf"].q50, ref[ref.nivel == "uf"].q50)
    ref_s = A.resumo_f(
        A.simular_f(partes, "cov_conley", A.sigma_sem(res, ["CC"]), nhat=nhat, link=link, B=40, gl_uf=6), **kw
    )
    assert d[d.nome == "S"].q50.iloc[0] == ref_s[ref_s.nome == "S"].q50.iloc[0]
