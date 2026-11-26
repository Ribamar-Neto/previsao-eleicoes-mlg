"""Auxiliares movidos da Parte 10 do `02-MLG`: o resultado tem de ser o do código original do notebook."""

import numpy as np
import pandas as pd
import pytest

from src import especificacoes as E
from src import modelos as M
from src import pressupostos as PR

MAPA = E.MAPA_2026


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(1)
    N = 300
    d = pd.DataFrame({MAPA[k]: rng.normal(size=N) for k in E.CONTINUAS})
    d[MAPA["pib_pc"]] = np.exp(rng.normal(10, 0.5, N))
    d[MAPA["bloco_pref"]] = rng.choice(["direita", "esquerda", "centro", "sem_prefeito"], N, p=[0.6, 0.3, 0.06, 0.04])
    d[MAPA["regiao"]] = rng.choice(["Sudeste", "Sul", "Norte", "Nordeste", "Centro-Oeste"], N)
    d[MAPA["uf"]] = rng.choice(E.UFS, N)
    n = rng.integers(1000, 50000, N)
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * d[MAPA["lag"]] + rng.normal(0, 0.15, N))))
    d[MAPA["n"]] = n
    d[MAPA["y"]] = rng.binomial(n, p)
    return d


def test_ajustar_candidato_devolve_e_grava(df, tmp_path):
    a, linha = E.ajustar_candidato("bin_logit", df, MAPA, tmp_path)
    ref = E.CANDIDATOS["bin_logit"].ajustar(df, MAPA)
    assert np.allclose(a.modelo.params, ref.modelo.params)
    assert linha == E.resumo_ajuste(E.CANDIDATOS["bin_logit"], ref, df, MAPA)
    tab = pd.read_csv(tmp_path / "coef_bin_logit.csv", index_col="termo")
    assert list(tab.index) == a.nomes
    E.ajustar_candidato("bin_logit", df, MAPA)  # sem pasta: não grava


def test_montar_diagnostico_reusa_ajuste_dado(df):
    c = E.CANDIDATOS["bb_logit"]
    a = c.ajustar(df, MAPA)
    m = E.montar_diagnostico(c, a, df, MAPA, seg_ajuste=2.5)
    assert m["aj"] is a and m["res"] is a.modelo and m["seg_ajuste"] == 2.5
    assert np.allclose(m["pi"], c.prever(a, df, MAPA)) and np.allclose(m["theta"], a.extra["theta"])
    m2 = E.montar_diagnostico(c, None, df, MAPA)
    assert np.allclose(m2["pi"], m["pi"]) and m2["seg_ajuste"] >= 0
    y, n = df[MAPA["y"]].to_numpy(float), df[MAPA["n"]].to_numpy(float)
    pi = m["pi"]
    phi = np.sum((y - n * pi) ** 2 / (n * pi * (1 - pi))) / (len(y) - len(a.modelo.params))
    txt = E.descrever_diagnostico("bb_logit", m, y, n)
    assert txt.startswith("bb_logit: ") and f"phi de Pearson = {phi:.1f}" in txt and "2.5 s por ajuste" in txt


def _env_original(nome, B, seed, df):
    """Cópia do `env_modelo` da Parte 10 do `02-MLG` antes da mudança."""
    y, n = df[MAPA["y"]].to_numpy(float), df[MAPA["n"]].to_numpy(float)
    c = E.CANDIDATOS[nome]
    col_y = MAPA["y"]
    estado = {"y": y}

    def ajustar_fn(ys=None):
        estado["y"] = y if ys is None else ys
        return c.ajustar(df if ys is None else df.assign(**{col_y: ys}), MAPA)

    def residuos_fn(res):
        return PR.residuos_quantilicos_rapido(
            estado["y"],
            n,
            c.prever(res, df, MAPA),
            theta=res.extra.get("theta") if c.familia == "betabin" else None,
            rng=0,
        )

    def simular_fn(res, rng):
        pi_ = c.prever(res, df, MAPA)
        if c.familia == "betabin":
            th = res.extra["theta"]
            pi_ = rng.beta(pi_ * th, (1 - pi_) * th)
        return rng.binomial(n.astype(int), pi_).astype(float)

    return M.envelope_meio_normal(residuos_fn, simular_fn, ajustar_fn, B=B, rng=seed)


@pytest.mark.parametrize("nome", ["bin_logit", "bb_logit"])
def test_envelope_do_modelo_igual_ao_original(df, nome):
    c = E.CANDIDATOS[nome]
    novo = PR.envelope_do_modelo(
        c, df, MAPA, lambda a: a.extra.get("theta") if c.familia == "betabin" else None, 3, 2026
    )
    antigo = _env_original(nome, 3, 2026, df)
    for k in antigo:
        assert np.array_equal(novo[k], antigo[k])


def test_envelope_com_theta_por_municipio(df):
    c = E.CANDIDATOS["bb_logit"]
    novo = PR.envelope_do_modelo(c, df, MAPA, lambda a: E.prever_theta(a, df, MAPA), 3, 2026)
    antigo = PR.envelope_do_modelo(c, df, MAPA, lambda a: a.extra["theta"], 3, 2026)
    for k in antigo:
        assert np.array_equal(novo[k], antigo[k])  # theta constante: array e escalar dão os mesmos sorteios
