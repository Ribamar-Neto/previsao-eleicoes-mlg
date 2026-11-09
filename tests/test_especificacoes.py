import numpy as np
import pandas as pd
import pytest

from src import especificacoes as E

MAPA = E.MAPA_2026


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    N = 500
    d = pd.DataFrame({MAPA[k]: rng.normal(size=N) for k in E.CONTINUAS})
    d[MAPA["pib_pc"]] = np.exp(rng.normal(10, 0.5, N))
    d[MAPA["bloco_pref"]] = rng.choice(["direita", "esquerda", "centro", "sem_prefeito"], N, p=[.6, .3, .06, .04])
    d[MAPA["regiao"]] = rng.choice(["Sudeste", "Sul", "Norte", "Nordeste", "Centro-Oeste"], N)
    d[MAPA["uf"]] = rng.choice(E.UFS, N)
    n = rng.integers(1000, 100000, N)
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * d[MAPA["lag"]] + 0.4 * d[MAPA["nota_dep"]] + rng.normal(0, 0.15, N))))
    d[MAPA["n"]] = n
    d[MAPA["y"]] = rng.binomial(n, p)
    return d


def test_padronizador_so_treino(df):
    tr, te = df.iloc[:400], df.iloc[400:]
    pad = E.Padronizador(MAPA).fit(tr)
    assert abs(pad.z(tr, "nota_dep").mean()) < 1e-10
    assert abs(pad.z(te, "nota_dep").mean()) > 1e-6  # teste não centra a si mesmo


def test_colunas_fixas_com_nivel_ausente(df):
    tr = df[df[MAPA["regiao"]] != "Norte"]
    pad = E.Padronizador(MAPA).fit(tr)
    Xa = E.montar_X(tr, MAPA, pad, E.Termos())
    Xb = E.montar_X(df, MAPA, pad, E.Termos())
    assert list(Xa.columns) == list(Xb.columns)
    assert "reg_Norte" in Xa.columns and Xa["reg_Norte"].sum() == 0


def test_bloco_funde_centro_e_sem_prefeito(df):
    pad = E.Padronizador(MAPA).fit(df)
    X = E.montar_X(df, MAPA, pad)
    b = df[MAPA["bloco_pref"]]
    assert (X["bloco_outro"] == b.isin(["centro", "sem_prefeito"])).all()


def test_offset_e_bruto_na_ultima_coluna(df):
    pad = E.Padronizador(MAPA).fit(df)
    X = E.montar_X(df, MAPA, pad, E.Termos(lag="offset"))
    assert X.columns[-1] == "lag_offset" and "lag" not in X.columns
    np.testing.assert_allclose(X["lag_offset"], df[MAPA["lag"]])


def test_spline_fora_do_suporte_nao_quebra(df):
    pad = E.Padronizador(MAPA).fit(df)
    fora = df.copy()
    fora[MAPA["lag"]] = fora[MAPA["lag"]] * 5
    X = E.montar_X(fora, MAPA, pad, E.Termos(lag="spline"))
    assert np.isfinite(X.to_numpy()).all() and X.filter(like="lag_bs").shape[1] == 4


def test_uf_aleatoria_tem_todos_os_niveis(df):
    pad = E.Padronizador(MAPA).fit(df)
    fixo = E.montar_X(df, MAPA, pad, E.Termos(usa_uf=True))
    alea = E.montar_X(df, MAPA, pad, E.Termos(usa_uf=True, uf_aleatoria=True))
    assert fixo.filter(like="uf_").shape[1] == 26 and alea.filter(like="uf_").shape[1] == 27
    assert not any(c.startswith("reg_") for c in fixo.columns)


def test_chaves_do_registro():
    for k in ["bin_logit", "bin_probit", "bin_cloglog", "bin_loglog", "bin_cauchit", "quasi_logit",
              "bb_logit", "betareg", "ols_elogit", "wls_elogit", "bb_offset", "bb_spline",
              "bb_reglag", "bb_uf", "glmm_uf", "l1_logit"]:
        assert k in E.CANDIDATOS
    assert E.CANDIDATOS["bb_uf"].usa_uf and not E.CANDIDATOS["bb_logit"].usa_uf


@pytest.mark.parametrize("nome", ["bin_logit", "quasi_logit", "bb_logit", "bb_offset", "quasi_offset",
                                  "bb_spline", "betareg", "wls_elogit", "l1_logit", "glmm_uf"])
def test_ajustar_prever(df, nome):
    cand = E.CANDIDATOS[nome]
    tr, te = df.iloc[:400], df.iloc[400:]
    aj = cand.ajustar(tr, MAPA)
    pi = cand.prever(aj, te, MAPA)
    assert pi.shape == (100,) and ((pi > 0) & (pi < 1)).all()
    # sinal do lag no gerador é positivo: o ajuste deve capturar
    assert np.corrcoef(pi, te[MAPA["lag"]])[0, 1] > 0.5


def test_tabela_comparacao(df):
    linhas = {}
    for nome in ["bin_logit", "quasi_logit", "bb_logit"]:
        c = E.CANDIDATOS[nome]
        linhas[nome] = E.resumo_ajuste(c, c.ajustar(df, MAPA), df, MAPA)
    t = E.tabela_comparacao(linhas)
    assert t.loc[t.modelo == "quasi_logit", "aic"].isna().all()
    assert t.loc[t.modelo == "quasi_logit", "qaic"].notna().all()
    assert t.aic.dropna().is_monotonic_increasing


def test_backward_remove_ruido(df):
    vs, trilha = E.selecao_backward_bic(df, MAPA)
    assert "nota_dep" in vs and "lag" in vs and len(vs) < len(E.CONTINUAS)
    assert trilha.bic.is_monotonic_decreasing


# --------------------------------------------------------------------------
# Dispersão e filtro espacial
# --------------------------------------------------------------------------
def test_montar_X_esf_antes_do_offset(df):
    d = df.copy()
    for i in (1, 2):
        d[f"ev_{i:03d}"] = np.linspace(-1, 1, len(d)) * i
    pad = E.Padronizador(MAPA).fit(d)
    X = E.montar_X(d, MAPA, pad, E.Termos(lag="offset", esf=2))
    assert list(X.columns[-3:]) == ["ev_001", "ev_002", E._COL_OFFSET]
    assert np.array_equal(X["ev_002"].to_numpy(), d["ev_002"].to_numpy())  # cru
    with pytest.raises(KeyError):
        E.montar_X(d, MAPA, pad, E.Termos(esf=3))
    assert list(E.montar_X(d, MAPA, pad, E.Termos()).columns) == list(
        E.montar_X(df, MAPA, pad, E.Termos()).columns)


def test_montar_Z(df):
    tr, te = df.iloc[:400], df.iloc[400:]
    pad = E.Padronizador(MAPA).fit(tr)
    assert list(E.montar_Z(te, MAPA, pad, E.Termos()).columns) == ["const"]
    Zr = E.montar_Z(te, MAPA, pad, E.Termos(disp="regiao"))
    assert list(Zr.columns) == ["const"] + [f"reg_{r}" for r in E.REGIOES]
    Zl = E.montar_Z(tr, MAPA, pad, E.Termos(disp="regiao_logn"))
    assert abs(Zl["log_n"].mean()) < 1e-10  # padronizado no treino
    assert abs(E.montar_Z(te, MAPA, pad, E.Termos(disp="regiao_logn"))["log_n"].mean()) > 1e-6
    with pytest.raises(ValueError):
        E.Termos(disp="x")


@pytest.fixture
def backward_tmp(df):
    ant = E.CANDIDATOS.pop("bb_backward", None)
    E.registrar_backward(["nota_dep"])
    yield
    E.EXTENSOES.clear()
    E.CANDIDATOS.pop("bb_backward", None)
    if ant is not None:
        E.CANDIDATOS["bb_backward"] = ant


def test_extensao_prever_theta_e_prever_fold(df, backward_tmp):
    antes = set(E.CANDIDATOS)
    c = E.registrar_extensao("bb_disp_teste", disp="regiao")
    assert "bb_disp_teste" in E.EXTENSOES and set(E.CANDIDATOS) == antes
    a = c.ajustar(df, MAPA)
    assert a.nomes[-1] == "disp_reg_Sul" and a.k == len(a.nomes)
    assert set(a.extra["theta_regiao"]) >= {"Norte", "Sudeste"}
    th = E.prever_theta(a, df, MAPA)
    assert th.shape == (len(df),) and len(np.unique(th.round(8))) == 5
    tab = E.tabela_coeficientes(a)
    assert list(tab.index) == a.nomes
    assert np.isfinite(E.resumo_ajuste(c, a, df, MAPA)["bic"])
    # theta único: array constante
    a0 = E.CANDIDATOS["bb_backward"].ajustar(df, MAPA)
    assert np.allclose(E.prever_theta(a0, df, MAPA), a0.extra["theta"])
    E.definir_dados_cv(df, MAPA)
    pi = E.prever_fold("bb_disp_teste", np.arange(400), np.arange(400, 500))
    assert pi.shape == (100,) and ((pi > 0) & (pi < 1)).all()
    assert E.parametros_ajuste("bb_disp_teste") == a.k


def test_extensao_exige_backward_e_offset_nao_suportado(df):
    ant = E.CANDIDATOS.pop("bb_backward", None)
    try:
        with pytest.raises(KeyError):
            E.registrar_extensao("x", disp="regiao")
    finally:
        if ant is not None:
            E.CANDIDATOS["bb_backward"] = ant
    with pytest.raises(NotImplementedError):
        E.Candidato("t", "betabin", "logit", E.Termos(lag="offset", disp="regiao")).ajustar(df, MAPA)
