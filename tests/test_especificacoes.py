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
    d[MAPA["bloco_pref"]] = rng.choice(["direita", "esquerda", "centro", "sem_prefeito"], N, p=[0.6, 0.3, 0.06, 0.04])
    d[MAPA["regiao"]] = rng.choice(["Sudeste", "Sul", "Norte", "Nordeste", "Centro-Oeste"], N)
    d[MAPA["uf"]] = rng.choice(E.UFS, N)
    n = rng.integers(1000, 100000, N)
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * d[MAPA["lag"]] + 0.4 * d[MAPA["nota_dep"]] + rng.normal(0, 0.15, N))))
    d[MAPA["n"]] = n
    d["n_2t_2022"], d["aptos_2t_2022"] = n * 0.9, n * 1.2
    d["eleitorado_2026"] = n * 1.25
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
    for k in [
        "bin_logit",
        "bin_probit",
        "bin_cloglog",
        "bin_loglog",
        "bin_cauchit",
        "quasi_logit",
        "bb_logit",
        "betareg",
        "ols_elogit",
        "wls_elogit",
        "bb_offset",
        "bb_spline",
        "bb_reglag",
        "bb_uf",
        "glmm_uf",
        "l1_logit",
    ]:
        assert k in E.CANDIDATOS
    assert E.CANDIDATOS["bb_uf"].usa_uf and not E.CANDIDATOS["bb_logit"].usa_uf


@pytest.mark.parametrize(
    "nome",
    [
        "bin_logit",
        "quasi_logit",
        "bb_logit",
        "bb_offset",
        "quasi_offset",
        "bb_spline",
        "betareg",
        "wls_elogit",
        "l1_logit",
        "glmm_uf",
    ],
)
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


def test_regiao_fundir_une_dummies_e_valida(df):
    pad = E.Padronizador(MAPA).fit(df)
    X0 = E.montar_X(df, MAPA, pad, E.Termos())
    X = E.montar_X(df, MAPA, pad, E.Termos(regiao_fundir=(("Nordeste", "Norte"),)))
    assert "reg_Nordeste" not in X and "reg_Norte" not in X
    assert np.array_equal(X["reg_Nordeste+Norte"], X0["reg_Nordeste"] + X0["reg_Norte"])
    assert X.shape[1] == X0.shape[1] - 1
    with pytest.raises(ValueError):
        E.Termos(regiao_fundir=(("Marte", "Sul"),))
    with pytest.raises(ValueError):
        E.Termos(regiao_fundir=(("Sul", "Norte"), ("Norte", "Sudeste")))
    with pytest.raises(ValueError):
        E.Termos(regiao_fundir=(("Sul", "Norte"),), regiao_x_lag=True)


def test_regiao_fundir_na_referencia_retira_as_dummies(df):
    pad = E.Padronizador(MAPA).fit(df)
    X0 = E.montar_X(df, MAPA, pad, E.Termos())
    X = E.montar_X(df, MAPA, pad, E.Termos(regiao_fundir=(("Norte", "Sudeste"), ("Nordeste", "Sul"))))
    assert "reg_Norte" not in X and not any("Sudeste" in c for c in X.columns)
    assert np.array_equal(X["reg_Nordeste+Sul"], X0["reg_Nordeste"] + X0["reg_Sul"])
    assert X.shape[1] == X0.shape[1] - 2


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
    assert list(E.montar_X(d, MAPA, pad, E.Termos()).columns) == list(E.montar_X(df, MAPA, pad, E.Termos()).columns)


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
def selecao_tmp(df):
    ant = E.CANDIDATOS.pop("bb_selecao", None)
    E.registrar_selecao(["nota_dep"])
    yield
    E.EXTENSOES.clear()
    E.CANDIDATOS.pop("bb_selecao", None)
    if ant is not None:
        E.CANDIDATOS["bb_selecao"] = ant


def test_extensao_prever_theta_e_prever_fold(df, selecao_tmp):
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
    a0 = E.CANDIDATOS["bb_selecao"].ajustar(df, MAPA)
    assert np.allclose(E.prever_theta(a0, df, MAPA), a0.extra["theta"])
    E.definir_dados_cv(df, MAPA)
    pi = E.prever_fold("bb_disp_teste", np.arange(400), np.arange(400, 500))
    assert pi.shape == (100,) and ((pi > 0) & (pi < 1)).all()
    assert E.parametros_ajuste("bb_disp_teste") == a.k


def test_extensao_exige_selecao_e_offset_nao_suportado(df):
    ant = E.CANDIDATOS.pop("bb_selecao", None)
    try:
        with pytest.raises(KeyError):
            E.registrar_extensao("x", disp="regiao")
    finally:
        if ant is not None:
            E.CANDIDATOS["bb_selecao"] = ant
    with pytest.raises(NotImplementedError):
        E.Candidato("t", "betabin", "logit", E.Termos(lag="offset", disp="regiao")).ajustar(df, MAPA)


def test_slx_vazio_nao_altera_desenho(df):
    mapa2 = {**MAPA, "w_lag": "w_col"}
    d2 = df.assign(w_col=np.random.default_rng(5).normal(size=len(df)))
    E.registrar_selecao(["nota_dep", "pct_dep_esq"])
    E.registrar_extensao("bb_disp", disp="regiao")
    for cand in (E.CANDIDATOS["bb_selecao"], E.EXTENSOES["bb_disp"]):
        a = E.montar_X(df, MAPA, E.Padronizador(MAPA).fit(df), cand.termos)
        b = E.montar_X(d2, mapa2, E.Padronizador(mapa2).fit(d2), cand.termos)
        pd.testing.assert_frame_equal(a, b)


def test_slx_coluna_w_lag_padronizada_no_treino(df):
    d = df.assign(w_col=np.random.default_rng(5).normal(2, 3, len(df)))
    mapa = {**MAPA, "w_lag": "w_col"}
    tr, te = d.iloc[:400], d.iloc[400:]
    pad = E.Padronizador(mapa).fit(tr)
    t = E.Termos(slx=("lag",))
    assert "w_lag" in t.vars and E.Termos(**t.__dict__).vars.count("w_lag") == 1
    X = E.montar_X(te, mapa, pad, t)
    esperado = (te["w_col"] - tr["w_col"].mean()) / tr["w_col"].std(ddof=0)
    np.testing.assert_allclose(X["w_lag"].to_numpy(), esperado.to_numpy())
    assert abs(E.montar_X(tr, mapa, pad, t)["w_lag"].mean()) < 1e-10


def test_mapa_padrao_sem_covariavel_do_2t_2026():
    proibidas = {k: v for k, v in E.MAPA_2026.items() if "2t_2026" in v and k not in ("y", "n")}
    assert proibidas == {}
    assert E.MAPA_2026_PRE is E.MAPA_2026
    E.validar_mapa_sem_2t_2026(E.MAPA_2026)
    with pytest.raises(ValueError):
        E.validar_mapa_sem_2t_2026({**E.MAPA_2026, "comparecimento": "taxa_comparecimento_2t_2026"})


def test_n_loso_formula():
    d = pd.DataFrame({"eleitorado_2026": [1000.0, 200.0], "n_2t_2022": [700.0, 90.0], "aptos_2t_2022": [1000.0, 100.0]})
    np.testing.assert_allclose(E.n_loso(d), [700.0, 180.0])


def test_prever_fold_usa_n_estimado_no_teste(monkeypatch):
    n_col = E.MAPA_2026["n"]
    d = pd.DataFrame(
        {
            n_col: [10.0, 20.0, 30.0],
            "eleitorado_2026": [100.0, 200.0, 300.0],
            "n_2t_2022": [5.0, 5.0, 5.0],
            "aptos_2t_2022": [10.0, 10.0, 10.0],
        }
    )
    vistos: dict = {}

    class Falso:
        def ajustar(self, df, mapa):
            vistos["treino"] = df[mapa["n"]].to_numpy()

        def prever(self, ajuste, df, mapa):
            vistos["teste"] = df[mapa["n"]].to_numpy()
            return np.full(len(df), 0.5)

    monkeypatch.setitem(E.EXTENSOES, "falso", Falso())
    E.definir_dados_cv(d)
    E.prever_fold("falso", np.array([0, 1]), np.array([2]))
    np.testing.assert_allclose(vistos["treino"], [10.0, 20.0])
    np.testing.assert_allclose(vistos["teste"], [150.0])
    assert d[n_col].tolist() == [10.0, 20.0, 30.0]


def test_mapas_do_backtest_inalterados():
    from src import backtest as B

    assert B.MAPA_FIT_2022["comparecimento"] == "taxa_comparecimento_2t_2018"
    assert B.MAPA_FIT_2022["delta_comp"] == "delta_comparecimento_2t_2022"
    assert B.MAPA_PREV_2026["comparecimento"] == "taxa_comparecimento_2t_2022"
    assert B.MAPA_PREV_2026["delta_comp"] == "delta_comparecimento_2t_2022"


def _montar_X_antigo(df, pad):
    """Codificação anterior do bloco (3 linhas), para o teste de regressão."""
    X = E.montar_X(df, MAPA, pad, E.Termos(usa_bloco=False))
    b = df[MAPA["bloco_pref"]].astype(str)
    X.insert(X.columns.get_loc("reg_Centro-Oeste"), "bloco_esquerda", (b == "esquerda").astype(float).to_numpy())
    X.insert(
        X.columns.get_loc("reg_Centro-Oeste"),
        "bloco_outro",
        (~b.isin(["esquerda", "direita"])).astype(float).to_numpy(),
    )
    return X


def test_montar_X_padrao_identico_a_codificacao_antiga(df):
    pad = E.Padronizador(MAPA).fit(df)
    novo = E.montar_X(df, MAPA, pad, E.Termos())
    antigo = _montar_X_antigo(df, pad)
    assert list(novo.columns) == list(antigo.columns)
    pd.testing.assert_frame_equal(novo, antigo)


def test_bloco_grupos_fundidos_antigo_bloco_fundir(df):
    pad = E.Padronizador(MAPA).fit(df)
    X = E.montar_X(df, MAPA, pad, E.Termos(bloco_grupos=(("esquerda", "centro", "sem_prefeito"),)))
    b = df[MAPA["bloco_pref"]]
    assert "bloco_esquerda+outro" in X.columns and "bloco_outro" not in X.columns
    assert (X["bloco_esquerda+outro"] == (b != "direita")).all()


def test_bloco_grupos_niveis_brutos(df):
    pad = E.Padronizador(MAPA).fit(df)
    X = E.montar_X(df, MAPA, pad, E.Termos(bloco_grupos=(("esquerda",), ("centro",), ("sem_prefeito",))))
    b = df[MAPA["bloco_pref"]]
    assert (X["bloco_centro"] == (b == "centro")).all()
    assert (X["bloco_sem_prefeito"] == (b == "sem_prefeito")).all()


def test_bloco_grupos_nivel_na_referencia_e_ausente(df):
    pad = E.Padronizador(MAPA).fit(df)
    X = E.montar_X(df, MAPA, pad, E.Termos(bloco_grupos=(("esquerda",),)))
    assert [c for c in X.columns if c.startswith("bloco_")] == ["bloco_esquerda"]
    X2 = E.montar_X(df, MAPA, pad, E.Termos(bloco_grupos=(("esquerda",), ("direita", "centro"))))
    assert [c for c in X2.columns if c.startswith("bloco_")] == ["bloco_esquerda"]


def test_bloco_valor_inesperado_vai_ao_grupo_de_sem_prefeito(df):
    pad = E.Padronizador(MAPA).fit(df)
    d = df.copy()
    d[MAPA["bloco_pref"]] = d[MAPA["bloco_pref"]].astype(object)
    d.iloc[0, d.columns.get_loc(MAPA["bloco_pref"])] = None
    X = E.montar_X(d, MAPA, pad, E.Termos())
    assert X["bloco_outro"].iloc[0] == 1.0


@pytest.mark.parametrize("grupos", [(("xx",),), (("esquerda",), ("esquerda", "centro")), (("esquerda", "esquerda"),)])
def test_bloco_grupos_invalido(grupos):
    with pytest.raises(ValueError):
        E.Termos(bloco_grupos=grupos)


def test_usa_suplementar_adiciona_coluna(df):
    pad = E.Padronizador(MAPA).fit(df)
    d = df.copy()
    d[MAPA["suplementar"]] = pd.array([True, False, None] + [False] * (len(d) - 3), dtype=object)
    X = E.montar_X(d, MAPA, pad, E.Termos(usa_suplementar=True))
    assert X["prefeito_suplementar"].iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert "prefeito_suplementar" not in E.montar_X(d, MAPA, pad, E.Termos()).columns


def test_registrar_selecao_leva_o_bloco_fundido_e_as_extensoes_herdam(df: pd.DataFrame) -> None:
    ant = E.CANDIDATOS.pop("bb_selecao", None)
    try:
        bloco = (("esquerda",),)
        cand = E.registrar_selecao(["nota_dep"], (("Nordeste", "Norte"),), bloco_grupos=bloco)
        assert cand.termos.bloco_grupos == bloco
        pad = E.Padronizador(MAPA).fit(df)
        X = E.montar_X(df, MAPA, pad, cand.termos)
        assert "bloco_esquerda" in X.columns
        assert not any(c.startswith("bloco_") and c != "bloco_esquerda" for c in X.columns)
        assert "bloco_outro" not in X.columns
        ext = E.registrar_extensao("bb_disp_bloco", disp="regiao")
        assert ext.termos.bloco_grupos == bloco
        # sem o parâmetro, vale o padrão histórico
        assert E.registrar_selecao(["nota_dep"]).termos.bloco_grupos == E.BLOCOS_PADRAO
    finally:
        E.EXTENSOES.clear()
        E.CANDIDATOS.pop("bb_selecao", None)
        if ant is not None:
            E.CANDIDATOS["bb_selecao"] = ant


def test_covariaveis_do_1t_do_proprio_ciclo() -> None:
    """lag1t e outros_1t são candidatas e leem só colunas do 1º turno (nada do 2º turno de 2026)."""
    from src import backtest as B

    assert {"lag1t", "outros_1t"} <= set(E.CONTINUAS)
    for mapa in (E.MAPA_2026, B.MAPA_FIT_2022, B.MAPA_PREV_2026, *B.mapas_entre_turnos()):
        for k in ("lag1t", "outros_1t"):
            assert "_1t_" in mapa[k] and "2t_2026" not in mapa[k]
    assert E.MAPA_2026["lag1t"] == "logit_1t_2026"
    assert B.MAPA_FIT_2022["lag1t"] == "logit_1t_2022" and B.MAPA_FIT_2022["outros_1t"] == "outros_1t_2022"
