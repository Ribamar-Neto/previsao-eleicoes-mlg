"""Testes da seleção aninhada no LOSO e dos testes de fusão de níveis (dados sintéticos)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src import especificacoes as E
from src.especificacoes import CONTINUAS, MAPA_2026, Termos, montar_X
from src.selecao import (
    ajuste_bb,
    analisar_fator,
    fundir,
    mae_uf_pp,
    rodar_folds,
    selecao_aninhada_loso,
)

CAND = ("nota_dep", "pct_dep_esq", "log_pop")
TERMOS = Termos(vars=("lag", *CAND))
UFS_TESTE = ["AC", "AM", "BA", "CE", "SP", "PR", "GO", "MT"]
REGIAO_UF = {
    "AC": "Norte",
    "AM": "Norte",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "SP": "Sudeste",
    "PR": "Sul",
    "GO": "Centro-Oeste",
    "MT": "Centro-Oeste",
}


def _dados(n_mun: int = 700, seed: int = 11, efeito_co: float = 0.5) -> pd.DataFrame:
    """Norte e Nordeste com o mesmo efeito (+0,3, distinto da referência Sudeste); Sul (-0,5) e
    Centro-Oeste (+0,5) diferem; bloco sem efeito. Com a fusão na referência, Norte e Nordeste
    precisam diferir de Sudeste para se fundirem entre si."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({MAPA_2026[k]: rng.normal(size=n_mun) for k in CONTINUAS})
    df[MAPA_2026["pib_pc"]] = np.exp(df[MAPA_2026["pib_pc"]])
    uf = rng.choice(UFS_TESTE, n_mun)
    reg = np.array([REGIAO_UF[u] for u in uf])
    efeito = np.where(
        reg == "Sul",
        -0.5,
        np.where(reg == "Centro-Oeste", efeito_co, np.where(np.isin(reg, ["Norte", "Nordeste"]), 0.3, 0.0)),
    )
    z = {k: df[MAPA_2026[k]].to_numpy() for k in CONTINUAS}
    p = expit(0.2 + 0.5 * z["lag"] + 0.6 * z["nota_dep"] - 0.6 * z["pct_dep_esq"] + efeito)
    n = rng.integers(300, 3000, n_mun).astype(float)
    p = rng.beta(p * 400, (1 - p) * 400)
    df[MAPA_2026["n"]] = n
    df[MAPA_2026["y"]] = rng.binomial(n.astype(int), p).astype(float)
    df[MAPA_2026["uf"]] = uf
    df[MAPA_2026["regiao"]] = reg
    df[MAPA_2026["bloco_pref"]] = rng.choice(["esquerda", "direita", "centro"], n_mun)
    df["aptos_2t_2022"] = n * 1.1
    df["n_2t_2022"] = n
    df["eleitorado_2026"] = n * 1.15
    return df


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _dados()


def test_fundir_soma_colunas() -> None:
    X = pd.DataFrame({"const": [1.0, 1.0], "reg_A": [1.0, 0.0], "reg_B": [0.0, 1.0]})
    Xn = fundir(X, ["reg_A", "reg_B"], "reg_")
    assert list(Xn.columns) == ["const", "reg_A+B"]
    assert Xn["reg_A+B"].tolist() == [1.0, 1.0]


def test_analisar_fator_funde_norte_nordeste_e_mantem_sul(df: pd.DataFrame) -> None:
    y, n = df[MAPA_2026["y"]].to_numpy(float), df[MAPA_2026["n"]].to_numpy(float)
    X = montar_X(df, MAPA_2026, E.Padronizador(MAPA_2026).fit(df), TERMOS)
    cols = [c for c in X.columns if c.startswith("reg_")]
    Xr, res, cols_r = analisar_fator(X, y, n, cols, "Região", "reg_", 0.05, verbose=False)
    assert "reg_Nordeste+Norte" in cols_r
    assert "reg_Sul" in cols_r and "reg_Centro-Oeste" in cols_r
    assert res.llf == pytest.approx(ajuste_bb(Xr, y, n).llf)


def test_analisar_fator_p_alto_nao_funde_se_limite_exige(df: pd.DataFrame) -> None:
    """Com p_fusao = 1 nada passa do limite (p > 1 é impossível): nenhuma fusão."""
    y, n = df[MAPA_2026["y"]].to_numpy(float), df[MAPA_2026["n"]].to_numpy(float)
    X = montar_X(df, MAPA_2026, E.Padronizador(MAPA_2026).fit(df), TERMOS)
    cols = [c for c in X.columns if c.startswith("reg_")]
    _, _, cols_r = analisar_fator(X, y, n, cols, "Região", "reg_", 1.0, verbose=False)
    assert cols_r == cols


def test_fold_aninhado_usa_so_treino_e_replica_convencao_do_loso(df: pd.DataFrame) -> None:
    r = selecao_aninhada_loso(df, MAPA_2026, "BA", candidatas=CAND, n_jobs=2, termos=TERMOS, log=lambda _: None)
    assert r["uf"] == "BA"
    assert {"nota_dep", "pct_dep_esq"} <= set(r["variaveis"].split(","))
    assert r["fusao_regiao"] == "Nordeste+Norte"
    # o bloco não tem efeito nos dados sintéticos: todos os níveis vão para a referência
    assert r["fusao_bloco"] == ""
    # previsão idêntica à de prever_fold com a mesma especificação
    vs = ("lag", *r["variaveis"].split(","))
    E.definir_dados_cv(df, MAPA_2026)
    E.EXTENSOES["_t"] = E.Candidato(
        "_t", "betabin", "logit", Termos(vars=vs, regiao_fundir=(("Nordeste", "Norte"),), bloco_grupos=())
    )
    try:
        uf = df[MAPA_2026["uf"]].to_numpy()
        pi = E.prever_fold("_t", np.flatnonzero(uf != "BA"), np.flatnonzero(uf == "BA"))
    finally:
        E.EXTENSOES.pop("_t")
    nh = E._DADOS_CV["n_hat"][uf == "BA"]
    assert r["prev_pct"] == pytest.approx(float((pi * nh).sum() / nh.sum()) * 100)
    assert r["erro_abs_pp"] == pytest.approx(abs(r["prev_pct"] - r["obs_pct"]))


def test_driver_grava_status_e_e_retomavel(df: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.selecao as S

    monkeypatch.setattr(S, "CANDIDATAS_PADRAO", CAND)
    orig = S.selecao_aninhada_loso

    def rapido(*a: object, **k: object) -> dict[str, object]:
        return orig(*a, **{**k, "candidatas": CAND, "termos": TERMOS})  # type: ignore[arg-type]

    monkeypatch.setattr(S, "selecao_aninhada_loso", rapido)
    arq, st = tmp_path / "a.parquet", tmp_path / "status.txt"
    r1 = rodar_folds(["BA"], "t", st, arq=arq, n_jobs=1, df=df)
    r2 = rodar_folds(["BA", "CE"], "t", st, arq=arq, n_jobs=1, df=df)
    assert list(r1["uf"]) == ["BA"] and sorted(r2["uf"]) == ["BA", "CE"]
    assert pd.read_parquet(arq)["uf"].is_unique
    linhas = st.read_text().splitlines()
    assert linhas[-1] == "DONE"
    assert any(l.startswith("t BA OK ja_pronto") for l in linhas)
    assert any(l.startswith("t CE OK") for l in linhas)
    assert mae_uf_pp(r2) == pytest.approx(r2["erro_abs_pp"].mean())


def _fator(seed: int = 5, n_mun: int = 1500) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fator A/B/C/D (ref A): B e C com o mesmo efeito (+0,6), D sem efeito, sobre beta-binomial."""
    rng = np.random.default_rng(seed)
    nivel = rng.choice(["A", "B", "C", "D"], n_mun)
    ef = {"A": 0.0, "B": 0.6, "C": 0.6, "D": 0.0}
    p = expit(0.1 + np.array([ef[v] for v in nivel]))
    n = rng.integers(500, 3000, n_mun).astype(float)
    p = rng.beta(p * 300, (1 - p) * 300)
    y = rng.binomial(n.astype(int), p).astype(float)
    X = pd.DataFrame({"const": 1.0, **{f"f_{k}": (nivel == k).astype(float) for k in "BCD"}})
    return X, y, n


def test_analisar_fator_nivel_sem_efeito_funde_na_referencia() -> None:
    X, y, n = _fator()
    Xr, _, cols = analisar_fator(X, y, n, ["f_B", "f_C", "f_D"], "F", "f_", 0.05, verbose=False)
    assert "f_D" not in Xr.columns and "f_D" not in cols
    assert "f_B+C" in cols


def test_analisar_fator_sem_fundir_na_referencia_mantem_nivel() -> None:
    X, y, n = _fator()
    _, _, cols = analisar_fator(
        X, y, n, ["f_B", "f_C", "f_D"], "F", "f_", 0.05, verbose=False, fundir_na_referencia=False
    )
    assert "f_D" in cols and "f_B+C" in cols


def test_analisar_fator_dois_niveis_iguais_fundem_em_par() -> None:
    x, y, n = _fator()
    x = x.drop(columns="f_D")
    _, _, cols = analisar_fator(x, y, n, ["f_B", "f_C"], "F", "f_", 0.05, verbose=False)
    assert cols == ["f_B+C"]


def test_analisar_fator_imprime_candidatos_na_referencia(capsys: pytest.CaptureFixture[str]) -> None:
    X, y, n = _fator()
    analisar_fator(X, y, n, ["f_B", "f_C", "f_D"], "F", "f_", 0.05, ref="A")
    assert "A" in capsys.readouterr().out


def test_fusoes_bloco_parte_dos_niveis_brutos(df: pd.DataFrame) -> None:
    from src.selecao import _fusoes

    grupos_reg, grupos_blo = _fusoes(df, MAPA_2026, TERMOS, 0.05)
    assert grupos_reg == (("Nordeste", "Norte"),)
    assert grupos_blo == ()  # bloco sem efeito: tudo vai à referência


def test_fusoes_regiao_sem_efeito_vai_para_a_referencia() -> None:
    """Centro-Oeste sem efeito (igual a Sudeste): o grupo de fusão inclui a referência."""
    from src.selecao import _fusoes

    grupos_reg, _ = _fusoes(_dados(efeito_co=0.0), MAPA_2026, TERMOS, 0.05)
    assert ("Nordeste", "Norte") in grupos_reg
    assert ("Centro-Oeste", "Sudeste") in grupos_reg


def test_analisar_fator_passos_p_igual_ao_lrt_feito_a_mao() -> None:
    from scipy import stats

    x, y, n = _fator()
    x = x.drop(columns="f_D")
    passos: list[pd.DataFrame] = []
    _, _, cols = analisar_fator(x, y, n, ["f_B", "f_C"], "F", "f_", 0.05, verbose=False, passos=passos)
    assert cols == ["f_B+C"]
    assert len(passos) == 2  # passo 0 funde B e C; passo 1 só tem f_B+C
    t0 = passos[0]
    linha = t0[(t0.nivel_a == "B") & (t0.nivel_b == "C")].iloc[0]
    cheio = ajuste_bb(x, y, n).llf
    delta = max(2 * (cheio - ajuste_bb(fundir(x, ["f_B", "f_C"], "f_"), y, n).llf), 0.0)
    assert linha["delta"] == pytest.approx(delta)
    assert linha["gl"] == 1
    assert linha["p"] == pytest.approx(stats.chi2.sf(delta, 1))
    assert t0["fundido"].sum() == 1
    assert bool(linha["fundido"])
    assert set(t0["passo"]) == {0}
    assert not passos[1]["fundido"].any()


def test_analisar_fator_rotulo_da_referencia_acumula_niveis_fundidos() -> None:
    x, y, n = _fator()
    passos: list[pd.DataFrame] = []
    analisar_fator(x, y, n, ["f_B", "f_C", "f_D"], "F", "f_", 0.05, verbose=False, ref="A", passos=passos)
    fundidos_na_ref = [p[p.fundido & p.nivel_b.str.startswith("A")] for p in passos]
    primeiro = next(i for i, f in enumerate(fundidos_na_ref) if not f.empty)
    nivel = fundidos_na_ref[primeiro].nivel_a.iloc[0]
    seguinte = passos[primeiro + 1]
    rotulos = set(seguinte.loc[seguinte.nivel_b.str.startswith("A"), "nivel_b"])
    assert rotulos == {f"A+{nivel}"}
