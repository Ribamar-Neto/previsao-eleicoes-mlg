"""Testes de src/desenho_parcial.py com dados sintéticos pequenos (10 UFs, ~10 municípios cada)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.special import expit, logit

from src import desenho_parcial as DP
from src import espacial as S
from src.especificacoes import CONTINUAS, MAPA_2026, Candidato, Termos

REGIAO_UF = {"AC": "Norte", "AM": "Norte", "BA": "Nordeste", "CE": "Nordeste", "SP": "Sudeste",
             "MG": "Sudeste", "PR": "Sul", "SC": "Sul", "GO": "Centro-Oeste", "MT": "Centro-Oeste"}


def _base(por_uf: dict[str, int], seed: int = 3) -> tuple[pd.DataFrame, SimpleNamespace]:
    rng = np.random.default_rng(seed)
    uf = np.concatenate([[u] * k for u, k in por_uf.items()])
    N = len(uf)
    df = pd.DataFrame({MAPA_2026[k]: rng.normal(size=N) for k in CONTINUAS})
    df[MAPA_2026["pib_pc"]] = np.exp(df[MAPA_2026["pib_pc"]])
    p = expit(0.2 + 0.5 * df[MAPA_2026["lag"]] + 0.6 * df[MAPA_2026["nota_dep"]])
    n = rng.integers(300, 3000, N).astype(float)
    df[MAPA_2026["n"]] = n
    df[MAPA_2026["y"]] = rng.binomial(n.astype(int), rng.beta(p * 300, (1 - p) * 300)).astype(float)
    df[MAPA_2026["uf"]] = uf
    df[MAPA_2026["regiao"]] = [REGIAO_UF[u] for u in uf]
    df[MAPA_2026["bloco_pref"]] = rng.choice(["esquerda", "direita", "centro"], N)
    df["lat"], df["lon"] = rng.uniform(-30, 0, N), rng.uniform(-60, -40, N)
    A = sp.diags([np.ones(N - 1), np.ones(N - 1)], [-1, 1], format="csr")   # cadeia i-1, i+1
    w = SimpleNamespace(n=N, sparse=sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A)
    return df, w


@pytest.fixture(scope="module")
def dados() -> DP.DadosCorte:
    df, w = _base({u: 10 for u in REGIAO_UF})
    return DP.dados_do_corte(df, MAPA_2026, w, df[MAPA_2026["y"]].to_numpy(), df[MAPA_2026["n"]].to_numpy())


# --- candidatas e UFs elegíveis ---------------------------------------------------------------
def test_2022_exclui_delta_comp_e_2026_mantem() -> None:
    assert "delta_comp" not in DP.candidatas_do_ano(2022)
    assert "delta_comp" in DP.candidatas_do_ano(2026)
    assert "lag" not in DP.candidatas_do_ano(2026)


class _Parar(Exception):
    """Interrompe `desenho_no_corte` logo depois da chamada espiada."""


def test_desenho_2022_passa_candidatas_sem_delta_comp(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    vistas: list[tuple[str, ...]] = []

    def espia(D: Any, mapa: Any, cands: Any, *a: Any, **k: Any) -> Termos:
        vistas.append(tuple(cands))
        raise _Parar

    monkeypatch.setattr(DP, "_selecionar_media", espia)
    with pytest.raises(_Parar):
        DP.desenho_no_corte(dados, 2022, n_jobs=1)
    assert vistas
    assert "delta_comp" not in vistas[0]


def test_loso_pula_ufs_com_menos_de_5_municipios() -> None:
    df, w = _base({"AC": 10, "AM": 10, "BA": 10, "CE": 10, "SP": 4, "MG": 10, "PR": 10, "SC": 10})
    D = DP.dados_do_corte(df, MAPA_2026, w, df[MAPA_2026["y"]].to_numpy(), df[MAPA_2026["n"]].to_numpy()).D
    assert "SP" not in DP.ufs_elegiveis(D[MAPA_2026["uf"]].to_numpy())
    cand = Candidato("t", "betabin", "logit", Termos(vars=("lag", "nota_dep")))
    r = DP._loso(cand, D, MAPA_2026, 1)
    assert "SP" not in r.ufs
    assert len(r.ufs) == 7
    assert np.all((r.prev > 0) & (r.prev < 1))


def test_dados_do_corte_so_municipios_apurados() -> None:
    df, w = _base({"AC": 10, "AM": 10})
    n_c = df[MAPA_2026["n"]].to_numpy().copy()
    n_c[:5] = 0.0
    d = DP.dados_do_corte(df, MAPA_2026, w, np.zeros(20), n_c)
    assert len(d.D) == 15
    assert (d.D[MAPA_2026["n"]] > 0).all()


# --- R3 e R7 à mão ----------------------------------------------------------------------------
def test_mae_ep_e_sigma_uf_a_mao() -> None:
    prev = np.array([0.50, 0.60, 0.40])
    obs = np.array([0.48, 0.55, 0.45])
    r = DP.Loso(("A", "B", "C"), prev, obs)
    e = np.array([2.0, 5.0, 5.0])
    assert r.mae_pp == pytest.approx(e.mean())
    assert r.ep_pp == pytest.approx(e.std(ddof=1) / np.sqrt(3))
    esperado = np.sqrt(np.mean((logit(prev) - logit(obs)) ** 2))
    assert DP.sigma_uf(r) == pytest.approx(esperado)
    assert esperado == pytest.approx(0.17348, abs=1e-4)   # diferenças em logit: 0,0800; 0,2048; -0,2048


# --- cache ------------------------------------------------------------------------------------
def _desenho() -> DP.DesenhoParcial:
    t = Termos(vars=("lag", "nota_dep"), disp="regiao", regiao_fundir=(("Nordeste", "Norte"),),
               bloco_grupos=(("esquerda",),))
    return DP.DesenhoParcial(t, 150.0, 0.2, 1.3, {"r1": 1.0, "final": 1.1}, 0.3,
                             {"disp": True, "slx": False}, {"A": 0.001, "B": float("nan")},
                             {"const": 10.0}, 20, 300, 1.5)


def test_cache_ida_e_volta(tmp_path: Path) -> None:
    d = _desenho()
    DP.gravar_cache(d, 2026, 25, tmp_path)
    lido = DP.ler_cache(2026, 25, tmp_path)
    assert lido is not None
    assert lido.termos == d.termos
    assert lido.corte_km == 150.0
    assert lido.n_ufs_loso == 20
    assert lido.aceitos == {"disp": True, "slx": False}
    assert (tmp_path / "desenho_parcial_2026_25.json").exists()


def test_cache_com_versao_antiga_e_ignorado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    DP.gravar_cache(_desenho(), 2022, 5, tmp_path)
    monkeypatch.setattr(DP, "VERSAO", DP.VERSAO + 1)
    assert DP.ler_cache(2022, 5, tmp_path) is None
    assert DP.ler_cache(2022, 90, tmp_path) is None   # ausente


def test_desenhos_por_corte_usa_cache(dados: DP.DadosCorte, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chamadas: list[int] = []

    def falso(d: Any, ano: int, **k: Any) -> DP.DesenhoParcial:
        chamadas.append(ano)
        return _desenho()

    monkeypatch.setattr(DP, "desenho_no_corte", falso)
    DP.desenhos_por_corte({(2026, 10): dados}, pasta=tmp_path)
    DP.desenhos_por_corte({(2026, 10): dados}, pasta=tmp_path)
    assert chamadas == [2026]
    DP.desenhos_por_corte({(2026, 10): dados}, pasta=tmp_path, refazer=True)
    assert chamadas == [2026, 2026]


# --- R1 e decisões R4/R5 ---------------------------------------------------------------------
def test_r1_devolve_termos_com_lag(dados: DP.DadosCorte) -> None:
    t = DP._selecionar_media(dados.D, dados.mapa, ("nota_dep", "pct_dep_esq"), 1, "t", lambda m: None)
    assert isinstance(t, Termos)
    assert t.vars[0] == "lag"
    assert set(t.vars) <= {"lag", "nota_dep", "pct_dep_esq"}


def _loso_falso(mae_por_disp: dict[str, float], mae_slx: float, ufs: int = 10) -> Any:
    def f(cand: Candidato, D: Any, mapa: Any, n_jobs: int) -> DP.Loso:
        m = mae_slx if cand.termos.slx else mae_por_disp[cand.termos.disp]
        obs = np.linspace(0.4, 0.6, ufs)
        err = np.linspace(0.5, 1.5, ufs) / 100          # MAE 1,0 p.p. antes da escala
        err = err / err.mean() * m / 100
        return DP.Loso(tuple(f"U{i}" for i in range(ufs)), obs + err, obs)
    return f


def _preparar(monkeypatch: pytest.MonkeyPatch, *, disp: str, slx_ok: bool, loso: Any) -> None:
    base = Termos(vars=("lag", "nota_dep"))
    monkeypatch.setattr(DP, "_selecionar_media", lambda *a, **k: base)
    monkeypatch.setattr(DP, "_wald_dispersao", lambda *a, **k: (disp, {"A": 0.0, "B": 0.0}, {"const": 1.0}, {"corte_A_km": 1.0, "corte_B_km": 1.0, "piso": False}))
    def slx(b: Termos, d: DP.DadosCorte) -> Any:
        df2, mapa2 = S.adicionar_slx(d.df, d.mapa, ["lag"], d.w)
        return ("lag",), slx_ok, {"nenhuma": 1.0}, DP.DadosCorte(df2, mapa2, d.w, d.ok)

    monkeypatch.setattr(DP, "_escolher_slx", slx)
    monkeypatch.setattr(DP, "_corte_do_desenho", lambda cand, D, mapa: (50.0, False))
    monkeypatch.setattr(DP, "_loso", loso)


def test_dispersao_rejeitada_se_loso_acima_do_limite(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    _preparar(monkeypatch, disp="regiao_logn", slx_ok=True,
              loso=_loso_falso({"const": 1.0, "regiao_logn": 5.0}, 1.0))
    d = DP.desenho_no_corte(dados, 2026, n_jobs=1, candidatas=("nota_dep",))
    assert d.termos.disp == "const"
    assert d.aceitos == {"disp": False, "slx": False}
    assert d.mae_pp["disp"] > d.limite
    assert "slx" not in d.mae_pp


def test_slx_rejeitado_se_loso_acima_do_limite(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    _preparar(monkeypatch, disp="regiao", slx_ok=True,
              loso=_loso_falso({"const": 1.0, "regiao": 1.0}, 9.0))
    d = DP.desenho_no_corte(dados, 2026, n_jobs=1, candidatas=("nota_dep",))
    assert d.aceitos == {"disp": True, "slx": False}
    assert d.termos.disp == "regiao"
    assert d.termos.slx == ()


def test_slx_rejeitado_se_moran_nao_cai(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    _preparar(monkeypatch, disp="const", slx_ok=False,
              loso=_loso_falso({"const": 1.0}, 0.1))
    d = DP.desenho_no_corte(dados, 2026, n_jobs=1, candidatas=("nota_dep",))
    assert d.aceitos["slx"] is False
    assert "slx" not in d.mae_pp


def test_aceitos_usam_loso_do_desenho_final_no_sigma(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _loso_falso({"const": 1.0, "regiao": 1.02}, 1.05)
    _preparar(monkeypatch, disp="regiao", slx_ok=True, loso=f)
    d = DP.desenho_no_corte(dados, 2026, n_jobs=1, candidatas=("nota_dep",))
    assert d.aceitos == {"disp": True, "slx": True}
    assert d.termos.slx == ("lag",)
    assert d.sigma_uf == pytest.approx(DP.sigma_uf(f(Candidato("x", "betabin", "logit", d.termos), None, None, 1)))
    assert d.limite == pytest.approx(d.mae_pp["r1"] + d.ep_r1_pp)
    assert d.n_ufs_loso == 10


# --- ponta a ponta com ajustes reais ------------------------------------------------------
def test_ponta_a_ponta_pequeno(dados: DP.DadosCorte) -> None:
    d = DP.desenho_no_corte(dados, 2026, n_jobs=1, candidatas=("nota_dep", "pct_dep_esq"), log=lambda m: None)
    assert isinstance(d.termos, Termos)
    assert d.termos.vars[0] == "lag"
    assert d.n_ufs_loso == 10
    assert d.n_municipios == 100
    assert d.corte_km >= DP.CORTE_MIN_KM
    assert d.sigma_uf > 0
    assert d.limite > d.mae_pp["r1"]
    assert DP.DesenhoParcial.de_dict(d.para_dict()).termos == d.termos


# --- cortes como no 02 ------------------------------------------------------------------------
def test_wald_usa_o_corte_proprio_de_cada_ajuste(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    cortes = iter([111.0, 222.0])
    usados: list[float] = []
    monkeypatch.setattr(DP, "P_WALD", 2.0)   # p < 2 sempre: passos A e B rodam
    monkeypatch.setattr(DP, "_corte_proprio", lambda res, D: (next(cortes), False))
    cov = S.cov_conley
    monkeypatch.setattr(S, "cov_conley", lambda m, la, lo, c: (usados.append(c), cov(m, la, lo, c))[1])
    forma, _, _, info = DP._wald_dispersao(Termos(vars=("lag", "nota_dep")), dados.D, dados.mapa)
    assert forma == "regiao_logn"
    assert usados == [111.0, 222.0]
    assert info["corte_A_km"] == 111.0
    assert info["corte_B_km"] == 222.0


def test_corte_proprio_aplica_piso_e_avisa(monkeypatch: pytest.MonkeyPatch, dados: DP.DadosCorte) -> None:
    monkeypatch.setattr(S, "corte_de", lambda cg: 0.0)
    assert DP._corte_proprio(np.random.default_rng(0).normal(size=100), dados.D) == (DP.CORTE_MIN_KM, True)
    monkeypatch.setattr(S, "corte_de", lambda cg: 200.0)
    assert DP._corte_proprio(np.random.default_rng(0).normal(size=100), dados.D) == (200.0, False)


def test_corte_final_vem_do_desenho_final_e_r1_fica_como_diagnostico(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    _preparar(monkeypatch, disp="regiao", slx_ok=False, loso=_loso_falso({"const": 1.0, "regiao": 1.0}, 1.0))
    monkeypatch.setattr(DP, "_corte_do_desenho", lambda cand, D, mapa: (222.0, False) if cand.termos.disp == "regiao" else (111.0, True))
    d = DP.desenho_no_corte(dados, 2026, n_jobs=1, candidatas=("nota_dep",))
    assert d.corte_km == 222.0
    assert d.extra["corte_r1_km"] == 111.0
    assert d.extra["piso_aplicado"] == {"r1": True, "final": False, "wald": False}


# --- dobras puladas e cortes que falham -------------------------------------------------------
def test_loso_pula_dobra_singular_e_registra(dados: DP.DadosCorte, monkeypatch: pytest.MonkeyPatch) -> None:
    def dobra(uf: str) -> tuple[float, float]:
        if uf == "AC":
            raise np.linalg.LinAlgError("singular")
        return 0.5, 0.49

    monkeypatch.setattr(DP, "_dobra", dobra)
    r = DP._loso(Candidato("t", "betabin", "logit", Termos(vars=("lag",))), dados.D, dados.mapa, 1)
    assert "AC" not in r.ufs
    assert r.puladas == ("AC",)
    assert len(r.ufs) == 9


def test_desenhos_por_corte_registra_falha_de_linalg(dados: DP.DadosCorte, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def falha(d: Any, ano: int, **k: Any) -> DP.DesenhoParcial:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(DP, "desenho_no_corte", falha)
    falhas: dict[tuple[int, int], str] = {}
    out = DP.desenhos_por_corte({(2026, 5): dados}, pasta=tmp_path, falhas=falhas)
    assert out == {}
    assert "singular" in falhas[(2026, 5)]
    assert DP.ler_cache(2026, 5, tmp_path) is None
