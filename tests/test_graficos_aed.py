"""Fumaça dos gráficos novos de AED e apuração: número de traços esperado."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from src import graficos as G


def test_psi_por_nivel_um_traco_de_ic_e_um_de_referencia() -> None:
    tab = pd.DataFrame({"psi": [1.0, 2.5, 0.7], "psi_inf95": [None, 2.4, 0.6], "psi_sup95": [None, 2.6, 0.8]}, index=["SP", "BA", "RS"])
    fig = G.psi_por_nivel(tab, "t")
    assert isinstance(fig, go.Figure) and len(fig.data) == 2
    assert list(fig.layout.yaxis.categoryarray) == ["RS", "SP", "BA"]


def test_representatividade_por_corte_quatro_tracos_por_ano() -> None:
    linhas = [{"ano": a, "corte": c, "grupo": g, "abs_apurado_pp": 2.0, "abs_modelo_pp": 1.0}
              for a in (2022, 2026) for c in (5, 10) for g in ("maiores", "demais")]
    fig = G.representatividade_por_corte(pd.DataFrame(linhas), "t")
    assert len(fig.data) == 8


def test_faixas_por_decil_isola_o_valor_mais_comum_do_minimo() -> None:
    import numpy as np
    x = np.r_[np.zeros(60), np.arange(1, 41, dtype=float)]  # 60% zeros
    f = G.faixas_por_decil(x)
    assert (f[:60] == 0).all() and f[60:].min() == 1 and f.nunique() == 11
    assert f[60:].value_counts().eq(4).all()


def test_faixas_por_decil_sem_massa_no_minimo_e_decil_simples() -> None:
    import numpy as np
    x = np.r_[np.zeros(5), np.arange(1, 96, dtype=float)]  # 5% zeros: abaixo do limite
    f = G.faixas_por_decil(x)
    assert f.nunique() == 10 and f.value_counts().eq(10).all()


def test_eixo_por_ordem_so_quando_uma_faixa_fica_isolada() -> None:
    assert G.eixo_por_ordem([0.0, 0.001, 0.002, 0.003, 0.005, 0.1])  # % indígenas: um salto domina a amplitude
    assert not G.eixo_por_ordem([0.05, 0.08, 0.1, 0.12, 0.15, 0.23, 0.38])  # % dep. esquerda: salto de 39%
    assert not G.eixo_por_ordem([1.0, 1.0])


def test_painel_linearidade_sem_legenda_e_eixo_em_ordem_para_variavel_assimetrica(monkeypatch: pytest.MonkeyPatch) -> None:
    capturado: dict[str, go.Figure] = {}
    monkeypatch.setattr(G, "mostrar_e_salvar", lambda fig, *a, **k: capturado.setdefault("fig", fig))
    dados = {"assim": pd.DataFrame({"x": [0.0, 0.001, 0.002, 0.1], "y": [0.2, 0.1, 0.0, 0.3]}),
             "linear": pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 0.1, 0.2, 0.3]})}
    G.painel_linearidade(list(dados), dados.__getitem__, "y", "teste", n_cols=2)
    fig = capturado["fig"]
    assert not any(t.showlegend for t in fig.data)
    assert list(fig.data[0].x) == [1, 2, 3, 4] and list(fig.layout.xaxis.ticktext) == ["0", "0,002", "0,1"]
    assert list(fig.data[1].x) == [1.0, 2.0, 3.0, 4.0] and fig.layout.xaxis2.ticktext is None
