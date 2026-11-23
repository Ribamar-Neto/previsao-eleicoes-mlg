"""Funções de gráfico (Plotly) dos notebooks do projeto.

Chame configurar(df, pasta_figuras) antes de usar mostrar_e_salvar e as funções de painel.
Mapas: matplotlib não está no ambiente, então os mapas são coropléticos Plotly com a
geometria simplificada embutida (um GeoJSON por figura).
Os gráficos de diagnóstico do MLG (Parte 10 em diante) ficam na segunda metade do módulo.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

COR_PONTO = "#2a78d6"
COR_CURVA = "#eb6834"
COR_REFERENCIA = "#898781"
COR_CRITICO = "#d03b3b"
ESCALA_DIVERGENTE = [[0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec835a"], [1, "#a32a2a"]]
CORES_REGIAO = {"Norte": "#1baf7a", "Nordeste": "#eb6834", "Centro-Oeste": "#eda100",
                "Sudeste": "#2a78d6", "Sul": "#4a3aa7"}

pio.templates["eleicoes"] = go.layout.Template(layout={
    "colorway": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "font": {"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": "#0b0b0b"},
    "paper_bgcolor": "#fcfcfb", "plot_bgcolor": "#fcfcfb", "hovermode": "closest",
    "hoverlabel": {"bgcolor": "#ffffff", "font_size": 12},
})
pio.templates.default = "plotly_white+eleicoes"

df: pd.DataFrame = pd.DataFrame()
output_fig_dir: str | Path | None = None


def configurar(dados: pd.DataFrame, pasta_figuras: str | Path) -> None:
    # Conjunto de dados (hover) e pasta onde as figuras HTML são salvas
    global df, output_fig_dir
    df = dados
    output_fig_dir = pasta_figuras
    os.makedirs(pasta_figuras, exist_ok=True)


def texto_hover(indices: Sequence[int], extras: Mapping[str, Any] | None = None) -> list[str]:
    indices = list(indices)
    extras = {k: np.asarray(v) for k, v in (extras or {}).items()}
    textos = []
    for k, i in enumerate(indices):
        partes = [f"<b>{df.at[i, 'nome_municipio']}</b> ({df.at[i, 'sigla_uf']})"]
        partes += [f"{nome}: {v[k]:.3f}" for nome, v in extras.items()]
        textos.append("<br>".join(partes))
    return textos


def pontos(x: Any, y: Any, indices: Sequence[int], extras: Mapping[str, Any] | None = None,
           nome: str = "Municípios", cor: str = COR_PONTO, tamanho: int = 5, **kwargs: Any) -> go.Scatter:
    return go.Scatter(
        x=np.asarray(x), y=np.asarray(y), mode="markers", name=nome,
        marker={"color": cor, "size": tamanho, "opacity": 0.5},
        hovertext=texto_hover(indices, extras), hovertemplate="%{hovertext}<extra></extra>", **kwargs)


def curva_lowess(x: Any, y: Any, nome: str = "LOWESS", cor: str = COR_CURVA, frac: float = 2 / 3,
                 it: int = 0) -> go.Scatter:
    s = lowess(np.asarray(y, float), np.asarray(x, float), frac=frac, it=it)
    return go.Scatter(x=s[:, 0], y=s[:, 1], mode="lines", name=nome, line={"color": cor, "width": 2.5},
                      hovertemplate=f"{nome}<br>x = %{{x:.3f}}<br>y = %{{y:.3f}}<extra></extra>")


def linha_ref(fig: go.Figure, valor: float, inicio: float, fim: float, nome: str, cor: str = COR_REFERENCIA,
              vertical: bool = False, traco: str = "dash", row: int | None = None,
              col: int | None = None) -> None:
    t = np.linspace(inicio, fim, 40)
    c = np.full_like(t, valor)
    x, y = (c, t) if vertical else (t, c)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nome, line={"color": cor, "width": 1.5, "dash": traco},
                             hovertemplate=f"{nome}: {valor:.4g}<extra></extra>"), row=row, col=col)


def legenda_unica(fig: go.Figure) -> None:
    vistos: set[str] = set()
    for traco in fig.data:
        if traco.name is None:
            continue
        traco.legendgroup = traco.name
        traco.showlegend = traco.name not in vistos and traco.showlegend is not False
        if traco.showlegend:
            vistos.add(traco.name)


def mostrar_e_salvar(fig: go.Figure, nome: str | None = None, titulo: str | None = None, altura: int = 500,
                     largura: int | None = None) -> None:
    legenda_unica(fig)
    fig.update_layout(title=titulo, height=altura, width=largura, margin={"t": 90 if titulo else 60})
    if nome:
        fig.write_html(f"{output_fig_dir}/{nome}.html", include_plotlyjs="cdn", div_id=nome)
    fig.show()


def histograma(valores: Any, nome_var: str, pesos: Any = None, name: str | None = None,
               titulo: str | None = None, bins: int = 50) -> None:
    # Histograma simples (frequência ou soma de pesos por classe)
    v = np.asarray(valores, float)
    fig = go.Figure(go.Histogram(
        x=v, y=None if pesos is None else np.asarray(pesos, float), histfunc="count" if pesos is None else "sum",
        nbinsx=bins, marker={"color": COR_PONTO, "line": {"width": 0.5, "color": "#fcfcfb"}},
        hovertemplate=f"{nome_var}: %{{x}}<br>%{{y:,.0f}}<extra></extra>"))
    fig.update_xaxes(title_text=nome_var)
    fig.update_yaxes(title_text="Municípios" if pesos is None else "Votos")
    fig.update_layout(showlegend=False)
    mostrar_e_salvar(fig, name, titulo=titulo, altura=380)


def heatmap_correlacao(matriz: pd.DataFrame, titulo: str, name: str) -> None:
    n = len(matriz)
    fig = go.Figure(go.Heatmap(
        z=matriz.values, x=list(matriz.columns), y=list(matriz.index), zmin=-1, zmax=1, zmid=0,
        colorscale=ESCALA_DIVERGENTE, text=matriz.round(2).values, texttemplate="%{text:.2f}", textfont={"size": 8},
        xgap=1, ygap=1, colorbar={"title": "ρ"}, hovertemplate="%{y} x %{x}<br>correlação = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    mostrar_e_salvar(fig, name, titulo=titulo, altura=max(600, 28 * n + 200), largura=max(800, 28 * n + 250))


def logit_empirico_por_decil(x: Any, y: Any, n: Any, q: int = 10) -> pd.DataFrame:
    # Por decil de x: média de x e log(ΣY / Σ(n−Y)). Pesos naturais: municípios grandes pesam mais.
    d = pd.DataFrame({"x": x, "y": y, "n": n})
    d["bin"] = pd.qcut(d["x"].rank(method="first"), q, labels=False)
    g = d.groupby("bin").agg(x=("x", "mean"), y=("y", "sum"), n=("n", "sum"))
    g["logit"] = np.log(g["y"] / (g["n"] - g["y"]))
    return g


def media_ponderada_por_decil(x: Any, z: Any, w: Any, q: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"x": x, "z": z, "w": w})
    d["bin"] = pd.qcut(d["x"].rank(method="first"), q, labels=False)
    return d.groupby("bin").apply(lambda g: pd.Series({"x": g["x"].mean(), "z": np.average(g["z"], weights=g["w"])}),
                                  include_groups=False)


def painel_linearidade(variaveis: Sequence[str], calcula: Callable[[str], pd.DataFrame], rotulo_y: str, name: str,
                       n_cols: int = 4, titulo: str | None = None) -> None:
    # calcula(var) -> DataFrame com colunas x e y (um ponto por decil)
    n_rows = math.ceil(len(variaveis) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=variaveis, vertical_spacing=0.06,
                        horizontal_spacing=0.06)
    for k, v in enumerate(variaveis):
        r, c = k // n_cols + 1, k % n_cols + 1
        g = calcula(v)
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers+lines", name="Decil",
                                 marker={"color": COR_PONTO, "size": 7}, line={"color": COR_PONTO, "width": 1.2},
                                 hovertemplate=f"{v}: %{{x:.3f}}<br>{rotulo_y}: %{{y:.3f}}<extra></extra>"), row=r, col=c)
    fig.update_yaxes(title_text=rotulo_y, col=1, title_font={"size": 10})
    mostrar_e_salvar(fig, name, titulo=titulo, altura=230 * n_rows + 80)


# ---------- espacial ----------

def mapa(geo_simpl: Any, valores: Any, titulo: str, name: str, rotulo: str,
         escala: Sequence[Sequence[Any]] = ESCALA_DIVERGENTE, centro: float | None = None, zmin: float | None = None,
         zmax: float | None = None, hover: Sequence[str] | None = None,
         ticks: tuple[Sequence[float], Sequence[str]] | None = None) -> None:
    # Mapa coroplético; geo_simpl com índice RangeIndex alinhado a `valores`; geometria arredondada para reduzir tamanho
    feats = []
    for i, g in enumerate(geo_simpl.geometry):
        feats.append({"type": "Feature", "id": i, "properties": {}, "geometry": _arredonda(g.__geo_interface__)})
    gj = {"type": "FeatureCollection", "features": feats}
    kw: dict[str, Any] = {"zmid": centro} if centro is not None else {}
    if zmin is not None:
        kw.update(zmin=zmin, zmax=zmax)
    fig = go.Figure(go.Choropleth(
        geojson=gj, locations=list(range(len(geo_simpl))), z=np.asarray(valores, float), colorscale=escala,
        marker_line_width=0, colorbar=dict(title=rotulo, **({} if ticks is None else {"tickvals": ticks[0], "ticktext": ticks[1]})), text=hover,
        hovertemplate="%{text}<br>" + rotulo + ": %{z:.3f}<extra></extra>", **kw))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#fcfcfb")
    mostrar_e_salvar(fig, name, titulo=titulo, altura=620, largura=700)


def _arredonda(gi: Mapping[str, Any], casas: int = 2) -> dict[str, Any]:
    def r(c: Any) -> list[Any]:
        if isinstance(c[0], (int, float)):
            return [round(c[0], casas), round(c[1], casas)]
        return [r(x) for x in c]
    return {"type": gi["type"], "coordinates": r(gi["coordinates"])}


# ---------- diagnóstico do MLG (notebook 02 em diante) ----------
# Salvos em results/figures/mlg_*.html.

def salvar(fig: go.Figure, caminho: str | Path) -> go.Figure:
    fig.write_html(str(caminho), include_plotlyjs="cdn", div_id=Path(caminho).stem)
    return fig


def qq_normal(residuos: Mapping[str, Any], titulo: str = "QQ-plot dos resíduos quantílicos") -> go.Figure:
    """Uma curva por modelo; sob o modelo correto os pontos ficam sobre a diagonal."""
    fig = go.Figure()
    for nome, r in residuos.items():
        r = np.sort(np.asarray(r, float))
        q = stats.norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
        passo = max(1, len(r) // 1500)
        fig.add_scatter(x=q[::passo], y=r[::passo], mode="markers", name=nome, marker={"size": 4})
    lim = [-4, 4]
    fig.add_scatter(x=lim, y=lim, mode="lines", line={"color": "gray", "dash": "dash"}, showlegend=False)
    fig.update_layout(title=titulo, xaxis_title="quantil normal teórico",
                      yaxis_title="resíduo quantílico", template="plotly_white")
    return fig


def forest_coeficientes(tabelas: Mapping[str, pd.DataFrame], variaveis: Sequence[str] | None = None,
                        titulo: str = "Coeficientes (IC 95%)") -> go.Figure:
    """Coeficientes padronizados com IC 95% lado a lado por modelo."""
    fig = go.Figure()
    for nome, t in tabelas.items():
        t = t if variaveis is None else t.loc[[v for v in variaveis if v in t.index]]
        t = t[t.index != "const"]
        fig.add_scatter(x=t["coef"], y=list(t.index), mode="markers", name=nome,
                        error_x={"type": "data", "array": 1.96 * t["ep"], "visible": True})
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=titulo, xaxis_title="coeficiente (logit, covariável em z-score)",
                      template="plotly_white", height=700)
    return fig


# --------------------------------------------------------------------------
# Parte 10 - Diagnóstico
# --------------------------------------------------------------------------
def _amostra(n: int, maximo: int = 1500, seed: int = 0) -> np.ndarray:
    """Índices para desenhar no máximo `maximo` pontos (a figura fica leve)."""
    if n <= maximo:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, maximo, replace=False))


def _medias_por_faixa(x: Any, r: Any, faixas: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Média do resíduo em faixas de x com o mesmo número de pontos (curva de tendência)."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    ordem = np.argsort(x)
    grupos = np.array_split(ordem, faixas)
    return (np.array([x[g].mean() for g in grupos]), np.array([r[g].mean() for g in grupos]))


def meio_normal(envelopes: Mapping[str, Mapping[str, np.ndarray]],
                titulo: str = "Gráfico half-normal com envelope simulado") -> go.Figure:
    """Um painel por modelo: |resíduo| ordenado contra quantis half-normal e banda simulada."""
    fig = make_subplots(rows=1, cols=len(envelopes), subplot_titles=list(envelopes))
    for j, (nome, e) in enumerate(envelopes.items(), start=1):
        idx = _amostra(len(e["observado"]), 1200)
        x = e["teorico"][idx]
        for chave, cor, dash in [("inferior", "gray", "dash"), ("superior", "gray", "dash"),
                                 ("mediana", "gray", "dot")]:
            fig.add_scatter(x=x, y=e[chave][idx], mode="lines", line={"color": cor, "dash": dash, "width": 1},
                            showlegend=False, row=1, col=j)
        fig.add_scatter(x=x, y=e["observado"][idx], mode="markers", marker={"size": 4, "color": "#1f77b4"},
                        showlegend=False, row=1, col=j)
        fig.update_xaxes(title_text="quantil half-normal", row=1, col=j)
    fig.update_yaxes(title_text="|resíduo quantílico| ordenado", row=1, col=1)
    fig.update_layout(title=titulo, template="plotly_white")
    return fig


def residuos_contra(x: Any, residuos: Mapping[str, Any], xtitulo: str, titulo: str, faixas: int = 20) -> go.Figure:
    """Painéis (um por modelo) de resíduo contra `x`, com a média por faixa em vermelho."""
    x = np.asarray(x, float)
    fig = make_subplots(rows=1, cols=len(residuos), subplot_titles=list(residuos), shared_yaxes=True)
    for j, (nome, r) in enumerate(residuos.items(), start=1):
        r = np.asarray(r, float)
        idx = _amostra(len(r), 1200)
        fig.add_scatter(x=x[idx], y=r[idx], mode="markers", marker={"size": 3, "opacity": 0.35},
                        showlegend=False, row=1, col=j)
        mx, mr = _medias_por_faixa(x, r, faixas)
        fig.add_scatter(x=mx, y=mr, mode="lines+markers", line={"color": "crimson"},
                        showlegend=False, row=1, col=j)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=j)
        fig.update_xaxes(title_text=xtitulo, row=1, col=j)
    fig.update_layout(title=titulo, template="plotly_white", height=420)
    return fig


def caixa_regiao(residuos: Mapping[str, Any], regiao: Any, titulo: str = "Resíduos quantílicos por região") -> go.Figure:
    """Box plot do resíduo por região, um traço por modelo."""
    fig = go.Figure()
    regiao = np.asarray(regiao).astype(str)
    for nome, r in residuos.items():
        fig.add_box(x=regiao, y=np.asarray(r, float), name=nome, boxpoints=False)
    fig.update_layout(title=titulo, boxmode="group", template="plotly_white",
                      yaxis_title="resíduo quantílico")
    return fig


def alavanca_cook(h: Any, cook: Any, rotulos: Sequence[str], p: int,
                  titulo: str = "Alavanca e distância de Cook (quase-binomial)") -> go.Figure:
    """Dispersão alavanca x Cook; linhas em 2p/n (alavanca) e 4/n (Cook)."""
    h, cook = np.asarray(h, float), np.asarray(cook, float)
    n = len(h)
    fig = go.Figure(go.Scattergl(x=h, y=cook, mode="markers", text=list(rotulos),
                                 marker={"size": 4, "opacity": 0.5}))
    fig.add_vline(x=2 * p / n, line_dash="dash", line_color="gray")
    fig.add_hline(y=4 / n, line_dash="dash", line_color="gray")
    fig.update_layout(title=titulo, xaxis_title="alavanca h", yaxis_title="distância de Cook",
                      yaxis_type="log", template="plotly_white")
    return fig


def mapa_residuos(gdf: Any, residuo: Any, titulo: str, tolerancia: float = 0.05) -> go.Figure:
    """Coroplético de resíduos; geometria simplificada (graus) para o HTML ficar leve."""
    g = gdf.copy()
    g["geometry"] = g.geometry.simplify(tolerancia, preserve_topology=True)
    g = g.reset_index(drop=True)
    fig = go.Figure(go.Choropleth(geojson=g.geometry.__geo_interface__, locations=g.index.astype(str),
                                  z=np.clip(np.asarray(residuo, float), -4, 4), zmin=-4, zmax=4,
                                  colorscale="RdBu", marker_line_width=0,
                                  colorbar={"title": "resíduo"}))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title=titulo, template="plotly_white", height=650)
    return fig


# --------------------------------------------------------------------------
# Blocos de figura que estavam nas células dos notebooks 01 e 04
# --------------------------------------------------------------------------
def barras_resultado_uf(tab_uf: pd.DataFrame) -> go.Figure:
    """Barras agrupadas da proporção do PT por UF (linhas) e ano (colunas), com a linha de 50%."""
    fig = go.Figure([go.Bar(x=tab_uf.index, y=tab_uf[a], name=str(a)) for a in tab_uf.columns])
    fig.add_hline(y=0.5, line_dash="dash", line_color=COR_REFERENCIA)
    fig.update_yaxes(title_text="Proporção PT no 2º turno", tickformat=".0%")
    fig.update_layout(barmode="group")
    return fig


def dispersao_lag(df: pd.DataFrame, y: str, x: str, nome: str, prefixo: str) -> tuple[Any, Any]:
    """Dispersão de y contra x (logit do PT em dois turnos) com LOWESS, OLS e y = x.

    Devolve os ajustes OLS e WLS (peso n do turno de y) para a célula comentar a inclinação.
    """
    d = df[[y, x, "n_2t_" + y[-4:], "regiao", "nome_municipio", "sigla_uf"]].copy()
    d.columns = ["y", "x", "n", "regiao", "nome_municipio", "sigla_uf"]
    ols = sm.OLS(d["y"], sm.add_constant(d["x"])).fit()
    wls = sm.WLS(d["y"], sm.add_constant(d["x"]), weights=d["n"]).fit()
    fig = go.Figure()
    for r, g in d.groupby("regiao"):
        fig.add_trace(go.Scattergl(x=g["x"], y=g["y"], mode="markers", name=r, marker={"size": 4, "opacity": 0.5, "color": CORES_REGIAO[r]},
                       hovertext=g["nome_municipio"] + " (" + g["sigla_uf"] + ")",
                       hovertemplate="%{hovertext}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>"))
    fig.add_trace(curva_lowess(d["x"], d["y"], "LOWESS"))
    t = np.array([d["x"].min(), d["x"].max()])
    fig.add_trace(go.Scatter(x=t, y=ols.params["const"] + ols.params["x"] * t, mode="lines", name="OLS", line={"color": "#0b0b0b", "width": 2}))
    fig.add_trace(go.Scatter(x=t, y=t, mode="lines", name="y = x", line={"color": COR_REFERENCIA, "dash": "dash"}))
    fig.update_xaxes(title_text=x); fig.update_yaxes(title_text=y)
    mostrar_e_salvar(fig, prefixo + nome, titulo=f"{y} x {x}", altura=520)
    return ols, wls


def figura_backtest(var: str, reg_t: pd.DataFrame, uf_t: pd.DataFrame, peso: str = "n_prev") -> go.Figure:
    """Barras do erro (previsto menos real, p.p.) por região e por UF, uma cor por modelo."""
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7], vertical_spacing=0.18,
                        subplot_titles=("Erro por região (p.p.)", "Erro por UF (p.p.)"))
    for i, (nome, g) in enumerate(reg_t.query("variante == @var and peso == @peso").groupby("modelo", sort=False)):
        fig.add_bar(x=g["grupo"], y=g["erro_pp"], name=nome, marker_color=cores[i % 6], legendgroup=nome, row=1, col=1)
    for i, (nome, g) in enumerate(uf_t.query("variante == @var and peso == @peso").groupby("modelo", sort=False)):
        fig.add_bar(x=g["grupo"], y=g["erro_pp"], name=nome, marker_color=cores[i % 6], legendgroup=nome,
                    showlegend=False, row=2, col=1)
    fig.update_layout(barmode="group", height=750, title=f"Backtest 2022→2026: {var} (previsto − real, p.p.)",
                      legend_title="modelo", template="plotly_white")
    return fig


# --------------------------------------------------------------------------
# Notebooks 05 e 06
# --------------------------------------------------------------------------

AZUL_FINAL = "#1b6ca8"
LARANJA_REAL = "#c2571a"


def dp_por_regiao(dp: pd.DataFrame, regs: Sequence[str], opcoes: Sequence[str]) -> go.Figure:
    """DP do resíduo quantílico por região, uma barra por opção de dispersão (ideal = 1)."""
    fig = go.Figure([go.Bar(name=o, x=regs, y=dp.loc[o, regs].values) for o in opcoes])
    fig.add_hline(y=1, line_dash="dash", line_color="#888")
    fig.update_layout(barmode="group", title="DP do resíduo quantílico por região, por opção de dispersão (ideal = 1)",
                      yaxis_title="DP", template="plotly_white", height=420)
    return fig


def moran_contra_k(esf: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=esf.k_esf, y=esf.I_moran, mode="lines+markers", marker={"size": 9, "color": AZUL_FINAL}))
    fig.update_layout(title="I de Moran dos resíduos quantílicos contra o nº de autovetores (k)", xaxis_title="k",
                      yaxis_title="I de Moran", template="plotly_white", height=400)
    return fig


def correlograma_residuos(cg: pd.DataFrame, nome_modelo: str, corte_km: float) -> go.Figure:
    fig = go.Figure(go.Bar(x=[f"{a:g}-{b:g}" for a, b in zip(cg.ini_km, cg.fim_km)], y=cg.correlacao, marker_color=AZUL_FINAL))
    fig.add_hline(y=0.05, line_dash="dash", line_color=LARANJA_REAL, annotation_text="limiar 0,05")
    fig.update_layout(title=f"Correlograma dos resíduos quantílicos ({nome_modelo}); corte = {corte_km:g} km",
                      xaxis_title="faixa de distância (km)", yaxis_title="correlação média", template="plotly_white", height=420)
    return fig


def regional_base_e_final(d: pd.DataFrame, base_pt: pd.Series, nome_modelo: str) -> go.Figure:
    """Share de Lula por região: ponto do bb_selecao, modelo final com IC95 de Conley e valor real."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_pt.loc[d.nome].values, y=d.nome, mode="markers", name="bb_selecao (ponto)",
                             marker={"color": "#888", "size": 8}))
    fig.add_trace(go.Scatter(x=d.lula_pct, y=d.nome, mode="markers", name=f"{nome_modelo}, Conley (IC95)",
                             error_x={"type": "data", "symmetric": False, "array": d.q975 - d.lula_pct, "arrayminus": d.lula_pct - d.q025},
                             marker={"color": AZUL_FINAL, "size": 9}))
    fig.add_trace(go.Scatter(x=d.real_pct, y=d.nome, mode="markers", name="real",
                             marker={"color": LARANJA_REAL, "symbol": "diamond", "size": 10}))
    fig.update_layout(title="Share de Lula por região (LOSO, σ_UF): bb_selecao e modelo final",
                      xaxis_title="% dos votos válidos", template="plotly_white", height=460)
    return fig


def erro_por_corte(tab: pd.DataFrame, anos: Sequence[int], coluna: str, rotulo_y: str, titulo: str) -> go.Figure:
    """Erro de cada método (bruta, swing, modelo) por corte de apuração, um painel por ano, eixo y em log."""
    cores = {"bruta": "#888888", "swing": LARANJA_REAL, "modelo": AZUL_FINAL}
    fig = make_subplots(rows=1, cols=2, subplot_titles=[str(a) for a in anos], shared_yaxes=True)
    for j, ano in enumerate(anos, 1):
        for met, cor in cores.items():
            t = tab[(tab.ano == ano) & (tab.metodo == met)]
            fig.add_trace(go.Scatter(x=t.corte, y=t[coluna], mode="lines+markers", name=met, legendgroup=met,
                                     showlegend=(j == 1), line={"color": cor}), row=1, col=j)
        fig.update_xaxes(title_text="% dos válidos recebidos", row=1, col=j)
    fig.update_yaxes(title_text=rotulo_y, type="log", row=1, col=1)
    fig.update_layout(title=titulo, template="plotly_white", height=420)
    return fig
