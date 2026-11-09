"""Funções de gráfico (Plotly) e de dados espaciais usadas pelos notebooks do projeto.

Chame configurar(df, pasta_figuras) antes de usar mostrar_e_salvar e as funções de painel.
Mapas: matplotlib não está no ambiente, então os mapas são coropléticos Plotly com a
geometria simplificada embutida (um GeoJSON por figura).
"""
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess

COR_PONTO = "#2a78d6"
COR_CURVA = "#eb6834"
COR_REFERENCIA = "#898781"
COR_CRITICO = "#d03b3b"
ESCALA_DIVERGENTE = [[0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec835a"], [1, "#a32a2a"]]
CORES_REGIAO = {"Norte": "#1baf7a", "Nordeste": "#eb6834", "Centro-Oeste": "#eda100",
                "Sudeste": "#2a78d6", "Sul": "#4a3aa7"}

pio.templates["eleicoes"] = go.layout.Template(layout=dict(
    colorway=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="#0b0b0b"),
    paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", hovermode="closest",
    hoverlabel=dict(bgcolor="#ffffff", font_size=12),
))
pio.templates.default = "plotly_white+eleicoes"

df = None
output_fig_dir = None


def configurar(dados, pasta_figuras):
    # Conjunto de dados (hover) e pasta onde as figuras HTML são salvas
    global df, output_fig_dir
    df = dados
    output_fig_dir = pasta_figuras
    os.makedirs(output_fig_dir, exist_ok=True)


def texto_hover(indices, extras=None):
    indices = list(indices)
    extras = {k: np.asarray(v) for k, v in (extras or {}).items()}
    textos = []
    for k, i in enumerate(indices):
        partes = [f"<b>{df.at[i, 'nome_municipio']}</b> ({df.at[i, 'sigla_uf']})"]
        partes += [f"{nome}: {v[k]:.3f}" for nome, v in extras.items()]
        textos.append("<br>".join(partes))
    return textos


def pontos(x, y, indices, extras=None, nome="Municípios", cor=COR_PONTO, tamanho=5, **kwargs):
    return go.Scatter(
        x=np.asarray(x), y=np.asarray(y), mode="markers", name=nome,
        marker=dict(color=cor, size=tamanho, opacity=0.5),
        hovertext=texto_hover(indices, extras), hovertemplate="%{hovertext}<extra></extra>", **kwargs)


def curva_lowess(x, y, nome="LOWESS", cor=COR_CURVA, frac=2 / 3, it=0):
    s = lowess(np.asarray(y, float), np.asarray(x, float), frac=frac, it=it)
    return go.Scatter(x=s[:, 0], y=s[:, 1], mode="lines", name=nome, line=dict(color=cor, width=2.5),
                      hovertemplate=f"{nome}<br>x = %{{x:.3f}}<br>y = %{{y:.3f}}<extra></extra>")


def linha_ref(fig, valor, inicio, fim, nome, cor=COR_REFERENCIA, vertical=False, traco="dash", row=None, col=None):
    t = np.linspace(inicio, fim, 40)
    c = np.full_like(t, valor)
    x, y = (c, t) if vertical else (t, c)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nome, line=dict(color=cor, width=1.5, dash=traco),
                             hovertemplate=f"{nome}: {valor:.4g}<extra></extra>"), row=row, col=col)


def legenda_unica(fig):
    vistos = set()
    for traco in fig.data:
        if traco.name is None:
            continue
        traco.legendgroup = traco.name
        traco.showlegend = traco.name not in vistos and traco.showlegend is not False
        if traco.showlegend:
            vistos.add(traco.name)


def mostrar_e_salvar(fig, nome=None, titulo=None, altura=500, largura=None):
    legenda_unica(fig)
    fig.update_layout(title=titulo, height=altura, width=largura, margin=dict(t=90 if titulo else 60))
    if nome:
        fig.write_html(f"{output_fig_dir}/{nome}.html", include_plotlyjs="cdn")
    fig.show()


def histograma(valores, nome_var, pesos=None, name=None, titulo=None, bins=50):
    # Histograma simples (frequência ou soma de pesos por classe)
    v = np.asarray(valores, float)
    fig = go.Figure(go.Histogram(
        x=v, y=None if pesos is None else np.asarray(pesos, float), histfunc="count" if pesos is None else "sum",
        nbinsx=bins, marker=dict(color=COR_PONTO, line=dict(width=0.5, color="#fcfcfb")),
        hovertemplate=f"{nome_var}: %{{x}}<br>%{{y:,.0f}}<extra></extra>"))
    fig.update_xaxes(title_text=nome_var)
    fig.update_yaxes(title_text="Municípios" if pesos is None else "Votos")
    fig.update_layout(showlegend=False)
    mostrar_e_salvar(fig, name, titulo=titulo, altura=380)


def heatmap_correlacao(matriz, titulo, name):
    n = len(matriz)
    fig = go.Figure(go.Heatmap(
        z=matriz.values, x=list(matriz.columns), y=list(matriz.index), zmin=-1, zmax=1, zmid=0,
        colorscale=ESCALA_DIVERGENTE, text=matriz.round(2).values, texttemplate="%{text:.2f}", textfont=dict(size=8),
        xgap=1, ygap=1, colorbar=dict(title="ρ"), hovertemplate="%{y} x %{x}<br>correlação = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    mostrar_e_salvar(fig, name, titulo=titulo, altura=max(600, 28 * n + 200), largura=max(800, 28 * n + 250))


def logit_empirico_por_decil(x, y, n, q=10):
    # Por decil de x: média de x e log(ΣY / Σ(n−Y)). Pesos naturais: municípios grandes pesam mais.
    d = pd.DataFrame({"x": x, "y": y, "n": n})
    d["bin"] = pd.qcut(d["x"].rank(method="first"), q, labels=False)
    g = d.groupby("bin").agg(x=("x", "mean"), y=("y", "sum"), n=("n", "sum"))
    g["logit"] = np.log(g["y"] / (g["n"] - g["y"]))
    return g


def media_ponderada_por_decil(x, z, w, q=10):
    d = pd.DataFrame({"x": x, "z": z, "w": w})
    d["bin"] = pd.qcut(d["x"].rank(method="first"), q, labels=False)
    return d.groupby("bin").apply(lambda g: pd.Series({"x": g["x"].mean(), "z": np.average(g["z"], weights=g["w"])}),
                                  include_groups=False)


def painel_linearidade(variaveis, calcula, rotulo_y, name, n_cols=4, titulo=None):
    # calcula(var) -> DataFrame com colunas x e y (um ponto por decil)
    n_rows = math.ceil(len(variaveis) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=variaveis, vertical_spacing=0.06,
                        horizontal_spacing=0.06)
    for k, v in enumerate(variaveis):
        r, c = k // n_cols + 1, k % n_cols + 1
        g = calcula(v)
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers+lines", name="Decil",
                                 marker=dict(color=COR_PONTO, size=7), line=dict(color=COR_PONTO, width=1.2),
                                 hovertemplate=f"{v}: %{{x:.3f}}<br>{rotulo_y}: %{{y:.3f}}<extra></extra>"), row=r, col=c)
    fig.update_yaxes(title_text=rotulo_y, col=1, title_font=dict(size=10))
    mostrar_e_salvar(fig, name, titulo=titulo, altura=230 * n_rows + 80)


# ---------- espacial ----------

def pesos_queen(geo):
    # Queen + ilhas ligadas ao vizinho mais próximo, padronizado por linha. `geo` na ordem das observações.
    from libpysal.weights import KNN, Queen
    from libpysal.weights.util import attach_islands
    w = Queen.from_dataframe(geo, use_index=False)
    if w.islands:
        w = attach_islands(w, KNN.from_dataframe(geo, k=1))
    w.transform = "r"
    return w


def mapa(geo_simpl, valores, titulo, name, rotulo, escala=ESCALA_DIVERGENTE, centro=None, zmin=None, zmax=None,
         hover=None, ticks=None):
    # Mapa coroplético; geo_simpl com índice RangeIndex alinhado a `valores`; geometria arredondada para reduzir tamanho
    feats = []
    for i, g in enumerate(geo_simpl.geometry):
        feats.append({"type": "Feature", "id": i, "properties": {}, "geometry": _arredonda(g.__geo_interface__)})
    gj = {"type": "FeatureCollection", "features": feats}
    kw = dict(zmid=centro) if centro is not None else {}
    if zmin is not None:
        kw.update(zmin=zmin, zmax=zmax)
    fig = go.Figure(go.Choropleth(
        geojson=gj, locations=list(range(len(geo_simpl))), z=np.asarray(valores, float), colorscale=escala,
        marker_line_width=0, colorbar=dict(title=rotulo, **({} if ticks is None else dict(tickvals=ticks[0], ticktext=ticks[1]))), text=hover,
        hovertemplate="%{text}<br>" + rotulo + ": %{z:.3f}<extra></extra>", **kw))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#fcfcfb")
    mostrar_e_salvar(fig, name, titulo=titulo, altura=620, largura=700)


def _arredonda(gi, casas=2):
    def r(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], casas), round(c[1], casas)]
        return [r(x) for x in c]
    return {"type": gi["type"], "coordinates": r(gi["coordinates"])}
