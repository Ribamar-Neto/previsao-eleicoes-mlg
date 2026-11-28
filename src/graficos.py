"""Funções de gráfico (Plotly) dos notebooks do projeto.

Chame configurar(df, pasta_figuras) antes de usar mostrar_e_salvar e as funções de painel.
Mapas: matplotlib não está no ambiente, então os mapas são coropléticos Plotly com a
geometria simplificada embutida (um GeoJSON por figura).
Os gráficos de diagnóstico do MLG (Parte 10 em diante) ficam na segunda metade do módulo.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import math
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from numpy.typing import ArrayLike
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

COR_PONTO = "#2a78d6"
COR_CURVA = "#eb6834"
COR_REFERENCIA = "#898781"
COR_CRITICO = "#d03b3b"
ESCALA_DIVERGENTE = [[0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec835a"], [1, "#a32a2a"]]
CORES_REGIAO = {
    "Norte": "#1baf7a",
    "Nordeste": "#eb6834",
    "Centro-Oeste": "#eda100",
    "Sudeste": "#2a78d6",
    "Sul": "#4a3aa7",
}

pio.templates["eleicoes"] = go.layout.Template(
    layout={
        "colorway": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "font": {"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": "#0b0b0b"},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "hovermode": "closest",
        "hoverlabel": {"bgcolor": "#ffffff", "font_size": 12},
        "separators": ",.",
    }
)
pio.templates.default = "plotly_white+eleicoes"

# No notebook as figuras são interativas (hover, zoom); o PNG em results/figures é a cópia estática que os docs citam.
# Decisão do autor: compatibilidade com o visualizador do GitHub não é requisito.
LARGURA_PNG = 900
ALTURA_PNG = 520
ESCALA_PNG = 1.5
PASTA_FIGURAS_PADRAO = Path(__file__).resolve().parent.parent / "results" / "figures"
pio.renderers.default = "notebook_connected"

_servidor_kaleido: bool = False


warnings.filterwarnings("ignore", message="The kopts argument is ignored if using a server")


def _iniciar_kaleido() -> None:
    # Um único Chrome por processo: sem isso cada write_image abre e fecha o navegador (~2 s por figura).
    global _servidor_kaleido
    if _servidor_kaleido:
        return
    _servidor_kaleido = True
    # sem servidor, write_image ainda funciona (mais lento)
    with contextlib.suppress(Exception):
        import kaleido

        kaleido.start_sync_server(silence_warnings=True)
        atexit.register(kaleido.stop_sync_server, silence_warnings=True)


df: pd.DataFrame = pd.DataFrame()
output_fig_dir: str | Path | None = None


def configurar(dados: pd.DataFrame, pasta_figuras: str | Path) -> None:
    # Conjunto de dados (hover) e pasta onde os PNGs são salvos
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


def pontos(
    x: Any,
    y: Any,
    indices: Sequence[int],
    extras: Mapping[str, Any] | None = None,
    nome: str = "Municípios",
    cor: str = COR_PONTO,
    tamanho: int = 5,
    **kwargs: Any,
) -> go.Scatter:
    return go.Scatter(
        x=np.asarray(x),
        y=np.asarray(y),
        mode="markers",
        name=nome,
        marker={"color": cor, "size": tamanho, "opacity": 0.5},
        hovertext=texto_hover(indices, extras),
        hovertemplate="%{hovertext}<extra></extra>",
        **kwargs,
    )


def curva_lowess(
    x: Any, y: Any, nome: str = "LOWESS", cor: str = COR_CURVA, frac: float = 2 / 3, it: int = 0
) -> go.Scatter:
    s = lowess(np.asarray(y, float), np.asarray(x, float), frac=frac, it=it)
    return go.Scatter(
        x=s[:, 0],
        y=s[:, 1],
        mode="lines",
        name=nome,
        line={"color": cor, "width": 2.5},
        hovertemplate=f"{nome}<br>x = %{{x:.3f}}<br>y = %{{y:.3f}}<extra></extra>",
    )


def linha_ref(
    fig: go.Figure,
    valor: float,
    inicio: float,
    fim: float,
    nome: str,
    cor: str = COR_REFERENCIA,
    vertical: bool = False,
    traco: str = "dash",
    row: int | None = None,
    col: int | None = None,
) -> None:
    t = np.linspace(inicio, fim, 40)
    c = np.full_like(t, valor)
    x, y = (c, t) if vertical else (t, c)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=nome,
            line={"color": cor, "width": 1.5, "dash": traco},
            hovertemplate=f"{nome}: {valor:.4g}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def legenda_unica(fig: go.Figure) -> None:
    vistos: set[str] = set()
    for traco in fig.data:
        if traco.name is None:
            continue
        traco.legendgroup = traco.name
        traco.showlegend = traco.name not in vistos and traco.showlegend is not False
        if traco.showlegend:
            vistos.add(traco.name)


def mostrar_e_salvar(
    fig: go.Figure,
    nome: str | None = None,
    titulo: str | None = None,
    altura: int | None = None,
    largura: int | None = None,
) -> None:
    # Salva o PNG (se houver nome) e mostra a figura; título/tamanho são opcionais para figuras já formatadas.
    legenda_unica(fig)
    if titulo is not None:
        fig.update_layout(title=titulo, margin={"t": 90})
    if altura is not None or largura is not None:
        fig.update_layout(height=altura, width=largura)
    if fig.layout.height is None:
        fig.update_layout(height=ALTURA_PNG)  # mesma altura no notebook e no PNG
    if nome:
        salvar(fig, nome)
    fig.show()


def histograma(
    valores: Any, nome_var: str, pesos: Any = None, name: str | None = None, titulo: str | None = None, bins: int = 50
) -> None:
    # Histograma simples (frequência ou soma de pesos por classe)
    v = np.asarray(valores, float)
    fig = go.Figure(
        go.Histogram(
            x=v,
            y=None if pesos is None else np.asarray(pesos, float),
            histfunc="count" if pesos is None else "sum",
            nbinsx=bins,
            marker={"color": COR_PONTO, "line": {"width": 0.5, "color": "#ffffff"}},
            hovertemplate=f"{nome_var}: %{{x}}<br>%{{y:,.0f}}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=nome_var)
    fig.update_yaxes(title_text="Municípios" if pesos is None else "Votos")
    fig.update_layout(showlegend=False)
    mostrar_e_salvar(fig, name, titulo=titulo, altura=380)


def heatmap_correlacao(matriz: pd.DataFrame, titulo: str, name: str) -> None:
    n = len(matriz)
    fig = go.Figure(
        go.Heatmap(
            z=matriz.values,
            x=list(matriz.columns),
            y=list(matriz.index),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=ESCALA_DIVERGENTE,
            text=matriz.round(2).values,
            texttemplate="%{text:.2f}",
            textfont={"size": 8},
            xgap=1,
            ygap=1,
            colorbar={"title": "ρ"},
            hovertemplate="%{y} x %{x}<br>correlação = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    mostrar_e_salvar(fig, name, titulo=titulo, altura=max(600, 28 * n + 200), largura=max(800, 28 * n + 250))


def faixas_por_decil(x: Any, q: int = 10, massa_minima: float = 0.10) -> pd.Series:
    """Faixa de cada observação: decis de x; se o mínimo concentra >= `massa_minima`, ele forma a faixa 0 sozinho
    e os decis valem para o resto (senão quase todos os decis colapsam no mínimo, ex.: % indígenas)."""
    x = pd.Series(np.asarray(x, float))
    empate = (x == x.min()).to_numpy()
    faixa = pd.Series(0, index=x.index, dtype=int)
    if empate.mean() >= massa_minima and not empate.all():
        resto = x[~empate]
        faixa[~empate] = 1 + pd.qcut(resto.rank(method="first"), q, labels=False, duplicates="drop")
        return faixa
    return pd.Series(pd.qcut(x.rank(method="first"), q, labels=False, duplicates="drop"), index=x.index)


def logit_empirico_por_decil(x: Any, y: Any, n: Any, q: int = 10) -> pd.DataFrame:
    # Por decil de x: média de x e log(ΣY / Σ(n−Y)). Pesos naturais: municípios grandes pesam mais.
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float), "n": np.asarray(n, float)})
    d["bin"] = faixas_por_decil(d["x"], q, massa_minima=0.10)
    g = d.groupby("bin").agg(x=("x", "mean"), y=("y", "sum"), n=("n", "sum"))
    g["logit"] = np.log(g["y"] / (g["n"] - g["y"]))
    return g


def media_ponderada_por_decil(x: Any, z: Any, w: Any, q: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"x": np.asarray(x, float), "z": np.asarray(z, float), "w": np.asarray(w, float)})
    d["bin"] = faixas_por_decil(d["x"], q, massa_minima=0.10)
    return d.groupby("bin").apply(
        lambda g: pd.Series({"x": g["x"].mean(), "z": np.average(g["z"], weights=g["w"])}), include_groups=False
    )


def _quebra_titulo(nome: str, limite: int = 20) -> str:
    # Quebra nomes longos em "_" (com <br>) para os títulos dos subpainéis não se sobreporem
    if len(nome) <= limite:
        return nome
    partes, linha = [], ""
    for p in nome.split("_"):
        if linha and len(linha) + 1 + len(p) > limite:
            partes.append(linha + "_")
            linha = p
        else:
            linha = f"{linha}_{p}" if linha else p
    partes.append(linha)
    return "<br>".join(partes)


_MIN_FAIXAS_EIXO = 3  # com 2 faixas o maior salto é sempre a amplitude inteira


def eixo_por_ordem(x: Sequence[float], limite: float = 0.5) -> bool:
    """True quando o maior salto entre médias de faixa consecutivas passa de `limite` da amplitude: no eixo em
    valor, as demais faixas se amontoam num ponto (ex.: % indígenas) e o painel fica ilegível."""
    v = np.sort(np.asarray(x, float))
    amplitude = v[-1] - v[0]
    return len(v) >= _MIN_FAIXAS_EIXO and amplitude > 0 and float(np.diff(v).max()) > limite * amplitude


def _num(v: float) -> str:
    return f"{v:.2g}".replace(".", ",")


def painel_linearidade(
    variaveis: Sequence[str],
    calcula: Callable[[str], pd.DataFrame],
    rotulo_y: str,
    name: str,
    n_cols: int = 4,
    titulo: str | None = None,
) -> None:
    # calcula(var) -> DataFrame com colunas x e y (um ponto por decil)
    n_rows = math.ceil(len(variaveis) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[_quebra_titulo(v, limite=20) for v in variaveis],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )
    fig.update_annotations(font_size=13)
    for k, v in enumerate(variaveis):
        r, c = k // n_cols + 1, k % n_cols + 1
        g = calcula(v)
        x = g["x"].to_numpy(float)
        por_ordem = eixo_por_ordem(x, limite=0.5)
        pos = np.arange(1, len(x) + 1) if por_ordem else x
        fig.add_trace(
            go.Scatter(
                x=pos,
                y=g["y"],
                customdata=x,
                mode="markers+lines",
                showlegend=False,
                marker={"color": COR_PONTO, "size": 7},
                line={"color": COR_PONTO, "width": 1.2},
                hovertemplate=f"{v}: %{{customdata:.3f}}<br>{rotulo_y}: %{{y:.3f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )
        if por_ordem:
            marcas = sorted({0, len(x) // 2, len(x) - 1})
            fig.update_xaxes(
                tickvals=pos[marcas],
                ticktext=[_num(x[i]) for i in marcas],
                title_text="faixa (em ordem; média no rótulo)",
                title_font={"size": 10},
                row=r,
                col=c,
            )
    fig.update_yaxes(title_text=rotulo_y, col=1, title_font={"size": 10})
    mostrar_e_salvar(fig, name, titulo=titulo, altura=260 * n_rows + 80)


# ---------- espacial ----------


def mapa(
    geo_simpl: Any,
    valores: Any,
    titulo: str,
    name: str,
    rotulo: str,
    escala: Sequence[Sequence[Any]] = ESCALA_DIVERGENTE,
    centro: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    hover: Sequence[str] | None = None,
    ticks: tuple[Sequence[float], Sequence[str]] | None = None,
    altura: int = 470,
    largura: int = 560,
) -> None:
    # Mapa coroplético; geo_simpl com índice RangeIndex alinhado a `valores`; geometria arredondada para reduzir tamanho
    feats = []
    for i, g in enumerate(geo_simpl.geometry):
        feats.append(
            {"type": "Feature", "id": i, "properties": {}, "geometry": _arredonda(_orientada(g).__geo_interface__)}
        )
    gj = {"type": "FeatureCollection", "features": feats}
    kw: dict[str, Any] = {"zmid": centro} if centro is not None else {}
    if zmin is not None:
        kw.update(zmin=zmin, zmax=zmax)
    fig = go.Figure(
        go.Choropleth(
            geojson=gj,
            locations=list(range(len(geo_simpl))),
            z=np.asarray(valores, float),
            colorscale=escala,
            marker_line_width=0,
            colorbar=dict(
                title=rotulo,
                thickness=14,
                len=0.75,
                x=1.0,
                xanchor="left",
                **({} if ticks is None else {"tickvals": ticks[0], "ticktext": ticks[1]}),
            ),
            text=hover,
            hovertemplate="%{text}<br>" + rotulo + ": %{z:.3f}<extra></extra>",
            **kw,
        )
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#ffffff", projection_type="mercator")
    # Título e margens aqui (não via `titulo=` de mostrar_e_salvar, que fixa t=90 e abre um vão acima do mapa);
    # a margem direita abriga a barra de cores, e o mapa ocupa o resto, centrado sob o título.
    fig.update_layout(
        title={"text": titulo, "x": 0.5, "xanchor": "center", "y": 0.97, "yanchor": "top"},
        margin={"t": 40, "l": 0, "r": 190 if ticks is not None else 90, "b": 0},
        title_font_size=14,
    )
    mostrar_e_salvar(fig, name, titulo=None, altura=altura, largura=largura)


def _orientada(geom: Any) -> Any:
    # O plotly (d3-geo) pede anéis externos no sentido horário; no sentido contrário o polígono preenche o complemento
    # (o mapa vira um retângulo de uma cor só).
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.geometry.polygon import orient

    if isinstance(geom, Polygon):
        return orient(geom, sign=-1.0)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient(pl, sign=-1.0) for pl in geom.geoms])
    return geom


def _arredonda(gi: Mapping[str, Any], casas: int = 2) -> dict[str, Any]:
    def r(c: Any) -> list[Any]:
        if isinstance(c[0], (int, float)):
            return [round(c[0], casas), round(c[1], casas)]
        return [r(x) for x in c]

    return {"type": gi["type"], "coordinates": r(gi["coordinates"])}


# ---------- diagnóstico do MLG (`02-MLG` em diante) ----------
# Salvos em results/figures/mlg_*.png.


def salvar(fig: go.Figure, nome: str | Path) -> go.Figure:
    """Grava `<pasta de figuras>/<nome>.png`; só o nome-base conta (extensão e diretório são descartados)."""
    _iniciar_kaleido()
    pasta = Path(output_fig_dir) if output_fig_dir is not None else PASTA_FIGURAS_PADRAO
    pasta.mkdir(parents=True, exist_ok=True)
    fig.write_image(
        str(pasta / f"{Path(nome).stem}.png"),
        width=fig.layout.width or LARGURA_PNG,
        height=fig.layout.height or ALTURA_PNG,
        scale=ESCALA_PNG,
    )
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
    fig.update_layout(
        title=titulo, xaxis_title="quantil normal teórico", yaxis_title="resíduo quantílico", template="plotly_white"
    )
    return fig


def forest_coeficientes(
    tabelas: Mapping[str, pd.DataFrame], variaveis: Sequence[str] | None = None, titulo: str = "Coeficientes (IC 95%)"
) -> go.Figure:
    """Coeficientes padronizados com IC 95% lado a lado por modelo."""
    fig = go.Figure()
    for nome, t in tabelas.items():
        t = t if variaveis is None else t.loc[[v for v in variaveis if v in t.index]]
        t = t[t.index != "const"]
        fig.add_scatter(
            x=t["coef"],
            y=list(t.index),
            mode="markers",
            name=nome,
            error_x={"type": "data", "array": 1.96 * t["ep"], "visible": True},
        )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=titulo, xaxis_title="coeficiente (logit, covariável em z-score)", template="plotly_white", height=700
    )
    return fig


# --------------------------------------------------------------------------
# Parte 10 - Diagnóstico
# --------------------------------------------------------------------------
def _amostra(n: int, maximo: int = 1500, seed: int = 0) -> np.ndarray:
    """Índices para desenhar no máximo `maximo` pontos (a figura fica leve)."""
    if n <= maximo:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, maximo, replace=False))


FAIXAS_TENDENCIA = 20  # faixas da curva de tendência do resíduo


def _medias_por_faixa(x: ArrayLike, r: ArrayLike, faixas: int) -> tuple[np.ndarray, np.ndarray]:
    """Média do resíduo em faixas de x com o mesmo número de pontos (curva de tendência)."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    ordem = np.argsort(x)
    grupos = np.array_split(ordem, faixas)
    return (np.array([x[g].mean() for g in grupos]), np.array([r[g].mean() for g in grupos]))


def meio_normal(
    envelopes: Mapping[str, Mapping[str, np.ndarray]], titulo: str = "Gráfico half-normal com envelope simulado"
) -> go.Figure:
    """Um painel por modelo: |resíduo| ordenado contra quantis half-normal e banda simulada."""
    fig = make_subplots(rows=1, cols=len(envelopes), subplot_titles=list(envelopes))
    for j, (nome, e) in enumerate(envelopes.items(), start=1):
        idx = _amostra(len(e["observado"]), 1200)
        x = e["teorico"][idx]
        for chave, cor, dash in [
            ("inferior", "gray", "dash"),
            ("superior", "gray", "dash"),
            ("mediana", "gray", "dot"),
        ]:
            fig.add_scatter(
                x=x,
                y=e[chave][idx],
                mode="lines",
                line={"color": cor, "dash": dash, "width": 1},
                showlegend=False,
                row=1,
                col=j,
            )
        fig.add_scatter(
            x=x,
            y=e["observado"][idx],
            mode="markers",
            marker={"size": 4, "color": "#1f77b4"},
            showlegend=False,
            row=1,
            col=j,
        )
        fig.update_xaxes(title_text="quantil half-normal", row=1, col=j)
    fig.update_yaxes(title_text="|resíduo quantílico| ordenado", row=1, col=1)
    fig.update_layout(title=titulo, template="plotly_white")
    return fig


def residuos_contra(
    x: ArrayLike, residuos: Mapping[str, Any], xtitulo: str, titulo: str, rotulos: Sequence[str] | None = None
) -> go.Figure:
    """Painéis (um por modelo) de resíduo contra `x`, com a média por faixa em vermelho.

    O título do eixo x é um só, sob todos os painéis: repetido em cada painel estreito os textos se sobrepõem.
    """
    x = np.asarray(x, float)
    fig = make_subplots(
        rows=1, cols=len(residuos), subplot_titles=list(residuos), shared_yaxes=True, horizontal_spacing=0.02
    )
    for j, (nome, r) in enumerate(residuos.items(), start=1):
        r = np.asarray(r, float)
        idx = _amostra(len(r), 1200)
        texto = None if rotulos is None else [rotulos[i] for i in idx]
        fig.add_scatter(
            x=x[idx],
            y=r[idx],
            mode="markers",
            marker={"size": 3, "opacity": 0.35},
            name=nome,
            hovertext=texto,
            hovertemplate="%{hovertext}<br>x = %{x:.3f}<br>resíduo = %{y:.2f}<extra></extra>",
            showlegend=False,
            row=1,
            col=j,
        )
        mx, mr = _medias_por_faixa(x, r, FAIXAS_TENDENCIA)
        fig.add_scatter(
            x=mx,
            y=mr,
            mode="lines+markers",
            line={"color": "crimson"},
            name="média por faixa",
            hovertemplate="média por faixa<br>x = %{x:.3f}<br>resíduo = %{y:.2f}<extra></extra>",
            showlegend=False,
            row=1,
            col=j,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=j)
    fig.add_annotation(
        text=xtitulo,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.14,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_yaxes(title_text="resíduo quantílico", row=1, col=1)
    fig.update_layout(title=titulo, template="plotly_white", height=440, margin={"b": 90})
    return fig


def caixa_regiao(
    residuos: Mapping[str, Any], regiao: Any, titulo: str = "Resíduos quantílicos por região"
) -> go.Figure:
    """Box plot do resíduo por região, um traço por modelo."""
    fig = go.Figure()
    regiao = np.asarray(regiao).astype(str)
    for nome, r in residuos.items():
        fig.add_box(x=regiao, y=np.asarray(r, float), name=nome, boxpoints=False)
    fig.update_layout(title=titulo, boxmode="group", template="plotly_white", yaxis_title="resíduo quantílico")
    return fig


def alavanca_cook(
    h: Any, cook: Any, rotulos: Sequence[str], p: int, titulo: str = "Alavanca e distância de Cook (quase-binomial)"
) -> go.Figure:
    """Dispersão alavanca x Cook; linhas em 2p/n (alavanca) e 4/n (Cook)."""
    h, cook = np.asarray(h, float), np.asarray(cook, float)
    n = len(h)
    fig = go.Figure(go.Scattergl(x=h, y=cook, mode="markers", text=list(rotulos), marker={"size": 4, "opacity": 0.5}))
    fig.add_vline(x=2 * p / n, line_dash="dash", line_color="gray")
    fig.add_hline(y=4 / n, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=titulo,
        xaxis_title="alavanca h",
        yaxis_title="distância de Cook",
        yaxis_type="log",
        template="plotly_white",
    )
    return fig


def mapa_residuos(gdf: Any, residuo: Any, titulo: str, tolerancia: float = 0.05) -> go.Figure:
    """Coroplético de resíduos; geometria simplificada (graus) para o HTML ficar leve."""
    g = gdf.copy()
    g["geometry"] = [_orientada(geom) for geom in g.geometry.simplify(tolerancia, preserve_topology=True)]
    g = g.reset_index(drop=True)
    fig = go.Figure(
        go.Choropleth(
            geojson=g.geometry.__geo_interface__,
            locations=g.index.astype(str),
            z=np.clip(np.asarray(residuo, float), -4, 4),
            zmin=-4,
            zmax=4,
            colorscale="RdBu",
            marker_line_width=0,
            colorbar={"title": "resíduo"},
        )
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#ffffff")
    fig.update_layout(title=titulo, height=650, width=800)
    return fig


# --------------------------------------------------------------------------
# Blocos de figura que estavam nas células dos antigos notebooks 01 e 04 (hoje `01-AED` e `02-MLG`)
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
        fig.add_trace(
            go.Scattergl(
                x=g["x"],
                y=g["y"],
                mode="markers",
                name=r,
                marker={"size": 4, "opacity": 0.5, "color": CORES_REGIAO[r]},
                hovertext=g["nome_municipio"] + " (" + g["sigla_uf"] + ")",
                hovertemplate="%{hovertext}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
            )
        )
    fig.add_trace(curva_lowess(d["x"], d["y"], "LOWESS"))
    t = np.array([d["x"].min(), d["x"].max()])
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ols.params["const"] + ols.params["x"] * t,
            mode="lines",
            name="OLS",
            line={"color": "#0b0b0b", "width": 2},
        )
    )
    fig.add_trace(go.Scatter(x=t, y=t, mode="lines", name="y = x", line={"color": COR_REFERENCIA, "dash": "dash"}))
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    mostrar_e_salvar(fig, prefixo + nome, titulo=f"{y} x {x}", altura=520)
    return ols, wls


# --------------------------------------------------------------------------
# Notebooks 05 e 06
# --------------------------------------------------------------------------

AZUL_FINAL = "#1b6ca8"
LARANJA_REAL = "#c2571a"


def dp_por_regiao(dp: pd.DataFrame, regs: Sequence[str], opcoes: Sequence[str]) -> go.Figure:
    """DP do resíduo quantílico por região, uma barra por opção de dispersão (ideal = 1)."""
    fig = go.Figure([go.Bar(name=o, x=regs, y=dp.loc[o, regs].values) for o in opcoes])
    fig.add_hline(y=1, line_dash="dash", line_color="#888")
    fig.update_layout(
        barmode="group",
        title="DP do resíduo quantílico por região, por opção de dispersão (ideal = 1)",
        yaxis_title="DP",
        template="plotly_white",
        height=420,
    )
    return fig


def moran_contra_k(esf: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=esf.k_esf, y=esf.I_moran, mode="lines+markers", marker={"size": 9, "color": AZUL_FINAL})
    )
    fig.update_layout(
        title="I de Moran dos resíduos quantílicos contra o nº de autovetores (k)",
        xaxis_title="k",
        yaxis_title="I de Moran",
        template="plotly_white",
        height=400,
    )
    return fig


def correlograma_residuos(cg: pd.DataFrame, nome_modelo: str, corte_km: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(x=[f"{a:g}-{b:g}" for a, b in zip(cg.ini_km, cg.fim_km)], y=cg.correlacao, marker_color=AZUL_FINAL)
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color=LARANJA_REAL, annotation_text="limiar 0,05")
    fig.update_layout(
        title=f"Correlograma dos resíduos quantílicos ({nome_modelo}); corte = {corte_km:g} km",
        xaxis_title="faixa de distância (km)",
        yaxis_title="correlação média",
        template="plotly_white",
        height=420,
    )
    return fig


def regional_base_e_final(d: pd.DataFrame, base_pt: pd.Series, nome_modelo: str) -> go.Figure:
    """Share de Lula por região: ponto do bb_selecao, modelo final com IC95 de Conley e valor real."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base_pt.loc[d.nome].values,
            y=d.nome,
            mode="markers",
            name="bb_selecao (ponto)",
            marker={"color": "#888", "size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d.lula_pct,
            y=d.nome,
            mode="markers",
            name=f"{nome_modelo}, Conley (IC95)",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": d.q975 - d.lula_pct,
                "arrayminus": d.lula_pct - d.q025,
            },
            marker={"color": AZUL_FINAL, "size": 9},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d.real_pct,
            y=d.nome,
            mode="markers",
            name="real",
            marker={"color": LARANJA_REAL, "symbol": "diamond", "size": 10},
        )
    )
    fig.update_layout(
        title="Share de Lula por região (LOSO, σ_UF): bb_selecao e modelo final",
        xaxis_title="% dos votos válidos",
        template="plotly_white",
        height=460,
    )
    return fig


def erro_por_corte(tab: pd.DataFrame, anos: Sequence[int], coluna: str, rotulo_y: str, titulo: str) -> go.Figure:
    """Erro de cada método do veredito (`apuracao.METODOS`) por corte, um painel por ano, eixo y em log."""
    # os dois swings dividem a cor e se distinguem pelo traço: são o mesmo método com bases diferentes
    estilos = {
        "bruta": ("#888888", "solid"),
        "swing": (LARANJA_REAL, "solid"),
        "swing_1t": (LARANJA_REAL, "dash"),
        "modelo": (AZUL_FINAL, "solid"),
    }
    fig = make_subplots(rows=1, cols=2, subplot_titles=[str(a) for a in anos], shared_yaxes=True)
    for j, ano in enumerate(anos, 1):
        for met, (cor, traco) in estilos.items():
            t = tab[(tab.ano == ano) & (tab.metodo == met)]
            fig.add_trace(
                go.Scatter(
                    x=t.corte,
                    y=t[coluna],
                    mode="lines+markers",
                    name=met,
                    legendgroup=met,
                    showlegend=(j == 1),
                    line={"color": cor, "dash": traco},
                ),
                row=1,
                col=j,
            )
        fig.update_xaxes(title_text="% dos válidos recebidos", row=1, col=j)
    fig.update_yaxes(
        type="log",
        matches="y",
        tickvals=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        ticktext=["0,05", "0,1", "0,2", "0,5", "1", "2", "5"],
    )
    fig.update_yaxes(title_text=rotulo_y, row=1, col=1)
    fig.update_layout(title=titulo, template="plotly_white", height=420)
    return fig


# --------------------------------------------------------------------------
# Figuras para a apresentação (PNG estático): diagnóstico do MLG, Moran, apuração e mapas por UF
# --------------------------------------------------------------------------
CINZA_TEXTO = "#52514e"
CODIGO_UF = {
    11: "RO",
    12: "AC",
    13: "AM",
    14: "RR",
    15: "PA",
    16: "AP",
    17: "TO",
    21: "MA",
    22: "PI",
    23: "CE",
    24: "RN",
    25: "PB",
    26: "PE",
    27: "AL",
    28: "SE",
    29: "BA",
    31: "MG",
    32: "ES",
    33: "RJ",
    35: "SP",
    41: "PR",
    42: "SC",
    43: "RS",
    50: "MS",
    51: "MT",
    52: "GO",
    53: "DF",
}
UNIDADES_APURACAO = ("Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul")


def _linhas_h(fig: go.Figure, ys: Sequence[float], row: int, col: int) -> None:
    for y in ys:
        fig.add_hline(y=y, line_dash="dash", line_color=COR_REFERENCIA, line_width=1.2, row=row, col=col)


def painel_diagnostico(
    h: Any,
    cook: Any,
    residuo: Any,
    ajustado: Any,
    *,
    p: int,
    rotulos: Sequence[str] | None = None,
    titulo: str,
    rotulo_residuo: str = "Resíduo quantílico",
) -> go.Figure:
    """Painel 2x2 de diagnóstico (estilo Paula): alavanca, Cook e resíduo contra o índice, e resíduo contra π̂.

    Cortes de alavanca em 2p/n e 3p/n; Cook em 4/n (influente, merece olhar) e 1 (regra clássica: muito
    influente), com eixo em log porque poucos municípios grandes comprimem o resto em zero; resíduos fora de
    ±2 e ±3 merecem atenção; os 5 maiores Cook são rotulados.
    """
    h, cook = np.asarray(h, float), np.asarray(cook, float)
    residuo, ajustado = np.asarray(residuo, float), np.asarray(ajustado, float)
    n = len(h)
    idx = np.arange(1, n + 1)
    corte_cook = 4 / n
    cook_log = np.clip(cook, 1e-8, None)  # o eixo log descarta zeros
    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
        subplot_titles=(
            f"Alavanca h<sub>ii</sub> (linhas: 2p/n = {2 * p / n:.4f} e 3p/n = {3 * p / n:.4f})".replace(".", ","),
            f"Distância de Cook, escala log (linhas: 4/n = {corte_cook:.5f} e 1)".replace(".", ","),
            f"{rotulo_residuo} × índice",
            f"{rotulo_residuo} × π̂",
        ),
    )
    nomes_mun = list(rotulos) if rotulos is not None else [str(i) for i in idx]
    cor_cook = np.where(cook > corte_cook, COR_CRITICO, "#5b9be0")
    for (r, c), y in {(1, 1): h, (1, 2): cook_log, (2, 1): residuo, (2, 2): residuo}.items():
        x = ajustado if (r, c) == (2, 2) else idx
        marcador = {"color": cor_cook if (r, c) == (1, 2) else "#5b9be0", "size": 3}
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=marcador,
                showlegend=False,
                hovertext=nomes_mun,
                hovertemplate="%{hovertext}<br>x = %{x:.4g}<br>y = %{y:.4g}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    _linhas_h(fig, [2 * p / n, 3 * p / n], 1, 1)
    # Sem annotation_text: em eixo log o plotly posiciona o rótulo da hline no valor bruto, fora da linha.
    for valor in (corte_cook, 1.0):
        fig.add_hline(
            y=valor,
            line_dash="dash",
            line_color=COR_CRITICO if valor == 1 else COR_REFERENCIA,
            line_width=1.2,
            row=1,
            col=2,
        )
    fig.update_yaxes(type="log", exponentformat="power", row=1, col=2)
    _linhas_h(fig, [-3, -2, 2, 3], 2, 1)
    _linhas_h(fig, [-2, 2], 2, 2)
    topo = np.argsort(cook)[::-1][:5]
    fig.add_trace(
        go.Scatter(
            x=idx[topo],
            y=cook_log[topo],
            mode="markers+text",
            text=[nomes_mun[i] for i in topo],
            textposition=["top center", "top center", "bottom center", "top center", "bottom center"],
            textfont={"size": 10, "color": CINZA_TEXTO},
            cliponaxis=False,
            marker={"color": COR_CRITICO, "size": 6},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    suave = lowess(residuo, ajustado, frac=2 / 3, it=0)
    fig.add_trace(
        go.Scatter(
            x=suave[:, 0],
            y=suave[:, 1],
            mode="lines",
            line={"color": COR_CURVA, "width": 2.5},
            name="LOWESS",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="Índice do município", row=1, col=1)
    fig.update_xaxes(title_text="Índice do município", row=1, col=2)
    fig.update_xaxes(title_text="Índice do município", row=2, col=1)
    fig.update_xaxes(title_text="Proporção ajustada π̂", row=2, col=2)
    fig.update_layout(title=titulo, height=760, width=1000, margin={"t": 90})
    fig.update_annotations(font_size=12)
    return fig


def forest_or(tabela: pd.DataFrame, titulo: str) -> go.Figure:
    """Razões de chances com IC 95% em eixo log (1 = sem efeito), termos ordenados pela razão de chances."""
    t = tabela.sort_values("or").reset_index(drop=True)
    cores = [COR_CURVA if (a > 1 or b < 1) else COR_REFERENCIA for a, b in zip(t["ic_inf"], t["ic_sup"], strict=True)]
    fig = go.Figure(
        go.Scatter(
            x=t["or"],
            y=t["termo"],
            mode="markers",
            marker={"color": cores, "size": 8},
            showlegend=False,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": t["ic_sup"] - t["or"],
                "arrayminus": t["or"] - t["ic_inf"],
                "color": CINZA_TEXTO,
                "thickness": 1.3,
            },
            hovertemplate="%{y}<br>OR = %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=1, line_dash="dash", line_color=COR_REFERENCIA)
    fig.update_xaxes(type="log", title_text="Razão de chances (escala log, IC 95%)")
    fig.update_yaxes(categoryorder="array", categoryarray=list(t["termo"]), automargin=True)
    fig.update_layout(
        title=titulo,
        height=max(420, 26 * len(t) + 160),
        width=900,
        annotations=[
            {
                "text": "laranja = IC exclui 1",
                "xref": "paper",
                "yref": "paper",
                "x": 1,
                "y": 1.04,
                "showarrow": False,
                "xanchor": "right",
                "font": {"size": 10, "color": CINZA_TEXTO},
            }
        ],
    )
    return fig


def moran_dispersao(z: ArrayLike, lag_z: ArrayLike, indice_moran: float, titulo: str) -> go.Figure:
    """Diagrama de Moran: z contra a defasagem espacial; a reta passa pela origem com inclinação indice_moran.

    AA/BB = agrupamentos de valores altos/baixos; AB/BA = municípios discrepantes dos vizinhos.
    """
    z, lag = np.asarray(z, float), np.asarray(lag_z, float)
    sel = _amostra(len(z), 3000, seed=0)
    lim = float(np.ceil(max(np.abs(z).max(), np.abs(lag).max())))
    fig = go.Figure(
        go.Scatter(
            x=z[sel],
            y=lag[sel],
            mode="markers",
            name="Municípios",
            marker={"color": COR_PONTO, "size": 4, "opacity": 0.4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-lim, lim],
            y=[-indice_moran * lim, indice_moran * lim],
            mode="lines",
            name=f"Inclinação = I = {indice_moran:.3f}".replace(".", ","),
            line={"color": COR_CURVA, "width": 2.5},
        )
    )
    fig.add_hline(y=0, line_color=COR_REFERENCIA, line_width=1)
    fig.add_vline(x=0, line_color=COR_REFERENCIA, line_width=1)
    for texto, x, y, xa, ya in [
        ("AA", 0.98, 0.98, "right", "top"),
        ("BA", 0.02, 0.98, "left", "top"),
        ("BB", 0.02, 0.02, "left", "bottom"),
        ("AB", 0.98, 0.02, "right", "bottom"),
    ]:
        fig.add_annotation(
            text=f"<b>{texto}</b>",
            xref="x domain",
            yref="y domain",
            x=x,
            y=y,
            showarrow=False,
            xanchor=xa,
            yanchor=ya,
            font={"size": 16, "color": CINZA_TEXTO},
        )
    fig.update_xaxes(title_text="Variável padronizada z", range=[-lim, lim])
    fig.update_yaxes(title_text="Defasagem espacial Wz", range=[-lim, lim])
    fig.update_layout(title=titulo, height=620, width=700, legend={"orientation": "h", "y": -0.15})
    return fig


def mae_por_modelo(
    tab: pd.DataFrame, *, limite: float, escolhido: str, titulo: str, rotulo_limite: str = "Limite 1-EP"
) -> go.Figure:
    """MAE ± EP por modelo (colunas `modelo`, `mae`, `se`), do melhor ao pior; o limite 1-EP = melhor MAE + seu EP."""
    t = tab.sort_values("mae", ascending=False).reset_index(drop=True)
    cores = [
        COR_CURVA if m == escolhido else (COR_PONTO if v <= limite else COR_REFERENCIA)
        for m, v in zip(t["modelo"], t["mae"], strict=True)
    ]
    fig = go.Figure(
        go.Scatter(
            x=t["mae"],
            y=t["modelo"],
            mode="markers",
            showlegend=False,
            marker={"color": cores, "size": 9},
            error_x={"type": "data", "array": t["se"], "color": CINZA_TEXTO, "thickness": 1.3},
            hovertemplate="%{y}<br>MAE = %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=limite,
        line_dash="dash",
        line_color=COR_CRITICO,
        annotation_text=f"{rotulo_limite} = {limite:.3f}".replace(".", ","),
        annotation_position="top",
        annotation_font={"size": 11, "color": COR_CRITICO},
    )
    fig.update_xaxes(title_text="MAE (p.p.)")
    fig.update_yaxes(categoryorder="array", categoryarray=list(t["modelo"]), automargin=True)
    fig.add_annotation(
        text=f"laranja = {escolhido} (escolhido); azul = dentro do limite; cinza = acima",
        xref="paper",
        yref="paper",
        x=1,
        y=0,
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font={"size": 11, "color": COR_CURVA},
    )
    fig.update_layout(title=titulo, height=max(420, 30 * len(t) + 170), width=900, margin={"t": 100})
    return fig


def estimativa_por_corte(ic: pd.DataFrame, titulo: str, ano: int | None = None) -> go.Figure:
    """Estimativa com IC 95% por corte de apuração contra o resultado final, uma faceta por unidade (Brasil e regiões)."""
    t = ic if ano is None or "ano" not in ic else ic[ic["ano"] == ano]
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=UNIDADES_APURACAO, vertical_spacing=0.18, horizontal_spacing=0.07
    )
    for k, un in enumerate(UNIDADES_APURACAO):
        r, c = divmod(k, 3)
        d = t[t["regiao"] == un].sort_values("pct_efetivo")
        if d.empty:
            continue
        mostra = k == 0
        fig.add_trace(
            go.Scatter(
                x=d["pct_efetivo"],
                y=d["estimativa_pct"],
                mode="markers",
                name="Estimativa (IC 95%)",
                legendgroup="e",
                showlegend=mostra,
                marker={"color": AZUL_FINAL, "size": 8},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": d["q975_pct"] - d["estimativa_pct"],
                    "arrayminus": d["estimativa_pct"] - d["q025_pct"],
                    "color": AZUL_FINAL,
                },
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.add_hline(
            y=float(d["real_pct"].iloc[0]),
            line_dash="dash",
            line_color=LARANJA_REAL,
            line_width=2,
            row=r + 1,
            col=c + 1,
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Resultado final",
            line={"color": LARANJA_REAL, "dash": "dash", "width": 2},
        )
    )
    fig.update_xaxes(title_text="% da apuração", row=2)
    fig.update_yaxes(title_text="% Lula", col=1)
    fig.update_layout(
        title=titulo,
        height=640,
        width=1000,
        margin={"t": 90},
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
    )
    return fig


def curva_apuracao(curva: pd.DataFrame, estimativas: pd.DataFrame | None, real: float, titulo: str) -> go.Figure:
    """Share bruto contado (`pct_recebido`, `share_lula`) contra o % apurado, estimativas do modelo e resultado final."""
    passo = max(1, len(curva) // 1500)
    c = curva.iloc[::passo]
    fig = go.Figure(
        go.Scatter(
            x=c["pct_recebido"],
            y=c["share_lula"],
            mode="lines",
            name="Apuração bruta",
            line={"color": COR_REFERENCIA, "width": 2},
        )
    )
    if estimativas is not None and len(estimativas):
        e = estimativas.sort_values("pct_efetivo")
        erro = (
            {
                "type": "data",
                "symmetric": False,
                "array": e["q975_pct"] - e["estimativa_pct"],
                "arrayminus": e["estimativa_pct"] - e["q025_pct"],
                "color": AZUL_FINAL,
            }
            if {"q025_pct", "q975_pct"} <= set(e.columns)
            else None
        )
        fig.add_trace(
            go.Scatter(
                x=e["pct_efetivo"],
                y=e["estimativa_pct"],
                mode="markers",
                name="Modelo (IC 95%)",
                marker={"color": AZUL_FINAL, "size": 9},
                error_y=erro,
            )
        )
    fig.add_hline(
        y=real,
        line_dash="dash",
        line_color=LARANJA_REAL,
        line_width=2,
        annotation_text=f"Resultado final = {real:.2f}%".replace(".", ","),
        annotation_position="top right",
        annotation_font={"color": LARANJA_REAL},
    )
    fig.update_xaxes(title_text="% das seções apuradas", range=[0, 100])
    fig.update_yaxes(title_text="% Lula nos votos válidos")
    fig.update_layout(title=titulo, height=520, width=900, legend={"orientation": "h", "y": -0.2})
    return fig


def _sem_buracos(geom: Any) -> Any:
    # O dissolve deixa frestas entre municípios; sem tirá-las o mapa mostra falhas brancas dentro da UF.
    from shapely.geometry import MultiPolygon, Polygon

    partes = [Polygon(pl.exterior) for pl in getattr(geom, "geoms", [geom]) if isinstance(pl, Polygon)]
    return MultiPolygon(partes)


@functools.cache
def _geometria_uf(tolerancia: float = 0.03) -> tuple[dict[str, Any], tuple[tuple[str, float, float], ...]]:
    """GeoJSON por UF (dissolve dos municípios) e um ponto de rótulo por UF; o dissolve é caro, por isso o cache."""
    import geopandas as gpd

    pasta = Path(__file__).resolve().parent.parent / "data" / "processed" / "municipios_geo.parquet"
    g = gpd.read_parquet(pasta)
    g["uf"] = g["id_municipio"].astype(str).str[:2].astype(int).map(CODIGO_UF)
    g["geometry"] = g.geometry.simplify(
        tolerancia / 3, preserve_topology=True
    ).make_valid()  # geometrias inválidas quebram o dissolve
    uf = g.dissolve(by="uf").reset_index()
    uf["geometry"] = [
        _orientada(_sem_buracos(geom.simplify(tolerancia, preserve_topology=True))) for geom in uf.geometry
    ]
    feats = [
        {"type": "Feature", "id": s, "properties": {}, "geometry": _arredonda(geom.__geo_interface__, 3)}
        for s, geom in zip(uf["uf"], uf.geometry, strict=True)
    ]
    pts = uf.geometry.representative_point()
    rotulos = tuple((s, float(p.x), float(p.y)) for s, p in zip(uf["uf"], pts, strict=True))
    return {"type": "FeatureCollection", "features": feats}, rotulos


def mapa_uf(
    valores: pd.Series,
    titulo: str,
    rotulo: str = "",
    divergente: bool = False,
    lim: float | None = None,
    formato: str = ".1f",
    rotular: bool = True,
) -> go.Figure:
    """Coroplético por UF (índice = sigla). `divergente=True` centra a escala em 0 (erros com sinal)."""
    gj, pontos_rotulo = _geometria_uf(tolerancia=0.03)
    v = valores.astype(float)
    kw: dict[str, Any] = {}
    if divergente:
        m = lim if lim is not None else float(np.nanmax(np.abs(v)))
        kw = {"zmid": 0, "zmin": -m, "zmax": m, "colorscale": ESCALA_DIVERGENTE}
    else:
        kw = {"colorscale": [[0, "#e8f0fb"], [1, "#184f95"]]}
    fig = go.Figure(
        go.Choropleth(
            geojson=gj,
            locations=list(v.index),
            z=v.values,
            marker_line_color="#ffffff",
            marker_line_width=0.8,
            colorbar={"title": rotulo, "thickness": 14, "len": 0.75, "x": 0.96},
            hovertemplate="%{location}: %{z:" + formato + "}<extra></extra>",
            **kw,
        )
    )
    if rotular:
        pts = [(s, x, y) for s, x, y in pontos_rotulo if s in v.index]
        fig.add_trace(
            go.Scattergeo(
                lon=[p[1] for p in pts],
                lat=[p[2] for p in pts],
                mode="text",
                text=[p[0] for p in pts],
                textfont={"size": 10, "color": "#0b0b0b"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#ffffff", projection_type="mercator")
    fig.update_layout(title=titulo, height=520, width=520, margin={"t": 60, "l": 0, "r": 0, "b": 0})
    return fig


def serie_regional(tab: pd.DataFrame, titulo: str) -> go.Figure:
    """Proporção do PT por região ao longo das eleições (colunas `regiao`, `ano`, `prop_pt`), com a linha de 50%."""
    fig = go.Figure()
    for reg, cor in CORES_REGIAO.items():
        d = tab[tab["regiao"] == reg].sort_values("ano")
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["ano"],
                y=d["prop_pt"],
                mode="lines+markers",
                name=reg,
                line={"color": cor, "width": 2.5},
                marker={"size": 9},
            )
        )
    fig.add_hline(y=0.5, line_dash="dash", line_color=COR_REFERENCIA)
    anos = sorted(tab["ano"].unique())
    fig.update_xaxes(title_text="Ano da eleição", tickmode="array", tickvals=anos, ticktext=[str(a) for a in anos])
    fig.update_yaxes(title_text="Proporção do PT (2º turno)", tickformat=".0%")
    fig.update_layout(title=titulo, height=480, width=860)
    return fig


# --- AED e apuração ---
def psi_por_nivel(
    tab: pd.DataFrame, titulo: str, rotulo_x: str = "Razão de chances ψ contra a referência"
) -> go.Figure:
    """Pontos de ψ com IC 95% por nível (colunas `psi`, `psi_inf95`, `psi_sup95`), em escala log e do menor ao maior ψ.

    A referência (ψ = 1, sem IC) aparece como ponto cinza; os IC ignoram a superdispersão.
    """
    t = tab.sort_values("psi")
    ref = t["psi_inf95"].isna()
    fig = go.Figure()
    com_ic = t[~ref]
    fig.add_trace(
        go.Scatter(
            x=com_ic["psi"],
            y=com_ic.index,
            mode="markers+text",
            name="ψ (IC 95%)",
            text=[f"{v:.2f}".replace(".", ",") for v in com_ic["psi"]],
            textposition="middle right",
            textfont={"size": 9, "color": CINZA_TEXTO},
            marker={"color": COR_PONTO, "size": 8},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": com_ic["psi_sup95"] - com_ic["psi"],
                "arrayminus": com_ic["psi"] - com_ic["psi_inf95"],
                "color": COR_PONTO,
                "thickness": 1.2,
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t.loc[ref, "psi"],
            y=t.index[ref],
            mode="markers",
            name="Referência (ψ = 1)",
            marker={"color": COR_REFERENCIA, "size": 9, "symbol": "diamond"},
        )
    )
    fig.add_vline(x=1, line_dash="dash", line_color=COR_REFERENCIA)
    marcas = [v for v in (0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6) if t["psi"].min() * 0.8 <= v <= t["psi"].max() * 1.25]
    fig.update_xaxes(
        title_text=rotulo_x,
        type="log",
        tickmode="array",
        tickvals=marcas,
        ticktext=[str(v).replace(".", ",") for v in marcas],
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(t.index), automargin=True, dtick=1)
    fig.update_layout(
        title=titulo, height=max(420, 22 * len(t) + 190), width=800, legend={"orientation": "h", "y": -0.2}
    )
    return fig


def representatividade_por_corte(rep: pd.DataFrame, titulo: str) -> go.Figure:
    """Lacuna absoluta (p.p.) entre o share apurado, ou o π̂ do modelo, e o share real do restante, por corte, ano e grupo.

    Espera as colunas de `apuracao.representatividade` mais `ano` e `corte`; uma faceta por ano.
    """
    anos = sorted(rep["ano"].unique())
    fig = make_subplots(
        rows=1, cols=len(anos), subplot_titles=[str(a) for a in anos], shared_yaxes=True, horizontal_spacing=0.05
    )
    estilos = [
        ("apurado", "abs_apurado_pp", COR_REFERENCIA, "Share apurado"),
        ("modelo", "abs_modelo_pp", COR_PONTO, "π̂ do modelo"),
    ]
    for j, ano in enumerate(anos, start=1):
        for grupo, traco in (("maiores", "solid"), ("demais", "dot")):
            d = rep[(rep["ano"] == ano) & (rep["grupo"] == grupo)].sort_values("corte")
            for _, col, cor, nome in estilos:
                rotulo = f"{nome}, {'20 maiores' if grupo == 'maiores' else 'demais'}"
                fig.add_trace(
                    go.Scatter(
                        x=d["corte"],
                        y=d[col],
                        mode="lines+markers",
                        name=rotulo,
                        legendgroup=rotulo,
                        showlegend=j == 1,
                        line={"color": cor, "dash": traco, "width": 2.5},
                        marker={"size": 7},
                    ),
                    row=1,
                    col=j,
                )
    cortes = sorted(rep["corte"].unique())
    fig.update_xaxes(
        title_text="Corte de apuração (%)", tickmode="array", tickvals=cortes, ticktext=[str(c) for c in cortes]
    )
    fig.update_yaxes(title_text="Lacuna absoluta (p.p.)", col=1)
    fig.update_layout(
        title=titulo,
        height=480,
        width=1000,
        margin={"t": 90},
        legend={"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"},
    )
    return fig
