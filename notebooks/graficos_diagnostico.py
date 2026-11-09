"""Gráficos plotly de diagnóstico do notebook 02 (MLG). Salvos em results/figures/mlg_*.html."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy import stats


def salvar(fig, caminho):
    fig.write_html(str(caminho), include_plotlyjs="cdn", div_id=Path(caminho).stem)
    return fig


def qq_normal(residuos: dict, titulo="QQ-plot dos resíduos quantílicos"):
    """Uma curva por modelo; sob o modelo correto os pontos ficam sobre a diagonal."""
    fig = go.Figure()
    for nome, r in residuos.items():
        r = np.sort(np.asarray(r, float))
        q = stats.norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
        passo = max(1, len(r) // 1500)
        fig.add_scatter(x=q[::passo], y=r[::passo], mode="markers", name=nome, marker=dict(size=4))
    lim = [-4, 4]
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="gray", dash="dash"), showlegend=False)
    fig.update_layout(title=titulo, xaxis_title="quantil normal teórico",
                      yaxis_title="resíduo quantílico", template="plotly_white")
    return fig


def forest_coeficientes(tabelas: dict, variaveis=None, titulo="Coeficientes (IC 95%)"):
    """Coeficientes padronizados com IC 95% lado a lado por modelo."""
    fig = go.Figure()
    for nome, t in tabelas.items():
        t = t if variaveis is None else t.loc[[v for v in variaveis if v in t.index]]
        t = t[t.index != "const"]
        fig.add_scatter(x=t["coef"], y=list(t.index), mode="markers", name=nome,
                        error_x=dict(type="data", array=1.96 * t["ep"], visible=True))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=titulo, xaxis_title="coeficiente (logit, covariável em z-score)",
                      template="plotly_white", height=700)
    return fig


# --------------------------------------------------------------------------
# Parte 10 - Diagnóstico
# --------------------------------------------------------------------------
def _amostra(n, maximo=1500, seed=0):
    """Índices para desenhar no máximo `maximo` pontos (a figura fica leve)."""
    if n <= maximo:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, maximo, replace=False))


def _medias_por_faixa(x, r, faixas=20):
    """Média do resíduo em faixas de x com o mesmo número de pontos (curva de tendência)."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    ordem = np.argsort(x)
    grupos = np.array_split(ordem, faixas)
    return (np.array([x[g].mean() for g in grupos]), np.array([r[g].mean() for g in grupos]))


def meio_normal(envelopes: dict, titulo="Gráfico half-normal com envelope simulado"):
    """Um painel por modelo: |resíduo| ordenado contra quantis half-normal e banda simulada."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(envelopes), subplot_titles=list(envelopes))
    for j, (nome, e) in enumerate(envelopes.items(), start=1):
        idx = _amostra(len(e["observado"]), 1200)
        x = e["teorico"][idx]
        for chave, cor, dash in [("inferior", "gray", "dash"), ("superior", "gray", "dash"),
                                 ("mediana", "gray", "dot")]:
            fig.add_scatter(x=x, y=e[chave][idx], mode="lines", line=dict(color=cor, dash=dash, width=1),
                            showlegend=False, row=1, col=j)
        fig.add_scatter(x=x, y=e["observado"][idx], mode="markers", marker=dict(size=4, color="#1f77b4"),
                        showlegend=False, row=1, col=j)
        fig.update_xaxes(title_text="quantil half-normal", row=1, col=j)
    fig.update_yaxes(title_text="|resíduo quantílico| ordenado", row=1, col=1)
    fig.update_layout(title=titulo, template="plotly_white")
    return fig


def residuos_contra(x, residuos: dict, xtitulo, titulo, faixas=20):
    """Painéis (um por modelo) de resíduo contra `x`, com a média por faixa em vermelho."""
    from plotly.subplots import make_subplots
    x = np.asarray(x, float)
    fig = make_subplots(rows=1, cols=len(residuos), subplot_titles=list(residuos), shared_yaxes=True)
    for j, (nome, r) in enumerate(residuos.items(), start=1):
        r = np.asarray(r, float)
        idx = _amostra(len(r), 1200)
        fig.add_scatter(x=x[idx], y=r[idx], mode="markers", marker=dict(size=3, opacity=0.35),
                        showlegend=False, row=1, col=j)
        mx, mr = _medias_por_faixa(x, r, faixas)
        fig.add_scatter(x=mx, y=mr, mode="lines+markers", line=dict(color="crimson"),
                        showlegend=False, row=1, col=j)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=j)
        fig.update_xaxes(title_text=xtitulo, row=1, col=j)
    fig.update_layout(title=titulo, template="plotly_white", height=420)
    return fig


def caixa_regiao(residuos: dict, regiao, titulo="Resíduos quantílicos por região"):
    """Box plot do resíduo por região, um traço por modelo."""
    fig = go.Figure()
    regiao = np.asarray(regiao).astype(str)
    for nome, r in residuos.items():
        fig.add_box(x=regiao, y=np.asarray(r, float), name=nome, boxpoints=False)
    fig.update_layout(title=titulo, boxmode="group", template="plotly_white",
                      yaxis_title="resíduo quantílico")
    return fig


def alavanca_cook(h, cook, rotulos, p, titulo="Alavanca e distância de Cook (quase-binomial)"):
    """Dispersão alavanca x Cook; linhas em 2p/n (alavanca) e 4/n (Cook)."""
    h, cook = np.asarray(h, float), np.asarray(cook, float)
    n = len(h)
    fig = go.Figure(go.Scattergl(x=h, y=cook, mode="markers", text=list(rotulos),
                                 marker=dict(size=4, opacity=0.5)))
    fig.add_vline(x=2 * p / n, line_dash="dash", line_color="gray")
    fig.add_hline(y=4 / n, line_dash="dash", line_color="gray")
    fig.update_layout(title=titulo, xaxis_title="alavanca h", yaxis_title="distância de Cook",
                      yaxis_type="log", template="plotly_white")
    return fig


def mapa_residuos(gdf, residuo, titulo, tolerancia=0.05):
    """Coroplético de resíduos; geometria simplificada (graus) para o HTML ficar leve."""
    g = gdf.copy()
    g["geometry"] = g.geometry.simplify(tolerancia, preserve_topology=True)
    g = g.reset_index(drop=True)
    fig = go.Figure(go.Choropleth(geojson=g.geometry.__geo_interface__, locations=g.index.astype(str),
                                  z=np.clip(np.asarray(residuo, float), -4, 4), zmin=-4, zmax=4,
                                  colorscale="RdBu", marker_line_width=0,
                                  colorbar=dict(title="resíduo")))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title=titulo, template="plotly_white", height=650)
    return fig


def residuos_quantilicos_rapido(y, n, pi, theta=None, rng=None, base=400, c=32):
    """Resíduo quantílico aleatorizado (Dunn & Smyth) com a FDA beta-binomial por quadratura.

    A `scipy.stats.betabinom.cdf` soma a pmf termo a termo (~9 s para 5.570 municípios).
    Aqui F(k) = E_p[Binomial(k; n, p)], p ~ Beta(pi*theta, (1-pi)*theta), integrada nos
    quantis do meio da beta. A binomial é um degrau de largura sqrt(p(1-p)/n) em p; por isso o
    número de pontos cresce com n: J = max(base, c*sqrt(n/theta)), arredondado para base*2^k.
    Erro máximo da FDA ~2e-4 (conferido contra a scipy no notebook). theta=None: binomial.
    """
    rng = np.random.default_rng(rng)
    y, n = np.asarray(y).astype(np.int64), np.asarray(n).astype(np.int64)
    pi = np.clip(np.asarray(pi, float), 1e-9, 1 - 1e-9)
    if theta is None:
        F = lambda k: stats.binom.cdf(k, n, pi)
    else:
        theta = np.asarray(theta, float)  # escalar ou um theta por linha
        J = np.maximum(base, np.ceil(c * np.sqrt(n / theta))).astype(int)
        nivel = base * 2 ** np.ceil(np.log2(J / base)).astype(int)

        def F(k):
            out = np.empty(len(k))
            for Jb in np.unique(nivel):
                i = np.where(nivel == Jb)[0]
                th = theta if theta.ndim == 0 else theta[i]
                p = stats.beta.ppf(((np.arange(Jb) + 0.5) / Jb)[:, None], pi[i] * th, (1 - pi[i]) * th)
                out[i] = stats.binom.cdf(k[i][None, :], n[i][None, :], p).mean(axis=0)
            return out
    lo, hi = F(y - 1), F(y)
    u = np.clip(lo + rng.uniform(size=y.shape) * (hi - lo), 1e-10, 1 - 1e-10)
    return stats.norm.ppf(u)
