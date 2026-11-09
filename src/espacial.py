"""Ferramentas espaciais: filtro de autovetores de Moran, erros-padrão de Conley,
correlograma e I de Moran dos resíduos.

Motivo: os resíduos de todos os modelos têm autocorrelação espacial positiva
(I de Moran ~0,3), então os erros-padrão baseados no modelo são pequenos demais.
Referências: Griffith (2003); Tiefelsdorf & Griffith (2007); Conley (1999).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.spatial import cKDTree

R_TERRA_KM = 6371.0088


def pesos_queen(geo):
    """Pesos Queen padronizados por linha; ilhas ligadas ao vizinho mais próximo (KNN k=1).

    `geo` deve estar na ordem das observações.
    """
    from libpysal.weights import KNN, Queen
    from libpysal.weights.util import attach_islands

    w = Queen.from_dataframe(geo, use_index=False)
    if w.islands:
        w = attach_islands(w, KNN.from_dataframe(geo, k=1))
    w.transform = "r"
    return w


def defasagem_espacial(valores, w):
    """W·x com os pesos `w` (padronizados por linha) na ordem das observações de `w`.

    Com `pesos_queen` não há linha vazia (ilhas já foram ligadas ao vizinho mais
    próximo): a ilha recebe o valor desse vizinho. Os covariáveis são conhecidos em
    todos os municípios, então calcular uma vez nos dados completos não vaza resposta.
    """
    x = np.asarray(valores, float)
    if x.shape[0] != w.n:
        raise ValueError(f"valores tem {x.shape[0]} linhas e w tem {w.n}")
    return np.asarray(w.sparse @ x).ravel()


def adicionar_slx(df, mapa, chaves, w):
    """Devolve (df2, mapa2) com a coluna W·x de cada chave e mapa2['w_<chave>'].

    A defasagem é tomada na escala em que o termo entra no modelo: para `pib_pc`,
    W·log(pib), como em `_coluna`. Não altera as entradas.
    """
    from src.especificacoes import _coluna

    df2, mapa2 = df.copy(), dict(mapa)
    for k in chaves:
        col = f"w_{mapa[k]}"
        df2[col] = defasagem_espacial(_coluna(df, mapa, k), w)
        mapa2[f"w_{k}"] = col
    return df2, mapa2


def wald_robusto(params, cov, indices, gl_den=None):
    """Wald de H0: params[indices] = 0 com a covariância `cov` dada (ex.: Conley).

    Devolve (estatística, gl, p): qui-quadrado com q = len(indices) gl; com `gl_den`,
    F = qui2/q com (q, gl_den) gl.
    """
    from scipy import stats

    idx = np.atleast_1d(np.asarray(indices, int))
    b = np.asarray(params, float)[idx]
    V = np.asarray(cov, float)[np.ix_(idx, idx)]
    q = len(idx)
    chi2 = float(b @ np.linalg.solve(V, b))
    if gl_den is None:
        return chi2, q, float(stats.chi2.sf(chi2, q))
    f = chi2 / q
    return f, q, float(stats.f.sf(f, q, gl_den))


def matriz_contiguidade(geo):
    """Matriz binária simétrica C (esparsa, diagonal nula): Queen + ilhas por KNN k=1.

    A ligação das ilhas via KNN pode ser assimétrica; simetriza-se C = max(C, C').
    """
    w = pesos_queen(geo)
    w.transform = "b"
    C = w.sparse.tocsr().astype(float)
    C = C.maximum(C.T).tolil()
    C.setdiag(0)
    C = C.tocsr()
    C.eliminate_zeros()
    return C


def autovetores_esf(geo, k=100):
    """Autovetores do filtro espacial de Moran (Griffith 2003).

    Autovetores de M C M, M = I - 11'/N, com os k maiores autovalores (I de Moran
    positivo, padrões de grande escala primeiro). I_j = N/(1'C1) * lambda_j.
    Ordenados por I decrescente, centrados, com desvio padrão 1 (ddof=0) e sinal
    determinístico (maior |valor| positivo).

    Depende só da geometria, nunca de y: por isso é legítimo usá-lo ao prever uma
    UF deixada de fora (não há vazamento de resposta).

    Retorna (DataFrame N x k com colunas ev_001.., Series com o I de Moran de cada um).
    """
    C = matriz_contiguidade(geo)
    N = C.shape[0]
    s = C.sum()

    def mv(v):
        x = np.asarray(v, dtype=float).reshape(N)
        x = x - x.mean()
        y = C @ x
        return y - y.mean()

    op = LinearOperator((N, N), matvec=mv, dtype=float)
    lam, vec = eigsh(op, k=k, which="LA", ncv=min(N, max(2 * k + 1, 20)),
                     v0=np.random.default_rng(0).normal(size=N))
    ordem = np.argsort(-lam)
    lam, vec = lam[ordem], vec[:, ordem]
    vec = vec - vec.mean(axis=0)
    vec = vec / vec.std(axis=0, ddof=0)
    sinal = np.sign(vec[np.abs(vec).argmax(axis=0), np.arange(vec.shape[1])])
    vec = vec * sinal
    nomes = [f"ev_{j:03d}" for j in range(1, k + 1)]
    moran = pd.Series(N / s * lam, index=nomes, name="I_moran")
    return pd.DataFrame(vec, index=geo.index, columns=nomes), moran


def _vetores_unitarios(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _pares_ate(lat, lon, dist_km):
    """Pares (i, j, d_km), incluindo i=j e ambas as direções, com d <= dist_km."""
    xyz = _vetores_unitarios(lat, lon)
    arvore = cKDTree(xyz)
    corda = 2.0 * np.sin(min(dist_km, np.pi * R_TERRA_KM) / (2 * R_TERRA_KM))
    p = arvore.sparse_distance_matrix(arvore, corda, output_type="ndarray")
    d = 2 * R_TERRA_KM * np.arcsin(np.clip(p["v"] / 2, 0, 1))
    return p["i"].astype(int), p["j"].astype(int), d


def _score_hessiana(resultado, score_obs, hessian):
    modelo, params = resultado.model, np.asarray(resultado.params)
    glm = hasattr(modelo, "family")
    if score_obs is None:
        score_obs = modelo.score_obs(params, scale=1.0) if glm else modelo.score_obs(params)
    if hessian is None:
        hessian = modelo.hessian(params, scale=1.0) if glm else modelo.hessian(params)
    return np.asarray(score_obs, float), np.asarray(hessian, float)


def _nomes(resultado, k):
    p = resultado.params
    return list(p.index) if hasattr(p, "index") else list(range(k))


def _sanduiche(S, H, meat_K):
    Hinv = np.linalg.inv(H)
    meio = S.T @ meat_K
    cov = Hinv @ meio @ Hinv
    return (cov + cov.T) / 2


def cov_conley(resultado, lat, lon, corte_km, score_obs=None, hessian=None):
    """Covariância espacial-HAC de Conley (1999): H^-1 (S' K S) H^-1.

    S = escores por observação, H = hessiana da log-verossimilhança no ótimo (binomial:
    escala 1). Núcleo de Bartlett K_ij = max(0, 1 - d_ij/corte), d = distância de
    círculo máximo, K_ii = 1. Sem fator de pequena amostra (com corte ~ 0 reproduz HC0
    puro, sem N/(N-k)). `score_obs`/`hessian` opcionais substituem os do modelo.
    """
    S, H = _score_hessiana(resultado, score_obs, hessian)
    N = S.shape[0]
    i, j, d = _pares_ate(lat, lon, corte_km)
    K = sp.csr_matrix((np.maximum(0.0, 1.0 - d / corte_km), (i, j)), shape=(N, N))
    cov = _sanduiche(S, H, K @ S)
    nomes = _nomes(resultado, S.shape[1])
    return pd.DataFrame(cov, index=nomes, columns=nomes)


def cov_cluster(resultado, grupos, correcao=True, score_obs=None, hessian=None):
    """Covariância robusta por conglomerado: núcleo K = indicador de mesmo grupo.

    Com `correcao`, aplica G/(G-1) * (N-1)/(N-k) como o statsmodels.
    """
    S, H = _score_hessiana(resultado, score_obs, hessian)
    N, k = S.shape
    cod, uniq = pd.factorize(np.asarray(grupos))
    G = len(uniq)
    soma = np.zeros((G, k))
    np.add.at(soma, cod, S)
    Hinv = np.linalg.inv(H)
    cov = Hinv @ (soma.T @ soma) @ Hinv
    if correcao:
        cov = cov * G / (G - 1) * (N - 1) / (N - k)
    nomes = _nomes(resultado, k)
    return pd.DataFrame((cov + cov.T) / 2, index=nomes, columns=nomes)


def correlograma(residuos, lat, lon, faixas_km):
    """Correlação espacial dos resíduos por faixa de distância.

    `faixas_km` são as bordas crescentes; a faixa é [ini, fim). Resíduos
    padronizados (média 0, desvio 1, ddof=0); correlação = média de z_i z_j sobre
    os pares i<j da faixa.
    """
    z = np.asarray(residuos, float)
    z = (z - z.mean()) / z.std()
    bordas = np.asarray(faixas_km, float)
    i, j, d = _pares_ate(lat, lon, bordas[-1])
    m = i < j
    i, j, d = i[m], j[m], d[m]
    prod = z[i] * z[j]
    linhas = []
    for ini, fim in zip(bordas[:-1], bordas[1:]):
        sel = (d >= ini) & (d < fim)
        n = int(sel.sum())
        linhas.append((ini, fim, n, prod[sel].mean() if n else np.nan))
    return pd.DataFrame(linhas, columns=["ini_km", "fim_km", "pares", "correlacao"])


def moran_residuos(residuos, w, permutacoes=999, semente=0):
    """I de Moran dos resíduos (esda) com p-valor por permutação; semente fixa."""
    import esda

    estado = np.random.get_state()
    np.random.seed(semente)
    try:
        mi = esda.Moran(np.asarray(residuos, float), w, permutations=permutacoes)
    finally:
        np.random.set_state(estado)
    return {"I": float(mi.I), "p_sim": float(mi.p_sim)}
