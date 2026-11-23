"""Ferramentas espaciais: filtro de autovetores de Moran, erros-padrão de Conley,
correlograma e I de Moran dos resíduos.

Motivo: os resíduos de todos os modelos têm autocorrelação espacial positiva
(I de Moran ~0,3), então os erros-padrão baseados no modelo são pequenos demais.
Referências: Griffith (2003); Tiefelsdorf & Griffith (2007); Conley (1999).
"""

import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from scipy.spatial import KDTree

if TYPE_CHECKING:
    import numpy.typing as npt
    from libpysal.weights import W

    from src.especificacoes import Ajuste

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

    op = LinearOperator((N, N), matvec=mv, dtype=float)  # ty: ignore[missing-argument, unknown-argument, parameter-already-assigned]  # stub do scipy incompleto
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
    arvore = KDTree(xyz)
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
    return {"I": float(mi.I), "p_sim": float(mi.p_sim), "esperado": float(mi.EI), "z_sim": float(mi.z_sim)}


def _restringir_pesos(w, grupos, dentro):
    """Submatriz de `w` só com vínculos dentro (ou entre) grupos, padronizada por linha.

    Devolve (matriz, mantidas): linhas sem vínculo saem; a matriz só tem as mantidas.
    """
    g = np.asarray(grupos)
    A = sp.coo_matrix(w.sparse)
    ok = (g[A.row] == g[A.col]) if dentro else (g[A.row] != g[A.col])
    A = sp.csr_matrix((A.data[ok], (A.row[ok], A.col[ok])), shape=A.shape)
    mantidas = np.arange(A.shape[0])
    while True:
        A = A[mantidas][:, mantidas]
        soma = np.asarray(A.sum(axis=1)).ravel()
        if (soma > 0).all():
            break
        mantidas = mantidas[soma > 0]
    return sp.diags(1.0 / soma) @ A, mantidas


def _moran_i(z, W):
    z = np.asarray(z, float)
    z = z - z.mean()
    return float(z @ (W @ z) / (z @ z))


def moran_decomposto(residuos, w, grupos):
    """I de Moran dos resíduos: W completo, só vínculos dentro do grupo, só entre grupos,
    e W completo depois de subtrair a média do grupo. Pesos padronizados por linha.

    Motivo: separar a correlação que vive dentro das UFs da que atravessa a divisa
    e da que é só um choque comum de UF. Devolve também o nº de linhas de cada W.
    """
    r = pd.Series(np.asarray(residuos, float))
    g = np.asarray(grupos)
    out = {"completo": _moran_i(r, sp.csr_matrix(w.sparse)), "n_completo": len(r)}
    for nome, dentro in (("dentro_uf", True), ("entre_uf", False)):
        W, mant = _restringir_pesos(w, g, dentro)
        out[nome], out[f"n_{nome}"] = _moran_i(r.to_numpy()[mant], W), len(mant)
    sem = (r - r.groupby(g).transform("mean")).to_numpy()
    out["completo_sem_media_uf"], out["n_completo_sem_media_uf"] = _moran_i(sem, sp.csr_matrix(w.sparse)), len(r)
    return out


def sd_betabinomial(pi: np.ndarray, n: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Desvio-padrão da proporção y/n no beta-binomial: sqrt(pi (1-pi) [1 + (n-1) theta] / n).

    Motivo: a variância do resíduo de resposta é cerca de 1/n, então resíduos sem
    padronizar dão peso desigual aos municípios na krigagem, no CAR e no I de Moran.
    """
    p = np.clip(np.asarray(pi, float), 1e-9, 1 - 1e-9)
    n = np.asarray(n, float)
    return np.sqrt(p * (1 - p) * (1 + (n - 1) * np.asarray(theta, float)) / n)


def residuo_pearson(y: np.ndarray, n: np.ndarray, pi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resíduo de Pearson beta-binomial z = (y/n - pi) / sd e o próprio sd (escala da proporção)."""
    sd = sd_betabinomial(pi, n, theta)
    return (np.asarray(y, float) / np.asarray(n, float) - np.asarray(pi, float)) / sd, sd


def krigagem_residuos(r_treino, lat, lon, treino, teste, grupos, corte_km, sd_teste=None):
    """Krigagem simples dos resíduos de treino com núcleo de Bartlett: (rho, k_teste).

    `lat`, `lon`, `grupos` cobrem todas as linhas; `treino` e `teste` são índices e
    `r_treino` segue a ordem de `treino`. k_i = soma K(d_ij) r_j / soma K(d_ij) sobre o
    treino, K = 1 - d/corte para d < corte (0 sem vizinho). rho é a inclinação sem
    intercepto de r_i em k_i, com k_i do treino calculado só com municípios de outros
    grupos (como o teste vê só outras UFs), cortada em [0, 1]. Com `sd_teste` (um por
    município de teste), r_treino é padronizado e k_teste volta na escala do desvio-padrão
    (k_i x sd_i); sem ele, nada muda.
    """
    tr, te = np.asarray(treino), np.asarray(teste)
    r = np.asarray(r_treino, float)
    todos = np.concatenate([tr, te])
    i, j, d = _pares_ate(np.asarray(lat)[todos], np.asarray(lon)[todos], corte_km)
    ntr = len(tr)
    k = np.clip(1.0 - d / corte_km, 0.0, None)
    g = np.asarray(grupos)[todos]
    vale = (k > 0) & (j < ntr) & ((i >= ntr) | (g[i] != g[j]))   # vizinho do treino; treino só de outros grupos
    K = sp.csr_matrix((k[vale], (i[vale], j[vale])), shape=(len(todos), ntr))
    den = np.asarray(K.sum(axis=1)).ravel()
    kb = np.divide(K @ r, den, out=np.zeros(len(todos)), where=den > 0)
    k_tr = kb[:ntr]
    s2 = float(k_tr @ k_tr)
    rho = float(np.clip((k_tr @ r) / s2, 0.0, 1.0)) if s2 > 0 else 0.0
    k_teste = kb[ntr:]
    return rho, (k_teste if sd_teste is None else k_teste * np.asarray(sd_teste, float))


def _adjacencia_binaria(w):
    A = sp.csr_matrix(w.sparse, dtype=float)
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return ((A + A.T) > 0).astype(float).tocsr()


def _precisao_leroux(A, lam):
    """Q sem o fator 1/tau: (1-lam) I + lam (D - A). Positiva definida para lam < 1,
    mesmo com ilhas (a parte (1-lam) I mantém o posto)."""
    d = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(1.0 - lam + lam * d) - lam * A).tocsc()


def _logdet_spd(M):
    return float(np.sum(np.log(np.abs(splu(M.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0).U.diagonal()))))


def car_residuos(r_treino, w, treino, teste, sd_teste=None):
    """Efeito BYM em dois estágios nos resíduos logit: (parâmetros, previsão do teste).

    r = u + e, u ~ CAR de Leroux com precisão (1/tau)[(1-lam) I + lam (D - A)] na
    adjacência queen binária, e ~ N(0, sigma2) iid. (tau, lam, sigma2) saem da máxima
    verossimilhança gaussiana no subgrafo de treino, então nada do teste entra no ajuste.
    A previsão é a média a posteriori de u no grafo completo dado só r de treino; a UF
    deixada de fora só é informada pelos vizinhos de treino pela divisa. Ilhas ficam com
    previsão 0 (mais a média removida), sem NaN. Com `sd_teste` (um por município de
    teste), r_treino é padronizado e a previsão volta multiplicada por sd_i; sem ele,
    nada muda.
    """
    tr, te = np.asarray(treino), np.asarray(teste)
    r = np.asarray(r_treino, float)
    mu = float(r.mean())
    rc = r - mu
    A = _adjacencia_binaria(w)
    Ao = A[tr][:, tr]
    n = len(tr)
    I = sp.identity(n, format="csc")
    falhas = [0]

    def nll(theta):
        tau, lam, s2 = np.exp(theta[0]), 0.99 / (1 + np.exp(-theta[1])), np.exp(2 * theta[2])
        Q = _precisao_leroux(Ao, lam) / tau
        try:
            ld = _logdet_spd(Q + I / s2) - _logdet_spd(Q) + n * np.log(s2)
            quad = rc @ rc / s2 - (rc @ splu(Q + I / s2).solve(rc)) / s2**2
        except RuntimeError:
            falhas[0] += 1
            return 1e12
        v = 0.5 * (ld + quad)
        return v if np.isfinite(v) else 1e12

    v0 = max(float(rc.var()), 1e-12)
    inicio = np.array([np.log(v0 / 2), 0.0, 0.5 * np.log(v0 / 2)])
    ot = minimize(nll, inicio, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
    if not ot.success:
        warnings.warn(f"car_residuos: otimizador não convergiu em {ot.nit} iterações ({ot.message}); "
                      "usando o melhor ponto encontrado", stacklevel=2)
    if falhas[0]:
        warnings.warn(f"car_residuos: {falhas[0]} avaliações da verossimilhança falharam na fatoração "
                      "e valeram 1e12", stacklevel=2)
    tau, lam, s2 = float(np.exp(ot.x[0])), float(0.99 / (1 + np.exp(-ot.x[1]))), float(np.exp(2 * ot.x[2]))

    N = A.shape[0]
    Do = np.zeros(N)
    Do[tr] = 1.0
    rfull = np.zeros(N)
    rfull[tr] = rc
    Q = _precisao_leroux(A, lam) / tau + sp.diags(Do / s2)
    m = splu(Q.tocsc()).solve(Do * rfull / s2)
    prev = m[te] + mu
    return {"tau": tau, "lam": lam, "sigma2": s2}, (prev if sd_teste is None else prev * np.asarray(sd_teste, float))


# --------------------------------------------------------------------------
# Notebook 05: Wald da dispersão, krigagem/CAR/ponderação por dobra e Moran fora da amostra
# --------------------------------------------------------------------------

LIMIAR_CORRELACAO = 0.05   # o corte de Conley é a 1ª faixa em que a correlação dos resíduos cai abaixo disto


def corte_de(cg: pd.DataFrame) -> float:
    """Corte de Conley (km): início da 1ª faixa com correlação < 0,05; sem faixa assim, o fim da última."""
    abaixo = cg[cg.correlacao < LIMIAR_CORRELACAO]
    return float(abaixo.iloc[0]["ini_km"]) if len(abaixo) else float(cg.iloc[-1]["fim_km"])


def passo_wald(passo: str, nome: str, ajuste: "Ajuste", residuos: np.ndarray, sel: Callable[[str], bool],
               lat: np.ndarray, lon: np.ndarray, uf: np.ndarray, faixas: list[float]) -> tuple[pd.DataFrame, float]:
    """Wald dos termos de `ajuste` escolhidos por `sel`, sob Conley (x0,5, x1, x2) e conglomerado por UF.

    O corte vem do correlograma dos resíduos do próprio ajuste. O qui-quadrado só sai com Conley x1;
    as demais covariâncias vão como F com (nº de UFs - 1) gl no denominador.
    """
    mod = ajuste.modelo
    nm = list(ajuste.nomes)
    ind = [i for i, s in enumerate(nm) if sel(s)]
    corte = corte_de(correlograma(residuos, lat, lon, faixas))
    covs = {"conley_05": cov_conley(mod, lat, lon, corte * 0.5), "conley_1": cov_conley(mod, lat, lon, corte),
            "conley_2": cov_conley(mod, lat, lon, corte * 2), "cluster_uf": cov_cluster(mod, uf, correcao=True)}
    fator = {"conley_05": 0.5, "conley_1": 1, "conley_2": 2}
    linhas = []
    for tag, cv in covs.items():
        for tipo, gd in ([("chi2", None)] if tag == "conley_1" else []) + [("F", len(np.unique(uf)) - 1)]:
            est, gl, p = wald_robusto(mod.params, cv, ind, gl_den=gd)
            linhas.append({"passo": passo, "ajuste": nome, "termos": ";".join(nm[i] for i in ind), "covariancia": tag,
                           "corte_km": corte * fator[tag] if tag != "cluster_uf" else np.nan,
                           "teste": tipo, "estat": est, "gl": gl, "gl_den": gd if gd else np.nan, "p": p})
    return pd.DataFrame(linhas), corte


def ep(cov: "npt.ArrayLike") -> np.ndarray:
    """Erro-padrão (raiz da diagonal) de uma matriz de covariância."""
    return np.sqrt(np.diag(np.asarray(cov, float)))


def substituir_linha_csv(arq: Path, linha: dict[str, Any], chave: str = "modelo") -> pd.DataFrame:
    """Põe `linha` no fim do CSV, trocando a que tem a mesma `chave` (idempotente) e mantendo o cabeçalho."""
    d = pd.read_csv(arq, float_precision="round_trip")
    d = pd.concat([d[d[chave] != linha[chave]], pd.DataFrame([linha])[list(d.columns)]], ignore_index=True)
    d.to_csv(arq, index=False)
    return d


# Contexto dos workers: ProcessPoolExecutor com fork herda este dicionário, então
# `definir_contexto_dobras` roda no processo principal antes de criar o pool.
_CTX: dict[str, Any] = {}
COLUNAS_EXTENSAO = {"krig": ["rho"], "car": ["tau", "lam", "sigma2"]}


def definir_contexto_dobras(**contexto: object) -> None:
    """Guarda o que as dobras leem: cand, df, mapa e, conforme a dobra, y, n, lz, lat, lon, uf, corte, w, nhat."""
    _CTX.clear()
    _CTX.update(contexto)


def _dobra_extensao(args: tuple[np.ndarray, np.ndarray]) -> tuple[list[float], np.ndarray]:
    """Uma dobra de krigagem ou CAR nos resíduos logit; devolve (parâmetros, π de teste corrigido)."""
    tr, te = args
    c = _CTX
    cand, df, mapa = c["cand"], c["df"], c["mapa"]
    a = cand.ajustar(df.iloc[tr], mapa)
    pi_tr = np.asarray(cand.prever(a, df.iloc[tr], mapa), float)
    r = c["lz"][tr] - logit(pi_tr)
    if c["metodo"] == "krig":
        rho, k = krigagem_residuos(r, c["lat"], c["lon"], tr, te, c["uf"], c["corte"])
        par, u = [rho], rho * k
    else:
        p, u = car_residuos(r, c["w"], tr, te)
        par = [p["tau"], p["lam"], p["sigma2"]]
    pi_te = np.asarray(cand.prever(a, df.iloc[te], mapa), float)
    return par, expit(logit(pi_te) + u)


def rodar_extensao(metodo: str, folds: list[tuple[np.ndarray, np.ndarray]], arq: Path, recalcular: bool,
                   max_workers: int) -> tuple[np.ndarray, np.ndarray]:
    """π corrigido e parâmetros por município (uma linha por dobra) da krigagem ('krig') ou do CAR ('car').

    Lê o cache `arq` salvo ao fim da 1ª rodada, a menos que `recalcular`. Exige `definir_contexto_dobras`.
    """
    cols = COLUNAS_EXTENSAO[metodo]
    if arq.exists() and not recalcular:
        d = pd.read_parquet(arq)
        return d["pi"].to_numpy(), d[cols].to_numpy()
    _CTX["metodo"] = metodo
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as ex:
        res = list(ex.map(_dobra_extensao, folds))
    N = len(_CTX["df"])
    pi, P = np.full(N, np.nan), np.full((N, len(cols)), np.nan)
    for (_, te), (par, p_) in zip(folds, res, strict=True):
        pi[te], P[te] = p_, par
    pd.DataFrame({"pi": pi, **{c: P[:, j] for j, c in enumerate(cols)}}).to_parquet(arq)
    return pi, P


def parametros_dobra(nome_uf: str) -> dict[str, Any]:
    """LOSO do modelo final: β, covariâncias do bloco β, X e θ da UF retida (peso n estimado `nhat`)."""
    from src import especificacoes as E

    c = _CTX
    cand, df, mapa, uf = c["cand"], c["df"], c["mapa"], c["uf"]
    tr, te = np.flatnonzero(uf != nome_uf), np.flatnonzero(uf == nome_uf)
    a = cand.ajustar(df.iloc[tr], mapa)
    dte = df.iloc[te].copy()
    dte[mapa["n"]] = c["nhat"][te]   # UF retida não apurada: peso n estimado
    X = E.montar_X(dte, mapa, a.pad, cand.termos)
    nb_ = a.extra["n_beta"]
    assert X.shape[1] == nb_
    th = E.prever_theta(a, dte, mapa)
    cm = np.asarray(a.modelo.cov_params(), float)[:nb_, :nb_]
    cc = np.asarray(cov_conley(a.modelo, c["lat"][tr], c["lon"][tr], c["corte"]), float)[:nb_, :nb_]
    return {"beta": np.asarray(a.modelo.params, float)[:nb_], "cov_modelo": cm, "cov_conley": cc,
            "X": X.to_numpy(float), "theta": th, "te": te}


def dobra_ponderada(args: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """Uma dobra de krigagem e CAR em resíduos de Pearson (padronizados pelo sd beta-binomial)."""
    from src import especificacoes as E

    tr, te = args
    c = _CTX
    cand, df, mapa, y, n = c["cand"], c["df"], c["mapa"], c["y"], c["n"]
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        a = cand.ajustar(df.iloc[tr], mapa)
        pi_tr = np.asarray(cand.prever(a, df.iloc[tr], mapa), float)
        z_tr, _ = residuo_pearson(y[tr], n[tr], pi_tr, E.prever_theta(a, df.iloc[tr], mapa))
        pi_te = np.asarray(cand.prever(a, df.iloc[te], mapa), float)
        th_te = E.prever_theta(a, df.iloc[te], mapa)
        sd_te = sd_betabinomial(pi_te, n[te], th_te)
        rho, k = krigagem_residuos(z_tr, c["lat"], c["lon"], tr, te, c["uf"], c["corte"], sd_teste=sd_te)
        par, u = car_residuos(z_tr, c["w"], tr, te, sd_teste=sd_te)
    return {"rho": rho, "lam": par["lam"], "pi": pi_te, "th": th_te, "krig": np.clip(pi_te + k, 1e-6, 1 - 1e-6),
            "car": np.clip(pi_te + u, 1e-6, 1 - 1e-6), "avisos": sorted({str(w.message)[:90] for w in avisos})}


def moran_oos(lz: np.ndarray, pi: np.ndarray, w: "W") -> float:
    """I de Moran (999 permutações, semente 0) do resíduo logit fora da amostra."""
    return moran_residuos(lz - logit(pi), w, permutacoes=999, semente=0)["I"]


def moran_z(y: np.ndarray, n: np.ndarray, pi: np.ndarray, theta: np.ndarray, w: "W") -> float:
    """I de Moran (999 permutações, semente 0) do resíduo de Pearson beta-binomial."""
    return moran_residuos(residuo_pearson(y, n, pi, theta)[0], w, permutacoes=999, semente=0)["I"]


def linha_sens(modelo: str, metrica: str, atual: float, pond: float, ca: bool | None = None,
               cp: bool | None = None) -> dict[str, Any]:
    """Linha da tabela de sensibilidade: valor atual contra ponderado, com o critério de cada um (se houver)."""
    return {"modelo": modelo, "metrica": metrica, "atual": atual, "ponderado": pond, "criterio_atual": ca, "criterio_ponderado": cp}
