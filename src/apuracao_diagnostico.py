"""Diagnóstico da apuração parcial: de onde vem a largura do IC e onde o restante é extrapolação.

Mede o modelo como está, sem alterá-lo. Três perguntas:
D1, qual componente da simulação faz a variância do share (um componente ligado por vez);
D2, como é composta a parte já apurada;
D3, quanto do restante está fora do suporte do ajuste (alavancagem, Paula 1.7.2 e 2.8.1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from src import apuracao as A
from src import especificacoes as E

# Capitais dos 27 entes (código IBGE de 7 dígitos). Não há marca de capital nas bases do projeto.
CAPITAIS_IBGE: tuple[int, ...] = (
    1100205,  # Porto Velho
    1200401,  # Rio Branco
    1302603,  # Manaus
    1400100,  # Boa Vista
    1501402,  # Belém
    1600303,  # Macapá
    1721000,  # Palmas
    2111300,  # São Luís
    2211001,  # Teresina
    2304400,  # Fortaleza
    2408102,  # Natal
    2507507,  # João Pessoa
    2611606,  # Recife
    2704302,  # Maceió
    2800308,  # Aracaju
    2927408,  # Salvador
    3106200,  # Belo Horizonte
    3205309,  # Vitória
    3304557,  # Rio de Janeiro
    3550308,  # São Paulo
    4106902,  # Curitiba
    4205407,  # Florianópolis
    4314902,  # Porto Alegre
    5002704,  # Campo Grande
    5103403,  # Cuiabá
    5208707,  # Goiânia
    5300108,  # Brasília
)

Z95 = 1.96
TOLERANCIA_SOMA = 0.10  # a soma das partes deve ficar a 10 % da variância com tudo ligado
COMPONENTES: tuple[str, ...] = ("beta", "uf", "theta", "nu")


def _sem_beta(pm: A.ParcialModelo) -> A.ParcialModelo:
    """β fixo em β̂: cov nula mantém o consumo do gerador, só zera a variância."""
    return replace(pm, cov=np.zeros_like(pm.cov))


def _sem_theta(pm: A.ParcialModelo) -> A.ParcialModelo:
    """Sem sobredispersão a contagem é binomial pura."""
    return replace(pm, theta=None)


def variancias_componentes(
    pm: A.ParcialModelo,
    base_m: pd.DataFrame,
    base: pd.DataFrame,
    pi_uf: pd.Series,
    ligacao: str,
    sigma_uf: float,
    B: int,
    semente: int,
    fatores: np.ndarray,
    gl_uf: int | None = None,
) -> pd.DataFrame:
    """Variância do share simulado por unidade (regiões e Brasil), um componente ligado por vez.

    `var_piso` é o piso binomial (tudo desligado); `var_tudo` liga os quatro componentes (é o
    IC com n_u simulado) e `var_ic` liga só β, UF e θ (é o IC do notebook, n_u fixo). Todas as
    simulações usam a mesma semente: assim cada coluna é comparável e `var_tudo`/`var_ic`
    reproduzem os quantis do notebook. `gl_uf` troca o choque de UF normal pelo de Student.
    """
    pm0 = _sem_theta(_sem_beta(pm))
    cenarios = {
        "var_piso": (pm0, 0.0, None),
        "var_beta": (_sem_theta(pm), 0.0, None),
        "var_uf": (pm0, sigma_uf, None),
        "var_theta": (_sem_beta(pm), 0.0, None),
        "var_nu": (pm0, 0.0, fatores),
        "var_tudo": (pm, sigma_uf, fatores),
        "var_ic": (pm, sigma_uf, None),
    }
    com_uf = {"var_uf", "var_tudo", "var_ic"}  # só aí há choque de UF
    cols: dict[str, np.ndarray] = {}
    regs: list[str] = []
    for nome, (p, sg, fk) in cenarios.items():
        share, regs = A.simular_shares(
            p, base_m, base, pi_uf, ligacao, sg, B, semente, fk, gl_uf if nome in com_uf else None
        )
        cols[nome] = share.var(axis=0, ddof=1)
    return pd.DataFrame(cols, index=regs + ["Brasil"])


def decompor(var: pd.DataFrame) -> pd.DataFrame:
    """Parte de cada componente na variância com tudo ligado.

    Cada componente entra como acréscimo sobre o piso binomial (as partes somam ao piso mais os
    quatro acréscimos). `soma_dentro_10pct` diz se essa soma fica a 10 % de `var_tudo`; fora
    disso há interação entre componentes e as partes não são aditivas.
    """
    out = var.copy()
    out["inc_beta"] = var["var_beta"] - var["var_piso"]
    out["inc_uf"] = var["var_uf"] - var["var_piso"]
    out["inc_theta"] = var["var_theta"] - var["var_piso"]
    out["inc_nu"] = var["var_nu"] - var["var_piso"]
    out["parte_piso"] = var["var_piso"] / var["var_tudo"]
    for c in COMPONENTES:
        out[f"parte_{c}"] = out[f"inc_{c}"] / var["var_tudo"]
    soma = out["var_piso"] + out[[f"inc_{c}" for c in COMPONENTES]].sum(axis=1)
    out["soma_partes"] = soma / var["var_tudo"]
    out["soma_dentro_10pct"] = (out["soma_partes"] - 1.0).abs() <= TOLERANCIA_SOMA
    inc = out[[f"inc_{c}" for c in COMPONENTES]].copy()
    inc["piso"] = var["var_piso"]
    out["maior_componente"] = inc.idxmax(axis=1).str.removeprefix("inc_")
    return out


def meia_largura_formula(
    base: pd.DataFrame, pi: pd.Series, sigma_uf: float, regiao: str | None, gl_uf: int | None = None
) -> tuple[float, float]:
    """Meia-largura (p.p.) do choque de UF por linearização: 1,96 σ π(1-π) √(Σ w_u²).

    Com `gl_uf` = k (choque de Student) o desvio é σ√(k/(k−2)); exige k > 2.

    w_u é a parte da UF nos votos restantes da unidade e π o share médio do restante (ponderado
    por n_u). Devolve (sem fator, com fator do restante). O fator f = n_u/(n_c+n_u) leva o
    desvio dos votos restantes para o share do total final, que é o que o IC mede.
    """
    b = base if regiao is None else base[base["regiao"] == regiao]
    p = A.completar_por_uf(pi.reindex(base.index), base["n_u"], base["uf"]).reindex(b.index)
    n_u = b["n_u"]
    if n_u.sum() <= 0:
        return 0.0, 0.0
    pi_u = float((p * n_u).sum() / n_u.sum())
    w = n_u.groupby(b["uf"]).sum() / n_u.sum()
    sigma_ef = sigma_uf if gl_uf is None else sigma_uf * float(np.sqrt(gl_uf / (gl_uf - 2)))
    meia = Z95 * sigma_ef * pi_u * (1 - pi_u) * float(np.sqrt((w**2).sum())) * 100
    f = float(n_u.sum() / (b["n_c"] + n_u).sum())
    return meia, meia * f


def composicao_do_corte(base: pd.DataFrame, capitais_tse: set[int], fundir: str) -> pd.DataFrame:
    """Composição do apurado por região e Brasil.

    `apurado_do_final` = votos válidos já contados / votos válidos finais; `mun_com_apuracao` =
    parte dos municípios com ao menos uma seção contada; `capitais_do_apurado` = parte dos votos
    contados que vêm de capitais. `regiao_fundir` é o agrupamento de regiões do desenho do corte.
    """
    linhas = []
    por_reg = {r: g for r, g in base.groupby("regiao")}
    por_reg["Brasil"] = base
    for r, g in por_reg.items():
        cap = g.index.isin(list(capitais_tse))
        linhas.append(
            {
                "regiao": r,
                "apurado_do_final": float(g["n_c"].sum() / g["n_final"].sum()),
                "mun_com_apuracao": float((g["n_c"] > 0).mean()),
                "capitais_do_apurado": float(g.loc[cap, "n_c"].sum() / g["n_c"].sum())
                if g["n_c"].sum() > 0
                else np.nan,
                "regiao_fundir": fundir,
            }
        )
    return pd.DataFrame(linhas)


def alavancagem(X: np.ndarray, cov_beta: np.ndarray) -> np.ndarray:
    """v_i = x_i' (X'ŴX)⁻¹ x_i, com (X'ŴX)⁻¹ a informação de Fisher inversa do ajuste (Paula 2.8.1)."""
    X = np.asarray(X, float)
    return np.einsum("ij,jk,ik->i", X, np.asarray(cov_beta, float), X)


def extrapolacao_por_regiao(
    v_ajuste: np.ndarray, v_resto: np.ndarray, n_u: np.ndarray, regiao: np.ndarray
) -> pd.DataFrame:
    """Parte do restante (ponderada por n_u) cuja alavancagem passa a máxima do ajuste.

    Uma linha do restante está fora do suporte quando v0 > max v_i das linhas ajustadas
    (extrapolação no espaço das covariáveis, Paula 1.7.2).
    """
    v_max = float(np.max(v_ajuste))
    n_u = np.asarray(n_u, float)
    regiao = np.asarray(regiao).astype(str)
    fora = np.asarray(v_resto) > v_max
    linhas = []
    for r in sorted(set(regiao)):
        m = (regiao == r) & (n_u > 0)
        tot = n_u[m].sum()
        linhas.append(
            {
                "regiao": r,
                "v_max_ajuste": v_max,
                "v0_max_resto": float(np.max(v_resto[m])) if m.any() else np.nan,
                "extrapolado_nu": float(n_u[m & fora].sum() / tot) if tot > 0 else np.nan,
            }
        )
    return pd.DataFrame(linhas)


def informacao_do_ajuste(
    cand: E.Candidato,
    dfm: pd.DataFrame,
    mapa: dict[str, str],
    y_c: np.ndarray,
    n_c: np.ndarray,
    n_u: np.ndarray,
    treino: Mapping[str, np.ndarray] | None = None,
    resto: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cov β do ajuste, X das linhas ajustadas, X de todas as linhas), no mesmo corte de `ajustar_parcial`.

    Treina nas linhas com n_c > 0, como `ajustar_parcial`; o X de previsão usa a padronização do
    treino. A covariância é a do modelo (Fisher inversa), não a de Conley. `treino` e `resto` são as
    sobrescritas por parte de `A.sobrescritas_1t`, as mesmas de `ajustar_parcial`.
    """
    d = dfm.copy()
    d[mapa["y"]] = y_c
    d[mapa["n"]] = n_c
    ok = n_c > 0
    todas = np.ones(len(d), bool)
    aj = cand.ajustar(A.aplicar_sobrescritas(d, treino, todas)[ok], mapa)
    dte = A.aplicar_sobrescritas(d, resto, todas).copy()
    dte[mapa["n"]] = np.maximum(n_u, 1.0)
    nb = int(aj.extra["n_beta"])
    X = E.montar_X(dte, mapa, aj.pad, cand.termos).to_numpy(float)
    cov = np.asarray(aj.modelo.cov_params(), float)[:nb, :nb]
    return cov, X[ok], X


# --------------------------------------------------------------------------
# D4: seleção dentro do município (Paula 4.6.6) e swing do 1º turno
# --------------------------------------------------------------------------
def _ponderar_lacunas(gap: pd.Series, peso: pd.Series, regiao: pd.Series) -> pd.DataFrame:
    """Média n_u-ponderada da lacuna e do seu módulo, por região e Brasil."""
    ok = gap.notna() & (peso > 0)
    g, w, r = gap[ok], peso[ok], regiao[ok]
    linhas = {}
    for nome, m in [*((k, r == k) for k in sorted(r.unique())), ("Brasil", pd.Series(True, index=r.index))]:
        wm = w[m]
        linhas[nome] = {
            "gap": float((g[m] * wm).sum() / wm.sum()),
            "gap_abs": float((g[m].abs() * wm).sum() / wm.sum()),
            "n_u": float(wm.sum()),
            "n_unidades": int(m.sum()),
        }
    return pd.DataFrame.from_dict(linhas, orient="index")


def selecao_municipal(base: pd.DataFrame) -> pd.DataFrame:
    """Lacuna logit(share_c) − logit(share_u) do 1º turno, n_u-ponderada, nos municípios com as duas partes.

    A ordem de apuração escolhe seções, não municípios inteiros: se a parte apurada já difere da restante no
    1º turno (mesmos eleitores, outro turno), a seleção dentro do município vale como seleção em x e o
    modelo municipal a ignora. Usa o logit empírico `lag1t_*` de `A.partes_1t` (regra de zeros dela) e só
    municípios cujas duas partes têm dado próprio (`fonte` = "secoes").
    """
    ambos = (base["fonte_c"] == "secoes") & (base["fonte_u"] == "secoes")
    gap = (base["lag1t_c"] - base["lag1t_u"]).where(ambos)
    return _ponderar_lacunas(gap, base["n_u"], base["regiao"])


def _logit_empirico(pt: pd.Series, bo: pd.Series) -> pd.Series:
    return np.log((pt + 0.5) / (bo + 0.5))


def selecao_uf(
    secoes: pd.DataFrame, secoes_1t: pd.DataFrame, recebida: pd.Series, ano: int, base: pd.DataFrame
) -> pd.DataFrame:
    """Mesma lacuna no nível da UF: logit empírico das somas do 1º turno nas seções apuradas e nas restantes.

    Peso: n_u da UF. Contraste com `selecao_municipal`: lacunas de sinais opostos entre municípios se
    cancelam na UF, que é o nível do choque σ_UF do modelo.
    """
    s = secoes[A.CHAVE_SECAO].assign(_rec=recebida.to_numpy())
    m = s.merge(
        secoes_1t.loc[secoes_1t["ano"] == ano, [*A.CHAVE_SECAO, "votos_lula", "votos_bolsonaro"]],
        on=A.CHAVE_SECAO,
        how="inner",
    )
    soma = m.groupby(["uf", "_rec"])[["votos_lula", "votos_bolsonaro"]].sum().unstack("_rec")
    ok = soma.notna().all(axis=1)
    lag = {r: _logit_empirico(soma["votos_lula"][r], soma["votos_bolsonaro"][r]) for r in (True, False)}
    gap = (lag[True] - lag[False]).where(ok)
    peso = base.groupby("uf")["n_u"].sum().reindex(gap.index)
    regiao = base.groupby("uf")["regiao"].first().reindex(gap.index)
    return _ponderar_lacunas(gap, peso, regiao)


def estimativa_swing_1t(base: pd.DataFrame) -> pd.Series:
    """Comparador descritivo: swing 1º → 2º turno das partes apuradas, aplicado ao 1º turno das restantes.

    Mesma escala de `A.swing_uniforme` (Σ y_c − Σ share_c·n_c)/Σ n_c, com `share_c` (1º turno da parte
    apurada) no lugar do 2º turno anterior; π da parte restante = share_u + swing. Nível da parte
    (apurada/restante por município), não da seção. Entra em `A.METODOS` e no veredito de utilidade.
    """
    sw = A.swing_uniforme(base["y_c"].to_numpy(), base["n_c"].to_numpy(), base["share_c"].to_numpy())
    pi = (base["share_u"] + sw).clip(0.0, 1.0)
    pi = A.completar_por_uf(pi, base["n_u"], base["uf"])
    return A.agregar_share(base["y_c"] + pi * base["n_u"], base["n_c"] + base["n_u"], base["regiao"])
