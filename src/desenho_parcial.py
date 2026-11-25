"""Desenho do modelo refeito a cada (ano, corte) da apuração parcial, só com o já apurado.

Motivo: o `03-apuracao-parcial` usava termos, dispersão, SLX, corte de Conley e σ_UF escolhidos
no resultado final de 2026 (Parte 11 a 16 do `02-MLG`). Isso vaza a resposta. Aqui o desenho inteiro
sai dos municípios com voto apurado (D), com as mesmas regras do `02-MLG`, escritas em
2026-11-25 antes de ver resultados.

Regras:
- R1 termos da média: fusão de níveis por LRT e passo a passo por BIC (`selecao._fusoes` e
  `selecao._melhor_bic`), como na seleção aninhada do LOSO. `delta_comp` não é candidata em 2022
  (precisaria de 2014, fora da base).
- R2 corte de Conley: regra da Parte 15 (1ª faixa do correlograma com correlação < 0,05) nos
  resíduos quantílicos do desenho FINAL sobre D (depois de R4 e R5), sem a checagem cruzada com
  2022. O corte do ajuste R1 fica só como diagnóstico (`extra["corte_r1_km"]`).
- R3 LOSO em D: cada UF com pelo menos `MIN_MUN_UF` municípios apurados sai do treino; prevê-se o
  share apurado dela (peso = n apurado) contra o observado. Limite L = MAE por UF do R1 + EP do R1.
- R4 dispersão (Parte 13): Wald sequencial de Conley, cada passo no corte dos resíduos do próprio
  ajuste, como no `02-MLG` (passo A região contra const,
  passo B log n contra região, p < 0,01); aceita só se o MAE por UF no LOSO ficar em L.
  O BIC e o LRT só motivam, como no `02-MLG`; a decisão é do Wald.
- R5 SLX (Parte 14): W·x escolhido por BIC entre nenhuma, W·lag e W·(termos de R1); aceito só se o
  I de Moran dos resíduos cair contra nenhuma e o MAE por UF no LOSO ficar em L. SLX só é testado
  se a dispersão aceita pelo Wald também passou no LOSO (regra de aceitação da Parte 14).
- R6 ESF, krigagem e CAR não são retestados: foram rejeitados no `02-MLG` e só poderiam
  acrescentar termos.
- R7 σ_UF do corte = raiz da média dos quadrados de logit(share previsto) - logit(share observado)
  entre as UFs do R3, no LOSO do desenho final (reaproveita o LOSO já feito, sem reajustar).

O cache em `results/cv/desenho_parcial_<ano>_<corte>.json` guarda `VERSAO`; mudar uma regra
exige subir `VERSAO` para invalidar os arquivos antigos.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logit
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src import espacial as S
from src import especificacoes as E
from src import pressupostos as PR
from src import selecao as SEL
from src.especificacoes import Ajuste, Candidato, Termos

if TYPE_CHECKING:
    from libpysal.weights import W


VERSAO = 1
MIN_MUN_UF = 5
N_JOBS = 14   # mesmo número de processos de selecao.py
P_FUSAO = 0.05
P_WALD = 0.01
CORTE_MIN_KM = 1.0   # corte 0 zeraria o núcleo de Bartlett (divisão por zero); o desenho registra quando vale
FAIXAS_KM: tuple[float, ...] = (0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000)
ANO_SEM_DELTA_COMP = 2022
SEM_2022: frozenset[str] = frozenset({"delta_comp"})
PASTA_CACHE = Path(__file__).resolve().parents[1] / "results/cv"

Log = Callable[[str], None]


@dataclass(frozen=True, eq=False)
class DadosCorte:
    """Base completa de municípios com y e n substituídos pelo apurado; `ok` marca D (n apurado > 0)."""

    df: pd.DataFrame
    mapa: dict[str, str]
    w: W              # pesos Queen dos municípios (só geometria): usa `.sparse` e `.n`
    ok: np.ndarray

    @property
    def D(self) -> pd.DataFrame:
        return self.df.loc[self.ok].reset_index(drop=True)


def dados_do_corte(df: pd.DataFrame, mapa: dict[str, str], w: W, y_c: np.ndarray,
                   n_c: np.ndarray) -> DadosCorte:
    """Mesmas linhas e valores que `apuracao.ajustar_parcial` usa no treino (n apurado > 0)."""
    d = df.copy()
    d[mapa["y"]] = np.asarray(y_c, float)
    d[mapa["n"]] = np.asarray(n_c, float)
    return DadosCorte(d, dict(mapa), w, np.asarray(n_c, float) > 0)


def candidatas_do_ano(ano: int) -> tuple[str, ...]:
    """Candidatas da seleção; em 2022 `delta_comp` (e seu W·x) fica de fora."""
    base = SEL.CANDIDATAS_PADRAO
    return tuple(c for c in base if ano != ANO_SEM_DELTA_COMP or c not in SEM_2022)


# --------------------------------------------------------------------------
# Resultado
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DesenhoParcial:
    """Desenho escolhido num (ano, corte) e os números que justificam cada decisão."""

    termos: Termos
    corte_km: float
    sigma_uf: float
    limite: float                      # L = MAE(R1) + EP(R1), p.p.
    mae_pp: dict[str, float]           # por etapa testada: r1, disp, slx, final
    ep_r1_pp: float
    aceitos: dict[str, bool]           # disp, slx
    wald: dict[str, float]             # p dos passos A e B (nan se não testado)
    bic_dispersao: dict[str, float]
    n_ufs_loso: int
    n_municipios: int
    tempo_s: float
    versao: int = VERSAO
    extra: dict[str, Any] = field(default_factory=dict)

    def para_dict(self) -> dict[str, Any]:
        t = self.termos
        return {"versao": self.versao,
                "termos": {"vars": list(t.vars), "disp": t.disp, "slx": list(t.slx),
                           "regiao_fundir": [list(g) for g in t.regiao_fundir],
                           "bloco_grupos": [list(g) for g in t.bloco_grupos]},
                "corte_km": self.corte_km, "sigma_uf": self.sigma_uf, "limite": self.limite,
                "mae_pp": self.mae_pp, "ep_r1_pp": self.ep_r1_pp, "aceitos": self.aceitos,
                "wald": self.wald, "bic_dispersao": self.bic_dispersao,
                "n_ufs_loso": self.n_ufs_loso, "n_municipios": self.n_municipios,
                "tempo_s": self.tempo_s, "extra": self.extra}

    @classmethod
    def de_dict(cls, d: Mapping[str, Any]) -> DesenhoParcial:
        t = d["termos"]
        termos = Termos(vars=tuple(t["vars"]), disp=t["disp"], slx=tuple(t["slx"]),
                        regiao_fundir=tuple(tuple(g) for g in t["regiao_fundir"]),
                        bloco_grupos=tuple(tuple(g) for g in t["bloco_grupos"]))
        return cls(termos, float(d["corte_km"]), float(d["sigma_uf"]), float(d["limite"]),
                   dict(d["mae_pp"]), float(d["ep_r1_pp"]), dict(d["aceitos"]), dict(d["wald"]),
                   dict(d["bic_dispersao"]), int(d["n_ufs_loso"]), int(d["n_municipios"]),
                   float(d["tempo_s"]), int(d["versao"]), dict(d.get("extra", {})))


def arquivo_cache(ano: int, corte: int, pasta: Path = PASTA_CACHE) -> Path:
    return pasta / f"desenho_parcial_{ano}_{corte}.json"


def gravar_cache(des: DesenhoParcial, ano: int, corte: int, pasta: Path = PASTA_CACHE) -> Path:
    arq = arquivo_cache(ano, corte, pasta)
    arq.parent.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_suffix(".tmp")
    tmp.write_text(json.dumps(des.para_dict(), indent=1))
    os.replace(tmp, arq)
    return arq


def ler_cache(ano: int, corte: int, pasta: Path = PASTA_CACHE) -> DesenhoParcial | None:
    """None se o arquivo não existe ou foi gravado com outra `VERSAO` de regras."""
    arq = arquivo_cache(ano, corte, pasta)
    if not arq.exists():
        return None
    d = json.loads(arq.read_text())
    return DesenhoParcial.de_dict(d) if d.get("versao") == VERSAO else None


# --------------------------------------------------------------------------
# LOSO em D
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Loso:
    """Share apurado previsto e observado por UF (frações), uma UF por dobra."""

    ufs: tuple[str, ...]
    prev: np.ndarray
    obs: np.ndarray
    puladas: tuple[str, ...] = ()   # UFs cuja dobra deu ajuste singular: ficam fora do MAE e do σ_UF

    @property
    def erros_pp(self) -> np.ndarray:
        return np.abs(self.prev - self.obs) * 100

    @property
    def mae_pp(self) -> float:
        return float(self.erros_pp.mean())

    @property
    def ep_pp(self) -> float:
        e = self.erros_pp
        return float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else float("nan")


def sigma_uf(loso: Loso) -> float:
    """R7: RMS entre UFs de logit(share previsto) - logit(share observado)."""
    p = np.clip(loso.prev, 1e-9, 1 - 1e-9)
    o = np.clip(loso.obs, 1e-9, 1 - 1e-9)
    return float(np.sqrt(np.mean((logit(p) - logit(o)) ** 2)))


def ufs_elegiveis(uf: np.ndarray, minimo: int = MIN_MUN_UF) -> tuple[str, ...]:
    """UFs com pelo menos `minimo` municípios apurados; as menores ficam sempre no treino."""
    cont = pd.Series(np.asarray(uf).astype(str)).value_counts()
    return tuple(sorted(cont[cont >= minimo].index))


_CTX_LOSO: dict[str, Any] = {}   # herdado pelos filhos do fork


def _dobra(uf_ret: str) -> tuple[float, float]:
    c = _CTX_LOSO
    uf, D, mapa, cand = c["uf"], c["D"], c["mapa"], c["cand"]
    tr, te = np.flatnonzero(uf != uf_ret), np.flatnonzero(uf == uf_ret)
    aj = cand.ajustar(D.iloc[tr], mapa)
    dte = D.iloc[te]
    pi = np.asarray(cand.prever(aj, dte, mapa), float)
    n = dte[mapa["n"]].to_numpy(float)
    y = dte[mapa["y"]].to_numpy(float)
    return float((pi * n).sum() / n.sum()), float(y.sum() / n.sum())


def _dobra_segura(uf_ret: str) -> tuple[float, float] | None:
    """Dobra que falha por matriz singular (corte inicial, região sem município apurado) é pulada."""
    try:
        return _dobra(uf_ret)
    except (np.linalg.LinAlgError, PerfectSeparationError):
        return None


def _loso(cand: Candidato, D: pd.DataFrame, mapa: dict[str, str], n_jobs: int) -> Loso:
    """Ajusta `cand` em D sem cada UF elegível e prevê o share apurado dela (peso n apurado)."""
    uf = D[mapa["uf"]].astype(str).to_numpy()
    ufs = ufs_elegiveis(uf)
    _CTX_LOSO.clear()
    _CTX_LOSO.update(uf=uf, D=D, mapa=mapa, cand=cand)
    res = SEL._mapear(_dobra_segura, ufs, n_jobs)
    bons = [(u, r) for u, r in zip(ufs, res) if r is not None]
    return Loso(tuple(u for u, _ in bons), np.array([r[0] for _, r in bons]),
                np.array([r[1] for _, r in bons]), tuple(u for u, r in zip(ufs, res) if r is None))


# --------------------------------------------------------------------------
# Regras R1 a R5
# --------------------------------------------------------------------------
def _candidato(termos: Termos, nome: str = "parcial") -> Candidato:
    return Candidato(nome, "betabin", "logit", termos)


def _selecionar_media(D: pd.DataFrame, mapa: dict[str, str], candidatas: Sequence[str],
                      n_jobs: int, rotulo: str, log: Log) -> Termos:
    """R1: fusões por LRT e conjunto de contínuas por BIC (lag sempre dentro)."""
    t0 = Termos(vars=("lag", *candidatas))
    grupos, grupos_bloco = SEL._fusoes(D, mapa, t0, P_FUSAO)
    tf = replace(t0, regiao_fundir=grupos, bloco_grupos=grupos_bloco)
    _, vs, _ = SEL._melhor_bic(D, mapa, tf, candidatas, n_jobs, rotulo, log)
    return replace(tf, vars=("lag", *vs))


def _residuos(cand: Candidato, aj: Ajuste, D: pd.DataFrame, mapa: dict[str, str]) -> np.ndarray:
    pi = np.asarray(cand.prever(aj, D, mapa), float)
    return PR.residuos_quantilicos_rapido(
        D[mapa["y"]].to_numpy(float), D[mapa["n"]].to_numpy(float), pi,
        theta=E.prever_theta(aj, D, mapa), rng=0)


def _corte_proprio(res: np.ndarray, D: pd.DataFrame) -> tuple[float, bool]:
    """Regra da Parte 15 nos resíduos dados; devolve (corte em km, se o piso foi aplicado)."""
    cg = S.correlograma(res, D["lat"].to_numpy(float), D["lon"].to_numpy(float), list(FAIXAS_KM))
    corte = S.corte_de(cg)
    return (CORTE_MIN_KM, True) if corte < CORTE_MIN_KM else (corte, False)


def _corte_do_desenho(cand: Candidato, D: pd.DataFrame, mapa: dict[str, str]) -> tuple[float, bool]:
    return _corte_proprio(_residuos(cand, cand.ajustar(D, mapa), D, mapa), D)


def _p_wald(aj: Ajuste, sel: Callable[[str], bool], D: pd.DataFrame, corte: float) -> float:
    ind = [i for i, s in enumerate(aj.nomes) if sel(s)]
    cov = S.cov_conley(aj.modelo, D["lat"].to_numpy(float), D["lon"].to_numpy(float), corte)
    return float(S.wald_robusto(aj.modelo.params, cov, ind)[2])


def _bic(aj: Ajuste, nobs: int) -> float:
    return float(aj.k * np.log(nobs) - 2 * aj.loglik)


def _wald_dispersao(base: Termos, D: pd.DataFrame, mapa: dict[str, str]
                    ) -> tuple[str, dict[str, float], dict[str, float], dict[str, Any]]:
    """R4: dispersão pelo Wald sequencial de Conley, cada passo no corte do próprio ajuste.

    Devolve (forma, p de A e B, BIC por forma, cortes usados e se o piso valeu).
    """
    nobs = len(D)
    aj0 = _candidato(base).ajustar(D, mapa)
    bic = {"const": _bic(aj0, nobs)}
    p = {"A": float("nan"), "B": float("nan")}
    cand_a = _candidato(replace(base, disp="regiao"))
    aj_a = cand_a.ajustar(D, mapa)
    bic["regiao"] = _bic(aj_a, nobs)
    corte_a, piso_a = _corte_proprio(_residuos(cand_a, aj_a, D, mapa), D)
    info: dict[str, Any] = {"corte_A_km": corte_a, "piso": piso_a}
    p["A"] = _p_wald(aj_a, lambda s: s.startswith("disp_reg_"), D, corte_a)
    if not p["A"] < P_WALD:
        return "const", p, bic, info
    cand_b = _candidato(replace(base, disp="regiao_logn"))
    aj_b = cand_b.ajustar(D, mapa)
    bic["regiao_logn"] = _bic(aj_b, nobs)
    corte_b, piso_b = _corte_proprio(_residuos(cand_b, aj_b, D, mapa), D)
    info.update(corte_B_km=corte_b, piso=piso_a or piso_b)
    p["B"] = _p_wald(aj_b, lambda s: s == "disp_log_n", D, corte_b)
    return ("regiao_logn" if p["B"] < P_WALD else "regiao"), p, bic, info


def _moran_em_d(res: np.ndarray, w: W, ok: np.ndarray) -> float:
    """I de Moran dos resíduos de D com os vínculos Queen entre municípios de D (linhas padronizadas)."""
    idx = np.flatnonzero(ok)
    A = sp.csr_matrix(w.sparse)[idx][:, idx]
    mant = np.arange(len(idx))
    while True:
        A = A[mant][:, mant]
        soma = np.asarray(A.sum(axis=1)).ravel()
        if (soma > 0).all():
            break
        mant = mant[soma > 0]
    return float(S._moran_i(np.asarray(res)[mant], sp.diags(1.0 / soma) @ A))


def _escolher_slx(base: Termos, dados: DadosCorte
                  ) -> tuple[tuple[str, ...], bool, dict[str, float], DadosCorte]:
    """R5: opção de W·x por BIC e diagnóstico (I de Moran cai contra nenhuma).

    Devolve (chaves escolhidas, diagnóstico ok, BIC por opção, dados com as colunas w_*).
    """
    opcoes = {"nenhuma": (), "W_lag": ("lag",), "W_vars": tuple(base.vars)}
    df2, mapa2 = S.adicionar_slx(dados.df, dados.mapa, list(base.vars), dados.w)
    d2 = DadosCorte(df2, mapa2, dados.w, dados.ok)
    D2, nobs = d2.D, int(dados.ok.sum())
    aj, res, bic = {}, {}, {}
    for nome, ch in opcoes.items():
        c = _candidato(replace(base, slx=ch))
        aj[nome] = c.ajustar(D2, mapa2)
        bic[nome] = _bic(aj[nome], nobs)
        res[nome] = _residuos(c, aj[nome], D2, mapa2)
    esc = min(bic, key=lambda k: bic[k])
    if esc == "nenhuma":
        return (), False, bic, d2
    ok = _moran_em_d(res[esc], dados.w, dados.ok) < _moran_em_d(res["nenhuma"], dados.w, dados.ok)
    return opcoes[esc], bool(ok), bic, d2


# --------------------------------------------------------------------------
# Entrada pública
# --------------------------------------------------------------------------
def desenho_no_corte(dados: DadosCorte, ano: int, *, n_jobs: int = N_JOBS,
                     candidatas: Sequence[str] | None = None, rotulo: str = "",
                     log: Log = print) -> DesenhoParcial:
    """Desenho do (ano, corte) a partir só dos municípios apurados (regras R1 a R7 do módulo)."""
    t0 = time.time()
    cands = tuple(candidatas) if candidatas is not None else candidatas_do_ano(ano)
    D, mapa = dados.D, dados.mapa
    rotulo = rotulo or str(ano)

    base = _selecionar_media(D, mapa, cands, n_jobs, rotulo, log)            # R1
    c1 = _candidato(base)
    corte_r1, piso_r1 = _corte_do_desenho(c1, D, mapa)                        # diagnóstico de R2
    l1 = _loso(c1, D, mapa, n_jobs)                                           # R3
    limite = l1.mae_pp + l1.ep_pp
    mae = {"r1": l1.mae_pp}
    puladas = {"r1": list(l1.puladas)}
    final_termos, final_loso = base, l1
    D_fin, mapa_fin = D, mapa

    forma, p, bic, info_wald = _wald_dispersao(base, D, mapa)                # R4
    disp_ok = False
    if forma != "const":
        cd = _candidato(replace(base, disp=forma))
        ld = _loso(cd, D, mapa, n_jobs)
        mae["disp"] = ld.mae_pp
        puladas["disp"] = list(ld.puladas)
        disp_ok = ld.mae_pp <= limite
        if disp_ok:
            final_termos, final_loso = cd.termos, ld

    slx_ok = False
    if forma == "const" or disp_ok:                                           # R5
        base_slx = replace(base, disp=final_termos.disp)
        chaves, diag_ok, _, d2 = _escolher_slx(base_slx, dados)
        if diag_ok:
            cs = _candidato(replace(base_slx, slx=chaves))
            ls = _loso(cs, d2.D, d2.mapa, n_jobs)
            mae["slx"] = ls.mae_pp
            puladas["slx"] = list(ls.puladas)
            slx_ok = ls.mae_pp <= limite
            if slx_ok:
                final_termos, final_loso, D_fin, mapa_fin = cs.termos, ls, d2.D, d2.mapa
    mae["final"] = final_loso.mae_pp
    corte, piso_fin = _corte_do_desenho(_candidato(final_termos), D_fin, mapa_fin)   # R2
    return DesenhoParcial(
        termos=final_termos, corte_km=corte, sigma_uf=sigma_uf(final_loso),   # R7
        limite=limite, mae_pp=mae, ep_r1_pp=l1.ep_pp,
        aceitos={"disp": disp_ok, "slx": slx_ok}, wald=p, bic_dispersao=bic,
        n_ufs_loso=len(l1.ufs), n_municipios=len(D), tempo_s=time.time() - t0,
        extra={"corte_r1_km": corte_r1, "wald_cortes_km": {k: v for k, v in info_wald.items() if k != "piso"},
               "ufs_puladas": puladas, "piso_aplicado": {"r1": piso_r1, "final": piso_fin, "wald": bool(info_wald["piso"])}})


def desenhos_por_corte(dados_por_corte: Mapping[tuple[int, int], DadosCorte], *,
                       pasta: Path = PASTA_CACHE, refazer: bool = False, n_jobs: int = N_JOBS,
                       log: Log = print, falhas: dict[tuple[int, int], str] | None = None
                       ) -> dict[tuple[int, int], DesenhoParcial]:
    """Desenho de cada (ano, corte); lê o cache quando a `VERSAO` bate e grava o que calcula.

    Um corte com matriz singular fora das dobras do LOSO (ex.: corte inicial sem município de uma
    região) fica de fora do resultado e entra em `falhas`; o notebook o conta como derrota do modelo.
    """
    out: dict[tuple[int, int], DesenhoParcial] = {}
    for (ano, corte), dados in dados_por_corte.items():
        des = None if refazer else ler_cache(ano, corte, pasta)
        if des is None:
            try:
                des = desenho_no_corte(dados, ano, n_jobs=n_jobs, rotulo=f"{ano}/{corte}", log=log)
            except np.linalg.LinAlgError as e:
                if falhas is not None:
                    falhas[(ano, corte)] = repr(e)[:200]
                continue
            gravar_cache(des, ano, corte, pasta)
        out[(ano, corte)] = des
    return out
