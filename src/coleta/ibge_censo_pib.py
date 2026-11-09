"""Covariáveis municipais do Censo 2022 e do PIB dos Municípios (IBGE/SIDRA).

Regras de negócio:
- Chave: id_municipio (7 dígitos IBGE), nível territorial N6 da API de agregados.
- Religião (tabela 9537) tem como base a população de 10+ anos (Total, incluindo
  "Não sabe" e "Sem declaração"); os percentuais não são sobre a população total.
- Alfabetização (9543) é a taxa publicada pelo IBGE para 15+ anos, sem recorte.
- pct_urbana = urbana / total da tabela 9923 (situação do domicílio).
- PIB per capita = PIB (mil R$ * 1000) / população; o PIB municipal não traz variável
  per capita. Usa-se a estimativa do mesmo ano (tabela 6579) quando existe; a série
  de estimativas não tem 2022 e 2023 (recomeça em 2024 após o Censo), então nesses
  anos o denominador é a população do Censo 2022 (aproximação do ano mais próximo).
- pct_vab_agro = VAB da agropecuária / VAB total, ambos a preços correntes.
- No SIDRA "-" é zero absoluto: nas contagens do Censo vira 0 (ex.: município sem
  indígenas). "X" (sigilo) e "..." (não disponível) viram NaN; nada é imputado.
- O VAB (total e agropecuária) de 2022 e 2023 está como "..." na tabela 5938 (só o
  PIB foi divulgado), então pct_vab_agro existe apenas até 2021; a coluna de um ano
  sem VAB não é criada.
- O universo é o dos municípios do Censo 2022 (5.570); códigos que só aparecem em
  outras tabelas (município instalado depois) são descartados.
- 2021 é o último ano antes da eleição de 2022 (ajuste do backtest); o ano mais
  recente publicado alimenta 2026.
"""
import json

import pandas as pd

from src.coleta.comum import RAW, baixar, salvar_interim

API = "https://servicodados.ibge.gov.br/api/v3/agregados"
CACHE = RAW / "ibge_censo_pib"
ANO_PIB_BACKTEST = 2021


def _consulta(tabela, periodo, var, classif=None, traco_zero=False):
    """Retorna Series (id_municipio -> valor numérico) de uma variável/recorte."""
    sufixo = f"&classificacao={classif}" if classif else ""
    url = (f"{API}/{tabela}/periodos/{periodo}/variaveis/{var}"
           f"?localidades=N6[all]{sufixo}")
    nome = f"t{tabela}_p{periodo}_v{var}_{(classif or 'sem').replace('[', '_').replace(']', '').replace('|', '-').replace(',', '-')}.json"
    destino = CACHE / nome
    CACHE.mkdir(parents=True, exist_ok=True)
    baixar(url, destino)
    dados = json.loads(destino.read_text(encoding="utf-8"))
    serie = dados[0]["resultados"][0]["series"]
    bruto = pd.Series(
        {int(s["localidade"]["id"]): s["serie"][str(periodo)] for s in serie}
    )
    if traco_zero:
        bruto = bruto.replace("-", "0")
    return pd.to_numeric(bruto, errors="coerce")


def _periodos(tabela):
    destino = CACHE / f"periodos_{tabela}.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    baixar(f"{API}/{tabela}/periodos", destino)
    return [int(p["id"]) for p in json.loads(destino.read_text(encoding="utf-8"))]


def _censo():
    pop = _consulta(9514, 2022, 93, "2[6794]|287[100362]|286[113635]")
    dom = _consulta(4712, 2022, 381)
    raca = {c: _consulta(9605, 2022, 93, f"86[{c}]", traco_zero=True) for c in (2777, 2779, 2780, 95251)}
    urb = {c: _consulta(9923, 2022, 93, f"1[{c}]", traco_zero=True) for c in (1, 6795)}
    rel = {c: _consulta(9537, 2022, 140, f"133[{c}]|2[6794]|58[95253]", traco_zero=True)
           for c in (95278, 95277, 95263, 2836)}
    alfa = _consulta(9543, 2022, 2513, "2[6794]|86[95251]|287[100362]")
    return pd.DataFrame({
        "populacao_2022": pop,
        "domicilios_2022": dom,
        "pct_pretos_pardos_2022": 100 * (raca[2777] + raca[2779]) / raca[95251],
        "pct_indigenas_2022": 100 * raca[2780] / raca[95251],
        "taxa_alfabetizacao_15mais_2022": alfa,
        "pct_evangelicos_2022": 100 * rel[95277] / rel[95278],
        "pct_catolicos_2022": 100 * rel[95263] / rel[95278],
        "pct_sem_religiao_2022": 100 * rel[2836] / rel[95278],
        "pct_urbana_2022": 100 * urb[1] / urb[6795],
    })


def _pib(ano, pop_censo):
    pib = _consulta(5938, ano, 37) * 1000
    vab = _consulta(5938, ano, 498)
    agro = _consulta(5938, ano, 513)
    est = _consulta(6579, ano, 9324) if ano in _periodos(6579) else pop_censo
    saida = pd.DataFrame({f"pib_per_capita_{ano}": pib / est})
    if vab.notna().any():
        saida[f"pct_vab_agro_{ano}"] = 100 * agro / vab
    return saida


def main():
    ultimo = max(_periodos(5938))
    anos = [ANO_PIB_BACKTEST, ultimo]
    censo = _censo()
    pibs = [_pib(a, censo["populacao_2022"]) for a in anos]
    df = pd.concat([censo] + pibs, axis=1).loc[censo.index]
    df.index.name = "id_municipio"
    df = df.reset_index()
    df["id_municipio"] = df["id_municipio"].astype(int)
    print("linhas:", len(df), "| pop total:", df["populacao_2022"].sum())
    print("NaN por coluna:\n", df.isna().sum().to_string())
    salvar_interim(df, "ibge_censo_pib")


if __name__ == "__main__":
    main()
