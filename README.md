# Previsão das Eleições 2026 por MLG — 2º turno presidencial (Lula × Flávio Bolsonaro)

## 📌 Sobre o projeto

Este repositório contém o desenvolvimento de um trabalho acadêmico sobre **Modelos Lineares Generalizados (MLG) da família binomial**, aplicados à votação por município no **segundo turno presidencial de 2026** (Lula × Flávio Bolsonaro), com agregação das previsões às **5 regiões** do Brasil.

O objetivo é comparar especificações de MLG com um critério de escolha fixado **antes** dos resultados e estimar a participação regional de cada candidato com o modelo escolhido.

> **Fontes principais:** TSE, IBGE, Portal da Transparência (CGU) e Bolognesi, Ribeiro e Codato (2023). Detalhes na seção [Sobre o Dataset](#-sobre-o-dataset).

A base final tem **5.570 municípios** (um censo, não uma amostra) e 160 colunas. Notebooks e documentação em português.

> **Ressalva (falácia ecológica).** O modelo descreve **municípios**, não eleitores: os coeficientes são associações entre municípios e não permitem inferir o comportamento individual (Robinson, 1950).

---

## 🎯 Objetivos

### Objetivo geral

Estimar a participação regional de Lula e de Flávio Bolsonaro no 2º turno de 2026 com um MLG binomial escolhido por regra fixada antes dos resultados ([`docs/pressupostos/01-visao-geral.md`](docs/pressupostos/01-visao-geral.md)).

### Objetivos específicos

* Coletar e consolidar as fontes de dados (TSE, IBGE, CGU) em uma base por município;
* Realizar uma análise exploratória da votação e das covariáveis;
* Comparar famílias e ligações (binomial, quase-binomial, beta-binomial, GLMM, regressão beta);
* Diagnosticar superdispersão, resíduos, influência e autocorrelação espacial;
* Escolher o modelo por validação cruzada leave-one-state-out (LOSO) e regra do 1 erro-padrão;
* Estimar as participações regionais apenas com previsões fora da amostra, com intervalos de incerteza;
* Testar a capacidade de previsão em um backtest (ajuste em 2018, previsão de 2026 a partir de 2022);
* Modelar a dispersão por região e a dependência espacial (SLX, filtro de autovetores, EPs de Conley);
* Comparar as estimativas finais com o resultado real.

---

## 📊 Sobre o Dataset

Fontes (regras de merge e limitações em [`docs/dados/consolidacao.md`](docs/dados/consolidacao.md)):

* **TSE:** votação presidencial 2018, 2022 e 2026 por município; perfil do eleitorado 2022 e 2026; votos em deputado federal e partido do prefeito (ideologia).
* **IBGE:** Censo 2022, PIB dos Municípios, tabela de equivalência IBGE × TSE e malha municipal.
* **Portal da Transparência (CGU):** famílias do Auxílio Brasil (set/2022) e Bolsa Família (set/2026), porque a API do MDS está indisponível.
* **Ideologia dos partidos:** classificação de Bolognesi, Ribeiro e Codato (2023).

Variáveis do modelo final (papéis em `MAPA_2026`, `src/especificacoes.py`):

| Variável | Coluna | Descrição | Fonte |
| --- | --- | --- | --- |
| `y` | `votos_pt_2t_2026` | Votos de Lula no 2º turno de 2026 | TSE |
| `n` | `n_2t_2026` | Votos em Lula + Flávio (só Brasil; brancos, nulos e exterior fora) | TSE |
| `lag` | `logit_pt_2t_2022` | Logit da participação de Lula no 2º turno de 2022 | TSE |
| `pct_dep_esq` | `pct_dep_esq_2026` | Percentual de votos de deputado federal em partidos de esquerda | TSE + Bolognesi et al. (2023) |
| `comparecimento` | `taxa_comparecimento_2t_2026` | Taxa de comparecimento no 2º turno | TSE |
| `log_dens` | `log_densidade_2022` | Log da densidade demográfica | IBGE (Censo 2022) |
| `pct_mulheres` | `pct_mulheres_2026` | Fração de mulheres no eleitorado | TSE (perfil) |
| `pct_16_24` | `pct_16_24_2026` | Fração do eleitorado de 16 a 24 anos | TSE (perfil) |
| `pct_superior` | `pct_superior_completo_2026` | Fração do eleitorado com ensino superior completo | TSE (perfil) |
| `pct_analfabeto` | `pct_analfabeto_2026` | Fração do eleitorado analfabeto | TSE (perfil) |
| `cobertura_pbf` | `cobertura_pbf_2026` | Cobertura do Bolsa Família | CGU + IBGE |
| `pct_vab_agro` | `pct_vab_agro_2021` | Fração do valor adicionado bruto na agropecuária | IBGE (PIB dos Municípios) |
| `pct_evangelicos` | `pct_evangelicos_2022` | Fração de evangélicos | IBGE (Censo 2022) |
| `pct_pretos_pardos` | `pct_pretos_pardos_2022` | Fração de pretos e pardos | IBGE (Censo 2022) |
| `regiao` | `regiao` | Região (referência: Sudeste) | IBGE |

Todos os percentuais são frações em [0, 1]. As demais covariáveis candidatas (`nota_dep`, `delta_comp`, `log_pop`, `pct_60_mais`, `pib_pc`, `pct_urbana`, `bloco_pref`) entram nos modelos candidatos e saem na seleção ([`docs/pressupostos/03-modelos-candidatos.md`](docs/pressupostos/03-modelos-candidatos.md)).

---

## 🎯 Variável resposta

A resposta é o número de votos de Lula no município, dado o total de votos nos dois candidatos:

* $Y_i$ = votos de Lula no município $i$;
* $n_i$ = votos de Lula + votos de Flávio (só Brasil; o exterior, UF `ZZ`, fica fora).

$$
Y_i \sim \mathrm{Bin}(n_i, \pi_i), \qquad g(\pi_i) = x_i^\top \beta, \qquad \pi_{\text{Flávio}} = 1 - \pi_{\text{Lula}}
$$

Participação regional (soma sobre os municípios da região $r$):

$$
\hat p_r = \frac{\sum_i \hat\pi_i\, n_i}{\sum_i n_i}
$$

onde:

* $g$ é a função de ligação (logit no modelo final);
* $x_i$ são as covariáveis do município;
* participações são frações nos dados e pontos percentuais (p.p.) nos relatórios.

**Exemplo.** Dois municípios na mesma região: $n = 10.000$ com $\hat\pi = 0{,}60$ e $n = 30.000$ com $\hat\pi = 0{,}40$. Então $\hat p = (6.000 + 12.000)/40.000 = 45{,}0\%$ para Lula e $55{,}0\%$ para Flávio.

**Casos de borda.**

* Boa Esperança do Norte (MT) não está nas 5.570 linhas (instalado depois do Censo 2022); seus votos (PT 945, Bolsonaro 2.353) são reportados à parte e mudam o Centro-Oeste em −0,004 p.p.;
* o voto no exterior (UF `ZZ`) é descartado;
* brancos e nulos ficam fora de $n_i$;
* com ligação canônica e dummies de região, o total regional **dentro da amostra** é exato por construção; por isso as estimativas regionais vêm só de previsões fora da amostra (LOSO).

---

## 🔎 Variáveis explicativas

O modelo final usa 12 variáveis de média (as do `bb_backward`, escolhido por seleção backward com BIC), mais as mesmas 12 defasadas espacialmente (SLX), mais dummies de região:

```text
logit_pt_2t_2022 (lag)   pct_dep_esq      comparecimento    log_dens
pct_mulheres             pct_16_24        pct_superior      pct_analfabeto
cobertura_pbf            pct_vab_agro     pct_evangelicos   pct_pretos_pardos
W·x de cada uma das 12 (W = vizinhança Queen, padronizada por linha)
```

As covariáveis contínuas são padronizadas só na dobra de treino (sem vazamento na validação cruzada). Dispersão: $\log\theta = Z\gamma$, com dummies de região e $\log n$.

---

## 🧪 Metodologia

O desenvolvimento seguiu os cinco notebooks. Cada parte tem seu documento em [`docs/pressupostos/`](docs/pressupostos/README.md).

### 1. Análise exploratória (`01-AED`, Partes 1 a 4)

* Dados, resposta e covariáveis; mapas coropléticos;
* A resposta de 2026 contra 2022 tem inclinação 0,96 (OLS), o que justifica estimar o coeficiente da defasagem em vez de fixá-lo em 1;
* Colinearidade (VIF) das covariáveis.

### 2. Modelos candidatos e diagnóstico (`02-MLG`, Partes 5 a 10)

* Binomial com cinco ligações, quase-binomial (φ de Pearson próprio), beta-binomial, GLMM com PQL próprio, regressão beta e linhas de base (OLS e WLS sobre o logit empírico);
* Defasagem livre, offset, spline, defasagem por região e efeito de UF;
* Diagnóstico: resíduos quantílicos, envelope meio-normal, alavanca e distância de Cook, I de Moran.

### 3. Validação cruzada e regra de seleção (`03-validacao`, Partes 11 e 12)

Regra fixada antes dos resultados:

1. menor erro absoluto médio da participação por UF no LOSO;
2. regra do 1 erro-padrão: entre os modelos dentro de 1 EP do melhor, o de menos parâmetros;
3. o MAE ponderado por votos do 5-fold é critério secundário.

O erro regional LOSO **não** é critério, porque favorece por construção as binomiais canônicas. As estimativas regionais usam previsões LOSO, intervalos por simulação (B = 1000, semente 7) e um choque de UF $\sigma_{UF}$.

### 4. Backtest (`04-backtest`, Parte 13)

Ajuste em 2022 (com 2018 como defasagem) e previsão de 2026 a partir de 2022, com e sem os dados do 1º turno de 2026 e com recalibração do intercepto.

### 5. Modelo final (`05-modelo-final`, Partes 14 a 17)

* **Dispersão por região e log n** (`bb_disp`), confirmada por teste de Wald robusto;
* **Dependência espacial:** filtro de autovetores (rejeitado), SLX (aceito) e erros-padrão de Conley a 200 km;
* **Estimativas regionais e por UF** do modelo final e cobertura dos intervalos.

---

## 🛠️ Tecnologias utilizadas

* **Python** (≥ 3.12) e **uv**
* **Jupyter Notebook**
* **Pandas**, **NumPy** e **PyArrow**
* **Plotly** (gráficos e mapas interativos)
* **SciPy**
* **Statsmodels** (GLM, beta-binomial)
* **Scikit-learn**
* **GeoPandas** e **geobr** (malha municipal)
* **PySAL** (`libpysal`, `esda`)
* **pytest**

---

## 📁 Estrutura do projeto

```text
.
├── src/          # coleta (src/coleta/), especificações dos modelos, modelos próprios, backtest
├── tests/        # testes unitários
├── notebooks/    # 01-AED, 02-MLG, 03-validacao, 04-backtest, 05-modelo-final
├── data/         # raw (cache por fonte), interim (um parquet por fonte), processed, referencia
├── results/      # cv, figures, tables
├── docs/         # dados (fontes) e pressupostos (decisões e resultados)
└── README.md
```

### `src/`

Coleta (`src/coleta/`, um módulo por fonte e o consolidador), especificações dos modelos (`especificacoes.py`: todo modelo é ajustado por `Candidato.ajustar`, reusado pelos notebooks 03 e 04), modelos próprios (quase-binomial, GLMM por PQL), componentes espaciais e backtest.

### `data/`

Dados baixados ficam em cache em `data/raw/<fonte>/`; cada fonte grava um `data/interim/<fonte>.parquet`; a consolidação grava `data/processed/municipios.parquet` (5.570 linhas).

### `notebooks/`

Rode nesta ordem:

1. `01-AED.ipynb`: análise exploratória (Partes 1 a 4);
2. `02-MLG.ipynb`: modelos candidatos e diagnóstico (Partes 5 a 10);
3. `03-validacao.ipynb`: validação cruzada e estimativas regionais (Partes 11 e 12);
4. `04-backtest.ipynb`: backtest (Parte 13);
5. `05-modelo-final.ipynb`: dispersão, dependência espacial e modelo final (Partes 14 a 17).

### `results/`

Previsões fora da amostra (`cv/`), gráficos interativos em HTML (`figures/`) e tabelas em CSV (`tables/`).

### `docs/`

* `docs/dados/`: fontes, regras de consolidação e dicionário;
* `docs/pressupostos/`: decisões, pressupostos e resultados de cada etapa; comece pelo [índice](docs/pressupostos/README.md).

### Reprodução

```bash
uv sync
# Coleta: o crosswalk TSE↔IBGE vem primeiro, porque Bolsa Família e a consolidação dependem dele.
uv run --no-sync python -m src.coleta.ibge_municipios
uv run --no-sync python -m src.coleta.tse_presidente
uv run --no-sync python -m src.coleta.tse_ideologia
uv run --no-sync python -m src.coleta.tse_perfil
uv run --no-sync python -m src.coleta.ibge_censo_pib
uv run --no-sync python -m src.coleta.mds_bolsa_familia
uv run --no-sync python -m src.coleta.consolidar   # grava data/processed/municipios.parquet
uv run --no-sync pytest
uv run --no-sync jupyter lab notebooks/
```

O caminho do repositório tem espaço: coloque-o entre aspas.

---

## 📈 Resultados

Os resultados abaixo são do modelo final (Parte 17, notebook `05-modelo-final.ipynb`). Detalhes em [`docs/pressupostos/11-modelo-final.md`](docs/pressupostos/11-modelo-final.md).

### Modelo ajustado

$$
Y_i \sim \mathrm{BetaBin}(n_i, \pi_i, \theta_i), \quad
\mathrm{logit}(\pi_i) = \beta_0 + \sum_{k=1}^{12} \beta_k x_{ik} + \sum_{k=1}^{12} \delta_k (Wx)_{ik} + \text{região}_i, \quad
\log \theta_i = Z_i \gamma
$$

* beta-binomial com ligação logit; 37 parâmetros;
* $W$ é a matriz de vizinhança Queen padronizada por linha; $(Wx)_{ik}$ é a média da variável $k$ nos vizinhos (SLX);
* $Z_i$ contém dummies de região e $\log n_i$;
* os erros-padrão são os de Conley a 200 km (corte conferido com o correlograma de 2022, que também dá 200 km), cerca de 1,8 vez os do modelo.

### Superdispersão

```text
φ de Pearson (binomial) = 21,7
beta-binomial: θ ≈ 630, ρ ≈ 0,0016
DP dos resíduos quantílicos: 1,03 (beta-binomial) contra 3,43 (binomial)
```

### Seleção por validação cruzada (LOSO, MAE por UF em p.p.)

```text
bb_backward = 1,0004   (escolhido pela regra do 1 EP: 20 parâmetros; 12 modelos dentro de 1 EP)
bb_disp     = 0,9951   (dispersão por região e log n)
bb_slx      = 1,0224   (modelo final)
bb_esf      = 1,4460   (rejeitado)
limite de aceitação = 1,1308
```

### Dispersão regional

```text
Wald robusto (Conley): dummies de região p = 1,5e-11 -> mantidas; log n p = 4,2e-16 -> mantido
max |DP − 1| por região: 0,409 -> 0,095
```

### Dependência espacial

```text
I de Moran dos resíduos: 0,339 (sem termo espacial) -> 0,333 (SLX)
filtro de autovetores (k = 50): I de Moran 0,227, mas LOSO 1,446 > 1,1308 -> rejeitado
```

O SLX é aceito pela regra de validação, mas quase não reduz o I de Moran: a autocorrelação espacial residual persiste e os EPs de Conley corrigem a inferência, não as estimativas pontuais ([`docs/pressupostos/10-dependencia-espacial.md`](docs/pressupostos/10-dependencia-espacial.md)).

### Estimativas regionais contra o real

Participação de Lula (%), LOSO com Conley e $\sigma_{UF}$ = 0,052; intervalo de 95% (Brasil: 90%):

| Região | Estimado | Intervalo | Real | Erro (p.p.) |
| --- | --- | --- | --- | --- |
| Centro-Oeste | 40,04 | [38,64; 41,51] | 39,99 | 0,05 |
| Nordeste | 69,41 | [68,54; 70,22] | 69,37 | 0,04 |
| Norte | 49,52 | [48,08; 50,90] | 50,01 | −0,48 |
| Sudeste | 47,86 | [46,30; 49,40] | 47,24 | 0,62 |
| Sul | 37,01 | [35,77; 38,51] | 37,42 | −0,41 |
| Brasil | 51,95 | [51,30; 52,58] (90%) | 51,78 | 0,17 |

Erro regional absoluto médio de 0,321 p.p. (0,357 em `bb_backward`). São só 5 regiões e o erro regional LOSO não escolhe modelos: é descrição, não evidência de superioridade.

### Cobertura dos intervalos

| Covariância + σ | UF 95% | Região 95% |
| --- | --- | --- |
| Conley + σ_UF | 100% (27/27) | 5/5 |
| Conley + σ = 0 | 59,3% (16/27) | 3/5 |

O choque de UF é o que sustenta a cobertura. Como $\sigma_{UF}$ é ajustado nas mesmas UFs em que a cobertura é conferida, ela é **otimista**.

### Backtest

Melhor variante: `C_recal` (participações do 1º turno de 2026, intercepto ancorado na participação nacional de 2022), com erro de 0,58 p.p. por UF e 0,68 p.p. por região. A linha de base ingênua (repetir 2022) dá 1,65 (UF) e 0,64 (região). Sem o 1º turno, os modelos ficam piores que a linha de base (UF de 3,3 a 3,9) ([`docs/pressupostos/08-backtest.md`](docs/pressupostos/08-backtest.md)).

### Limitações

* **Falácia ecológica:** o modelo descreve municípios, não eleitores;
* **Autocorrelação espacial residual** (I de Moran 0,333) em todos os modelos; Conley corrige a inferência, não as estimativas pontuais;
* **Cobertura otimista:** $\sigma_{UF}$ vem das mesmas UFs; falta validação aninhada;
* **Seleção:** 12 modelos ficam dentro de 1 EP, e a escolha repousa na parcimônia, não numa vantagem demonstrada;
* **Regras tardias:** o Wald robusto, o SLX e o corte de 2022 foram fixados antes de cada resultado, mas não antes do notebook 05 inteiro;
* **Dados:** Bolsa Família 2022 cerca de 2,8% abaixo da referência oficial; `pct_vab_agro` usa 2021;
* **Envelope meio-normal** de `bb_reglag` com 93,9% dos pontos fora (caudas pesadas): não valida a distribuição.

Pendências abertas, com efeito e correção possível: [`docs/pressupostos/07-pendencias.md`](docs/pressupostos/07-pendencias.md).

---

## 📚 Referências

* Bolognesi, B.; Ribeiro, E.; Codato, A. (2023). Uma nova classificação ideológica dos partidos políticos brasileiros. *Dados*, 66(2). https://doi.org/10.1590/dados.2023.66.2.303
* McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models* (2ª ed.). Chapman & Hall.
* Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics*, 31(4), 949-952 (beta-binomial).
* Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
* Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
* Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
* Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
* Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review*, 15(3), 351-357.
* Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
* Tiefelsdorf, M.; Griffith, D. A. (2007). Semiparametric filtering of spatial autocorrelation: the eigenvector approach. *Environment and Planning A*, 39(5), 1193-1221.
* Breslow, N. E.; Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *Journal of the American Statistical Association*, 88(421), 9-25.
* Rey, S. J.; Anselin, L. (2010). PySAL: a Python library of spatial analytical methods. In *Handbook of Applied Spatial Analysis*. Springer, 175-193.
* TSE. Dados abertos (resultados, perfil do eleitorado e candidaturas). https://dadosabertos.tse.jus.br
* IBGE. Censo Demográfico 2022 e PIB dos Municípios. https://www.ibge.gov.br
* CGU. Portal da Transparência (Auxílio Brasil e Bolsa Família). https://portaldatransparencia.gov.br

Lista completa e conferida: [`docs/pressupostos/referencias.md`](docs/pressupostos/referencias.md).

---

## 👥 Autores

**José de Ribamar Ferreira Neto**

---

## 🎓 Contexto acadêmico

Projeto desenvolvido como parte das atividades da disciplina de **Modelagem Estatística**, com foco na aplicação de **Modelos Lineares Generalizados (MLGs)**, inferência estatística e dependência espacial.
