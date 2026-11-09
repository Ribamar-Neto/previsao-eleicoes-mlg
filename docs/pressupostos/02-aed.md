# Pressupostos da AED (`notebooks/01-AED.ipynb`)

Células contadas a partir de 0 (markdown e código). Funções auxiliares: `notebooks/graficos.py`.
Figuras: `results/figures/aed_*.html`. Tabela: `results/tables/vif_covariaveis.csv`.
`matplotlib` e `kaleido` não estão no ambiente: os mapas são coropléticos Plotly (HTML), com geometria
simplificada (`simplify(0.12)`, coordenadas com 2 casas) para o caderno ficar com cerca de 14 MB. O mapa serve para ver o
padrão regional, não o contorno exato.

## Parte 1 — Dados (células 2–6)
- **O que**: `pd.read_parquet` de `municipios.parquet` (5.570 × 161 após `log_pib_pc_2023`); resultados nacionais, regionais e por UF vindos de `resultado_regional.csv` e `resultado_uf.csv` (soma das contagens, nunca média de proporções).
- **Pressuposto**: cada município é uma observação binomial `Y ~ Bin(n, p)`, com `n = votos_pt + votos_bolsonaro` do 2º turno (brancos, nulos e exterior fora).
- **Conclusão**: PT nacional em 2026 = 51,78 % (checagem da soma por região). `n_2t_2026` varia de 848 a 6,96 milhões, mediana 7.201 (célula 5). A variância
  binomial depende fortemente de `n`: justifica GLM binomial com pesos naturais em vez de OLS sobre a proporção.

## Parte 2 — Resposta (células 7–15)
- **Histogramas** (`graficos.histograma`, 50 classes; célula 8): média sem peso 0,537 contra 0,518 ponderada. A diferença vem do peso dos municípios grandes; por isso as métricas nacionais e regionais têm de ser somadas por votos.
- **Mapas** (células 10–11): `prop_pt_2t_2026` (centro em 0,5) e `swing_logit_2026` (escala simétrica, limite no percentil 99 de |swing|). O mapa da proporção mostra bloco Nordeste/Norte forte; o swing é mais disperso e regional.
- **Relação com o turno anterior** (células 13–14; `statsmodels` OLS e WLS com peso `n`, LOWESS `frac=2/3` do `statsmodels`, que dá uma curva de tendência sem forma imposta):

  | Relação | Inclinação OLS | R² OLS | Inclinação WLS | R² WLS |
  |---|---|---|---|---|
  | logit 2026 ~ logit 2022 | 0,962 | 0,976 | 0,929 | 0,969 |
  | logit 2022 ~ logit 2018 | 0,804 | 0,952 | 0,753 | 0,940 |

- **Conclusão**: inclinações abaixo de 1 indicam regressão à média (municípios extremos se aproximam do centro), mais forte em 2022 ← 2018 (houve mudança de candidato) do que em 2026 ← 2022 (mesma candidatura de Lula). Isso valida
  `logit_pt_2t_2022` como covariável central e justifica estimar o coeficiente em vez de fixá-lo em 1 (modelo com offset). O swing médio ponderado por região em 2026 é pequeno
  (Sudeste +0,06, Norte +0,04, Sul −0,03, Nordeste −0,02, Centro-Oeste +0,01 log-odds), logo o efeito regional é fraco, mas existe.

## Parte 3 — Covariáveis (células 16–27)
- **Escopo**: 24 numéricas (com `log(pib_per_capita_2023)`) mais duas indicadoras de `bloco_prefeito_2024` (esquerda e centro; base: direita e sem prefeito) e o fator `regiao` (tabelas na célula 17).
  Zero valores ausentes.
- **Linearidade** (células 19 e 21; `logit_empirico_por_decil` e `media_ponderada_por_decil`, 10 classes por posto, empates desfeitos por ordem): o logit empírico do decil é `log(ΣY / Σ(n−Y))`, que é o que o preditor linear do GLM tenta reproduzir.
  O eixo y do swing usa a média ponderada por `n`. Pressuposto: o preditor linear no logit é correto; curvatura em decis indica transformação ou termo extra.
  Resultado: `logit_pt_2t_2022`, `cobertura_pbf_2026`, `pct_evangelicos_2022` e `pct_urbana_2022` têm relação quase monotônica e aproximadamente linear no logit
  (por exemplo o logit de 2026 vai de −0,96 a 1,58 ao longo dos decis de `logit_pt_2t_2022`); `pct_dep_esq_2026` tem forma de U, `log_populacao_2022` e `pct_vab_agro_2021` têm forma de U invertido.
  No swing, que é a relação parcial dada a defasagem, os efeitos são pequenos (|média| ≲ 0,2) e em geral monotônicos fracos: os cinco com maior amplitude são `log_populacao_2022` (de −0,18 para +0,08), `pct_evangelicos_2022`, `pct_urbana_2022`, `pct_vab_agro_2021` e `pct_indigenas_2022`.
  Decisão: manter termos lineares como base e testar `pct_dep_esq_2026`, `log_populacao_2022` e `pct_vab_agro_2021` com termo quadrático ou spline no caderno de modelos (`02-MLG`); não impor a transformação antes de ver os resíduos.
- **Correlação de Spearman** (célula 23; `DataFrame.corr(method="spearman")`, robusto a assimetria e outliers): pares com |ρ| ≥ 0,7 incluem
  `log_populacao_2022`–`log_eleitorado_2026` (0,99), `pct_evangelicos_2022`–`pct_catolicos_2022` (−0,92), `pct_analfabeto_2026`–`taxa_alfabetizacao_15mais_2022` (−0,90), `pct_16_24_2026`–`pct_60_mais_2026` (−0,84) e `cobertura_pbf_2026` com várias (alfabetização, defasagem, jovens).
- **VIF** (célula 25; `variance_inflation_factor` com covariáveis padronizadas e intercepto; limiar: 10 severo, 5–10 atenção). Não inclui o fator `regiao`.

  | Lista | VIF máximo | Maiores |
  |---|---|---|
  | Cheia (26 colunas) | 124,2 (`log_populacao_2022`) | `log_eleitorado_2026` 121,4; `pct_catolicos_2022` 38,5; `pct_evangelicos_2022` 21,0; `taxa_alfabetizacao_15mais_2022` 8,8 |
  | Reduzida (22 colunas) | 6,9 (`pct_16_24_2026`) | `pct_ate_fund_incompleto_2026` 6,2; `cobertura_pbf_2026` 5,7; `pct_60_mais_2026` 5,5; `logit_pt_2t_2022` 5,0 |

- **Lista recomendada** (células 25 e 27): `logit_pt_2t_2022`, `pct_dep_esq_2026`, `taxa_comparecimento_2t_2026`, `delta_comparecimento_2t_2026`, `log_populacao_2022`, `log_densidade_2022`, `pct_mulheres_2026`,
  `pct_16_24_2026`, `pct_60_mais_2026`, `pct_superior_completo_2026`, `pct_analfabeto_2026`, `pct_ate_fund_incompleto_2026`, `cobertura_pbf_2026`, `log(pib_per_capita_2023)`, `pct_vab_agro_2021`,
  `pct_evangelicos_2022`, `pct_sem_religiao_2022`, `pct_pretos_pardos_2022`, `pct_indigenas_2022`, `pct_urbana_2022`, `bloco_prefeito_2024`, `regiao`.
- **Descartadas e motivo**:
  - `pct_catolicos_2022`: as três religiões somam quase 1 (compositional); com duas delas a terceira é determinada. Mantidas evangélicos e sem religião, que carregam a variação.
  - `log_eleitorado_2026`: ρ = 0,99 com `log_populacao_2022`; a população é do Censo e não depende da eleição.
  - `taxa_alfabetizacao_15mais_2022`: mede o mesmo que `pct_analfabeto_2026` (ρ = −0,90), este é do eleitorado e mais recente.
  - `nota_media_dep_2026`: mede o mesmo que `pct_dep_esq_2026` (VIF 2,9 e 2,4 na lista cheia); preferida a fração, que é mais interpretável e já é parcela binomial.
- **Atenção**: cinco variáveis ficam entre 5 e 7 de VIF (moderado). Aceitável para previsão; para inferência, conferir a estabilidade dos coeficientes. O fator `bloco_prefeito_2024` tem só 49 municípios no centro e 16 sem prefeito (agrupar com a direita se der instabilidade).

## Parte 4 — Espacial (células 28–31)
- **Pesos** (`graficos.pesos_queen`): Queen de `libpysal`, ilhas ligadas ao vizinho mais próximo (`KNN k=1`, `attach_islands`), padronização por linha. Alinhado a `municipios.parquet` por `id_municipio`. Média de 5,79 vizinhos.
- **Moran's I global** (`esda.Moran`, 999 permutações, semente 2026; célula 29): pressupõe estacionariedade e a mesma matriz de pesos para todos.

  | Variável | I | z (permutação) | p |
  |---|---|---|---|
  | `prop_pt_2t_2026` | 0,896 | 108,6 | 0,001 |
  | `swing_logit_2026` | 0,542 | 66,5 | 0,001 |

  p = 0,001 é o mínimo possível com 999 permutações.
- **LISA** (`esda.Moran_Local`, 999 permutações, p < 0,05 sem correção para múltiplos testes; célula 30) do swing: 3.508 não significativos, 1.045 Alto-Alto, 844 Baixo-Baixo, 90 Alto-Baixo, 83 Baixo-Alto.
- **Conclusão**: o voto tem dependência espacial muito forte e o swing, mesmo já descontando o nível anterior, ainda tem I = 0,54: a defasagem e o fator `regiao` não bastam.
  Os resíduos dos GLMs devem ser testados por Moran antes de confiar em erros-padrão clássicos; se a dependência persistir, usar erros robustos por UF, termo espacial ou defasagem espacial do swing como covariável.
  O LISA sem correção superestima agrupamentos; use-o descritivamente.

## Referências
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*.
- Anselin, L. (1995). Local indicators of spatial association (LISA). *Geographical Analysis*.
- Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. *JASA*.
- Belsley, D. A., Kuh, E. e Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. Wiley (fator de inflação da variância).
- Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*.
- McCullagh, P. e Nelder, J. A. (1989). *Generalized Linear Models*, 2ª ed. Chapman & Hall (GLM binomial e logit empírico).
- Rey, S. J. e Anselin, L. (2010). PySAL: a Python library of spatial analytical methods. In *Handbook of Applied Spatial Analysis*.
