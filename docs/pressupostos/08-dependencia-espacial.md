# 08 - Dependência espacial: filtro de autovetores, SLX e EPs de Conley (Partes 14 e 15 do notebook 05)

Notebook `notebooks/05-modelo-final.ipynb`. Tabelas em `results/tables/final_esf_k.csv`, `final_slx.csv`, `final_cv.csv`, `final_decisao.json`, `final_correlograma.csv`, `final_correlograma_2022.csv`, `final_ep.csv` e `final_ep_corte.csv`; figuras `results/figures/final_moran_k.html` e `final_correlograma.html`. A dispersão (Parte 13) está em [07-dispersao-regional.md](07-dispersao-regional.md); as estimativas do modelo final (Parte 16) em [09-modelo-final.md](09-modelo-final.md).

**Objetivo.** Os resíduos de todos os modelos têm autocorrelação espacial (I de Moran ≈ 0,34; ver [diagnóstico, seção 6](04-diagnostico.md) e [pendências](12-pendencias.md)), o que torna os EPs pequenos demais. Duas respostas: (a) tirar a dependência do preditor, com um filtro espacial ou com covariáveis dos vizinhos (Parte 14); (b) manter o preditor e corrigir só os EPs (Parte 15).

## 1. Regra comum de aceitação, escrita antes do resultado

Fixada em 2026-11-09, antes de ver os resultados das extensões. Toda extensão mantém os termos de média de `bb_selecao` e é aceita só se (1) o seu diagnóstico melhora **e** (2) o MAE por UF no LOSO é no máximo o limite L = MAE por UF aninhado de `bb_selecao` + EP do melhor modelo LOSO = 1,1168 + 0,1347 = **1,2515 p.p.** (1,1695 com o 1,0348 sem aninhar; antes, com o LOSO que vazava o 2º turno da UF retida, 1,1308; mesmo limite do [doc 07](07-dispersao-regional.md), seção 4). Razão: o ganho de diagnóstico não pode custar previsão fora da amostra, que é o objetivo do projeto. As métricas são ponderadas por n̂.

## 2. Filtro espacial de autovetores de Moran, ESF (Parte 14)

- **Construção.** Autovetores de M·C·M, com C a matriz de vizinhança rainha binária e simétrica (ilhas ligadas ao vizinho mais próximo, k = 1) e M o projetor que centra e retira as covariáveis. Entram como regressores extras no preditor da média (Griffith, 2003; Tiefelsdorf e Griffith, 2007).
- **Por que é legítimo na validação.** Os autovetores dependem só da geometria, não de y; usá-los para uma UF deixada de fora não vaza resposta.
- **Número de autovetores:** k ∈ {0, 10, 25, 50, 100}, escolhido por BIC. Diagnóstico da regra: queda do I de Moran dos resíduos.

| k | log-verossimilhança | parâmetros | BIC | I de Moran dos resíduos |
|---|---|---|---|---|
| 0 | -36087,5 | 24 | 72382,1 | 0,342 |
| 10 | -36011,1 | 34 | 72315,5 | 0,319 |
| 25 | -35820,7 | 49 | 72064,0 | 0,273 |
| 50 | -35651,5 | 74 | 71941,3 | 0,230 |
| 100 | -35530,3 | 124 | 72130,1 | 0,197 |

- **BIC escolhe k = 50.** O I de Moran cai de 0,342 para 0,230 (p simulado 0,001 em todos os k): o filtro tira parte da dependência, mas não toda.
- **Condição 1 cumprida** (I cai). **Condição 2 falha:** MAE por UF no LOSO = 1,484 p.p. (ponderado 4,78) contra o limite 1,2515. Na validação de 5 dobras o filtro melhora (0,645 contra 0,855).
- **Combinação `bb_final`** (dispersão + ESF): LOSO 1,454, também acima do limite.
- **Razão da divergência [deduced].** Numa UF deixada de fora, os coeficientes dos autovetores extrapolam um padrão espacial ajustado nas outras 26 UFs; nas 5 dobras os vizinhos continuam no treino, e o padrão local se transfere. O filtro captura estrutura que a validação por UF mostra não ser transferível.
- **Decisão: ESF rejeitado** (`final_decisao.json`, `aceitas.esf = false`). Com só o ESF fora, o modelo final seria `bb_disp`; ele foi superado pelo SLX (seção 3).

## 3. Covariáveis defasadas espacialmente, SLX (Parte 14)

O SLX acrescenta ao preditor as covariáveis médias dos vizinhos, sem termo autorregressivo (Halleck Vega e Elhorst, 2015); por isso o ajuste e a previsão por UF seguem sem iteração.

**Regra acrescentada em 2026-11-09, depois dos primeiros resultados do notebook**, e fixada antes de ver o resultado do SLX. Razão: o ESF falhou no LOSO porque carrega para a UF deixada de fora um padrão espacial ajustado em y. O SLX usa W·x, as covariáveis dos vizinhos, que são conhecidas numa UF nova; não carrega padrão de y.

- **Construção.** W = rainha com linhas padronizadas (ilhas ligadas ao vizinho mais próximo). Opções, escolhidas por BIC: nenhuma, W·lag, ou W·(as 12 variáveis do modelo-base). W·pib estaria na escala log.
- **Aceitação.** A mesma regra da seção 1: o I de Moran dos resíduos cai **e** o MAE por UF no LOSO é no máximo 1,2515 p.p. Aceito, o modelo final passa a `bb_disp` + SLX (`bb_slx`).

| opção | BIC | I de Moran | parâmetros |
|---|---|---|---|
| nenhuma | 72382,1 | 0,3424 | 24 |
| W·lag | 72375,4 | 0,3399 | 25 |
| W·12 variáveis | **72175,5** | 0,3361 | 36 |

- **Resultado.** O BIC escolhe W·12 variáveis. LOSO: MAE por UF de `bb_slx` = 1,078 ≤ 1,2515, **aceito** (`aceitas.slx = true`, `modelo_final = "bb_slx"`). Esse valor fica **acima** do de `bb_selecao` (1,035) e de `bb_disp` (1,0345): o SLX custa previsão, dentro da tolerância. 5 dobras: 0,846.
- **Leitura honesta.** A regra não fixou queda mínima do I de Moran. A queda é de 0,006 (0,342 para 0,336): o SLX quase não muda a estrutura espacial dos resíduos, e a aceitação vem do critério formal (I cai e LOSO dentro do limite), não de um ganho material. O fechamento do item (decomposição do I e testes de krigagem e CAR nos resíduos, ambos rejeitados) está no [doc 10](10-extensoes-rejeitadas.md). O BIC melhora em 207, mas esse BIC supõe independência. A dependência residual segue tratada por Conley; ver [pendências](12-pendencias.md).

## 4. EPs espaciais de Conley (Parte 15)

Como o ESF não foi aceito, a dependência residual permanece e é tratada só na inferência. Os números desta seção são do modelo final `bb_slx` (36 parâmetros: 30 de β e 6 de γ da dispersão).

- **Estimador.** Covariância HAC espacial de Conley (1999): núcleo de Bartlett sobre a distância de círculo máximo entre centroides, aplicada ao vetor completo de parâmetros. Os pontos estimados não mudam.
- **Por que Conley é a correção principal e não o agrupamento.** Os dados não têm códigos de microrregião; agrupar por UF (27 aglomerados) é um limite conservador, não o desenho natural.
- **Verificação da implementação.** Com corte ≈ 0 a covariância iguala a HC0; a covariância por aglomerado iguala a do statsmodels (diferença de 1e-15).

### Corte, fixado antes de ver os EPs

Regra: o corte é o limite inferior da primeira faixa do correlograma cuja correlação dos resíduos quantílicos é < 0,05. Faixas em km: 0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000.

| faixa (km) | 0-10 | 10-25 | 25-50 | 50-75 | 75-100 | 100-150 | 150-200 | 200-300 | 300-400 |
|---|---|---|---|---|---|---|---|---|---|
| correlação | 0,376 | 0,339 | 0,234 | 0,166 | 0,124 | 0,091 | 0,062 | 0,030 | 0,005 |

A primeira faixa abaixo de 0,05 é 200-300 km: **corte = 200 km.** Sensibilidade: 100 km (×0,5) e 400 km (×2). Limite conservador: agrupamento por UF com correção de amostra pequena.

### Resultado

Razão entre o EP e o EP do modelo, sobre os 30 coeficientes de β:

| EP | mediana | mínimo | máximo |
|---|---|---|---|
| Conley, 100 km | 1,44 | 0,97 | 2,38 |
| **Conley, 200 km** | **1,69** | 0,95 (`bloco_outro`) | 3,49 (`reg_Sul`) |
| Conley, 400 km | 1,78 | 0,92 | 4,85 |
| Aglomerado por UF | 2,00 | 0,72 | 6,07 |

- **O EP do modelo é pequeno demais por um fator de cerca de 1,7** (variância cerca de 3 vezes maior sob Conley a 200 km). Nos 6 γ da dispersão a razão vai de 1,75 a 2,56.
- **Significância a |z| > 1,96 perdida sob Conley a 200 km (4 de 36):** `w_pct_vab_agro`, `bloco_esquerda`, `reg_Centro-Oeste` e `reg_Nordeste+Norte`. Nenhuma foi ganha.
- **Sensibilidade ao corte.** Com 100 km perdem-se 3 (`w_pct_vab_agro`, `reg_Centro-Oeste`, `reg_Nordeste+Norte`); com 400 km, 5 (as 4 acima mais `w_pct_dep_esq`); por UF, 9. A ordem é monótona no corte, e a leitura qualitativa (efeitos fracos ou regionais frágeis, efeitos grandes firmes) não muda.
- **Uso.** Os EPs de Conley a 200 km entram nas simulações da [Parte 16](09-modelo-final.md), onde a covariância de β alarga os intervalos.

## 5. Estabilidade do corte de Conley em 2022 (Parte 15)

**Regra acrescentada em 2026-11-09, depois dos primeiros resultados do notebook**, fixada antes do resultado. Ajusta-se a especificação final em 2022 (lag = 2018) e aplica-se a mesma regra do correlograma. Se 0,5·CORTE ≤ CORTE_2022 ≤ 2·CORTE, mantém-se o CORTE de 2026; senão usa-se o maior.

- **Resultado.** A correlação de 2022 cai abaixo de 0,05 na faixa 200-300 km (0,029; a faixa anterior, 150-200 km, tem 0,084). CORTE_2022 = 200 km = CORTE_2026, dentro de [100, 400], então o corte de **200 km é mantido** (`corte_final` em `final_decisao.json`). Fecha o item de pendências sobre o corte.
- **Sensibilidade do EP ao corte** (`final_ep_corte.csv`, 50 a 800 km). A razão mediana Conley/modelo sobe de 1,19 (50 km) para 1,69 (200 km) e satura perto de 1,9 a partir de 600 km; o máximo vai de 1,64 a 4,85 (400 km) e se mantém perto de 5 acima. A escolha de 200 km fica no meio da faixa de crescimento, não no extremo.

## 6. Limites

- O I de Moran dos resíduos permanece em 0,336 no modelo final (`bb_slx`): Conley corrige a inferência, não as estimativas pontuais. A decomposição do I e os testes de krigagem e CAR dos resíduos estão no [doc 10](10-extensoes-rejeitadas.md).
- O corte foi conferido contra 2022 (seção 5); a evidência vem de duas eleições, não de uma.
- O p-valor do LRT do [doc 07](07-dispersao-regional.md) supõe municípios independentes, o que a correlação residual contradiz. A diferença de BIC, o ganho de DP e o teste de Wald robusto ([doc 07](07-dispersao-regional.md), seção 5) não dependem desse p-valor.

## Referências

- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*. Springer.
- Halleck Vega, S.; Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55(3), 339-363.
- Tiefelsdorf, M.; Griffith, D. A. (2007). Semiparametric filtering of spatial autocorrelation: the eigenvector approach. *Environment and Planning A*, 39(5), 1193-1221.
- Cameron, A. C.; Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
