# Dependência espacial: filtro de autovetores e EPs de Conley (Partes 15 e 16 do notebook 05)

Células citadas contam a partir de 0 em `notebooks/05-modelo-final.ipynb` (7 a 21). Tabelas em `results/tables/final_esf_k.csv`, `final_slx.csv`, `final_cv.csv`, `final_decisao.json`, `final_correlograma.csv`, `final_correlograma_2022.csv`, `final_ep.csv` e `final_ep_corte.csv`; figuras `results/figures/final_moran_k.html` e `final_correlograma.html`. A dispersão (Parte 14) está em [09-dispersao-regional.md](09-dispersao-regional.md); as estimativas do modelo final (Parte 17) em [11-modelo-final.md](11-modelo-final.md).

**Objetivo.** Os resíduos de todos os modelos têm autocorrelação espacial (I de Moran ≈ 0,34; ver [diagnóstico, seção 6](04-diagnostico.md) e [pendências](07-pendencias.md)), o que torna os EPs pequenos demais. Duas respostas: (a) tirar a dependência do preditor com um filtro espacial (Parte 15); (b) manter o preditor e corrigir só os EPs (Parte 16).

## 1. Regra comum de aceitação, escrita antes do resultado (célula 12)

Fixada em 2026-11-09, antes de ver os resultados das extensões. Toda extensão mantém os 12 termos de β de `bb_backward` e é aceita só se (1) o seu diagnóstico melhora **e** (2) o MAE por UF no LOSO é no máximo o de `bb_backward` mais 1 EP: 1,0004 + 0,1304 = **1,1308 p.p.** (mesmo limite do [doc 09](09-dispersao-regional.md), seção 4). Razão: o ganho de diagnóstico não pode custar previsão fora da amostra, que é o objetivo do projeto.

## 2. Filtro espacial de autovetores de Moran, ESF (células 7 a 9)

- **Construção.** Autovetores de M·C·M, com C a matriz de vizinhança rainha binária e simétrica (ilhas ligadas ao vizinho mais próximo, k = 1) e M o projetor que centra e retira as covariáveis. Entram como regressores extras no preditor da média (Griffith, 2003; Tiefelsdorf e Griffith, 2007).
- **Por que é legítimo na validação.** Os autovetores dependem só da geometria, não de y; usá-los para uma UF deixada de fora não vaza resposta.
- **Número de autovetores:** k ∈ {0, 10, 25, 50, 100}, escolhido por BIC. Diagnóstico da regra: queda do I de Moran dos resíduos.

| k | log-verossimilhança | parâmetros | BIC | I de Moran dos resíduos |
|---|---|---|---|---|
| 0 | -36064,6 | 25 | 72344,9 | 0,339 |
| 10 | -35997,4 | 35 | 72296,7 | 0,319 |
| 25 | -35802,7 | 50 | 72036,7 | 0,271 |
| 50 | -35631,5 | 75 | 71910,0 | 0,227 |
| 100 | -35511,4 | 125 | 72101,0 | 0,192 |

- **BIC escolhe k = 50.** O I de Moran cai de 0,339 para 0,227 (p simulado 0,001 em todos os k): o filtro tira parte da dependência, mas não toda.
- **Condição 1 cumprida** (I cai). **Condição 2 falha:** MAE por UF no LOSO = 1,446 p.p. (ponderado 5,17) contra o limite 1,1308. Na validação de 5 dobras o filtro melhora.
- **Combinação `bb_final`** (dispersão + ESF): LOSO 1,412, também acima do limite.
- **Razão da divergência [deduced].** Numa UF deixada de fora, os coeficientes dos autovetores extrapolam um padrão espacial ajustado nas outras 26 UFs; nas 5 dobras os vizinhos continuam no treino, e o padrão local se transfere. O filtro captura estrutura que a validação por UF mostra não ser transferível.
- **Decisão: ESF rejeitado** (`final_decisao.json`, `aceitas.esf = false`). Com só o ESF fora, o modelo final seria `bb_disp`; ele foi superado pelo SLX (seção 5).

## 3. EPs espaciais de Conley (células 16 a 21)

Como o ESF não foi aceito, a dependência residual (I de Moran 0,339) permanece e é tratada só na inferência. Os números desta seção (25 parâmetros, razões de EP) são de `bb_disp`, medidos antes do SLX; os do modelo final `bb_slx` estão na seção 5.

- **Estimador.** Covariância HAC espacial de Conley (1999): núcleo de Bartlett sobre a distância de círculo máximo entre centroides, aplicada ao vetor completo de 25 parâmetros (20 de β e 5 de γ da dispersão). Os pontos estimados não mudam.
- **Por que Conley é a correção principal e não o agrupamento.** Os dados não têm códigos de microrregião; agrupar por UF (27 aglomerados) é um limite conservador, não o desenho natural.
- **Verificação da implementação.** Com corte ≈ 0 a covariância iguala a HC0; a covariância por aglomerado iguala a do statsmodels (diferença de 1e-15).

### Corte, fixado antes de ver os EPs (célula 17)

Regra: o corte é o limite inferior da primeira faixa do correlograma cuja correlação dos resíduos quantílicos é < 0,05. Faixas em km: 0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000.

| faixa (km) | 0-10 | 10-25 | 25-50 | 50-75 | 75-100 | 100-150 | 150-200 | 200-300 | 300-400 |
|---|---|---|---|---|---|---|---|---|---|
| correlação | 0,380 | 0,348 | 0,250 | 0,184 | 0,143 | 0,107 | 0,077 | 0,043 | 0,009 |

A primeira faixa abaixo de 0,05 é 200-300 km: **corte = 200 km.** Sensibilidade: 100 km (×0,5) e 400 km (×2). Limite conservador: agrupamento por UF com correção de amostra pequena.

### Resultado (células 20 e 21)

Razão entre o EP e o EP do modelo, sobre os 20 coeficientes de β:

| EP | mediana | mínimo | máximo |
|---|---|---|---|
| Conley, 100 km | 1,64 | 1,00 | 2,48 |
| **Conley, 200 km** | **2,06** | 0,95 (`bloco_outro`) | 3,67 (`reg_Sul`) |
| Conley, 400 km | 2,48 | 0,92 | 5,10 |
| Aglomerado por UF | 2,62 | 0,85 | 6,02 |

- **O EP do modelo é pequeno demais por um fator de cerca de 2** (variância cerca de 5 vezes maior sob Conley).
- **Significância a |z| > 1,96 perdida sob Conley a 200 km (6 de 25):** `pct_superior`, `pct_analfabeto`, `pct_evangelicos`, `reg_Nordeste`, `reg_Norte` e `disp_reg_Sul`. Nenhuma foi ganha.
- **Sensibilidade ao corte.** Com 100 km perdem-se 3 (`pct_superior`, `pct_evangelicos`, `reg_Norte`); com 400 km, 8 (as 6 acima mais `bloco_esquerda` e `reg_Centro-Oeste`); por UF, 10 (esses 8 mais `pct_16_24` e `pct_pretos_pardos`). A ordem é monótona no corte, e a leitura qualitativa (efeitos fracos ou regionais frágeis, efeitos grandes firmes) não muda.
- **Uso.** Os EPs de Conley a 200 km entram nas simulações da [Parte 17](11-modelo-final.md), onde a covariância de β alarga os intervalos.

## 4. Limites

- O I de Moran dos resíduos permanece em 0,333 no modelo final (`bb_slx`): Conley corrige a inferência, não as estimativas pontuais.
- O corte foi conferido contra 2022 (seção 6); a evidência vem de duas eleições, não de uma.
- O p-valor do LRT do [doc 09](09-dispersao-regional.md) supõe municípios independentes, o que a correlação residual contradiz. A diferença de BIC (450), o ganho de DP e o teste de Wald robusto ([doc 09](09-dispersao-regional.md), seção 5) não dependem desse p-valor.

## 5. Covariáveis defasadas espacialmente, SLX (adendo da Parte 15, células 10, 11, 13 e 14)

**Regra acrescentada em 2026-11-09, depois dos primeiros resultados do notebook**, e fixada antes de ver o resultado do SLX. Razão: o ESF falhou no LOSO porque carrega para a UF deixada de fora um padrão espacial ajustado em y. O SLX usa W·x, as covariáveis dos vizinhos, que são conhecidas numa UF nova; não carrega padrão de y.

- **Construção.** W = rainha com linhas padronizadas (ilhas ligadas ao vizinho mais próximo). Opções, escolhidas por BIC: nenhuma, W·lag, ou W·(as 12 variáveis do modelo-base). W·pib estaria na escala log.
- **Aceitação.** A mesma regra da seção 1: o I de Moran dos resíduos cai **e** o MAE por UF no LOSO é no máximo 1,1308 p.p. Aceito, o modelo final passa a `bb_disp` + SLX (`bb_slx`).

| opção | BIC | I de Moran | parâmetros |
|---|---|---|---|
| nenhuma | 72344,9 | 0,3390 | 25 |
| W·lag | 72340,2 | 0,3370 | 26 |
| W·12 variáveis | **72147,1** | 0,3328 | 37 |

- **Resultado.** O BIC escolhe W·12 variáveis. LOSO: MAE por UF de `bb_slx` = 1,0224 ≤ 1,1308, **aceito** (`aceitas.slx = true`, `modelo_final = "bb_slx"`). `bb_disp` tinha 0,9951; `bb_slx` fica um pouco pior, dentro da tolerância.
- **Leitura honesta.** A regra não fixou queda mínima do I de Moran. A queda é de 0,006 (0,339 para 0,333): o SLX quase não muda a estrutura espacial dos resíduos, e a aceitação vem do critério formal (I cai e LOSO dentro do limite), não de um ganho material. O BIC melhora em 198, mas esse BIC supõe independência. A dependência residual segue tratada por Conley; ver [pendências](07-pendencias.md).
- **EPs do modelo final** (Conley a 200 km, 37 parâmetros: 31 de β e 6 de γ). Razão EP de Conley / EP do modelo: mediana 1,78, mínimo 0,95, máximo 3,52 (`reg_Sul`); nos γ da dispersão, 1,74 a 2,54. Cinco parâmetros mudam de significância sob Conley: `w_pct_vab_agro`, `bloco_esquerda`, `reg_Centro-Oeste`, `reg_Norte` e `disp_reg_Sul`.

## 6. Estabilidade do corte de Conley em 2022 (adendo da Parte 16, células 18 e 19)

**Regra acrescentada em 2026-11-09, depois dos primeiros resultados do notebook**, fixada antes do resultado. Ajusta-se a especificação final em 2022 (lag = 2018) e aplica-se a mesma regra do correlograma. Se 0,5·CORTE ≤ CORTE_2022 ≤ 2·CORTE, mantém-se o CORTE de 2026; senão usa-se o maior.

- **Resultado.** A correlação de 2022 cai abaixo de 0,05 na faixa 200-300 km (0,030; a faixa anterior, 150-200 km, tem 0,084). CORTE_2022 = 200 km = CORTE_2026, então o corte de **200 km é mantido** (`corte_final` em `final_decisao.json`). Fecha o item de pendências sobre o corte.
- **Sensibilidade do EP ao corte** (`final_ep_corte.csv`, 50 a 800 km, 37 parâmetros). A razão mediana Conley/modelo sobe de 1,23 (50 km) para 1,78 (200 km) e satura perto de 2,0 a partir de 400 km; o máximo vai de 1,64 a 4,86 (400 km) e se mantém perto de 5 acima. A escolha de 200 km fica no meio da faixa de crescimento, não no extremo.

## Referências

- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*. Springer.
- Tiefelsdorf, M.; Griffith, D. A. (2007). Semiparametric filtering of spatial autocorrelation: the eigenvector approach. *Environment and Planning A*, 39(5), 1193-1221.
- Cameron, A. C.; Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
