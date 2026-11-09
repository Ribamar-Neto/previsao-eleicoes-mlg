# Dependência espacial: filtro de autovetores e EPs de Conley (Partes 15 e 16 do notebook 05)

Células citadas contam a partir de 0 em `notebooks/05-modelo-final.ipynb` (5 a 14). Tabelas em `results/tables/final_esf_k.csv`, `final_cv.csv`, `final_decisao.json`, `final_correlograma.csv` e `final_ep.csv`; figuras `results/figures/final_moran_k.html` e `final_correlograma.html`. A dispersão (Parte 14) está em [09-dispersao-regional.md](09-dispersao-regional.md); as estimativas do modelo final (Parte 17) em [11-modelo-final.md](11-modelo-final.md).

**Objetivo.** Os resíduos de todos os modelos têm autocorrelação espacial (I de Moran ≈ 0,34; ver [diagnóstico, seção 6](04-diagnostico.md) e [pendências](07-pendencias.md)), o que torna os EPs pequenos demais. Duas respostas: (a) tirar a dependência do preditor com um filtro espacial (Parte 15); (b) manter o preditor e corrigir só os EPs (Parte 16).

## 1. Regra comum de aceitação, escrita antes do resultado (célula 8)

Fixada em 2026-11-09, antes de ver os resultados das extensões. Toda extensão mantém os 12 termos de β de `bb_backward` e é aceita só se (1) o seu diagnóstico melhora **e** (2) o MAE por UF no LOSO é no máximo o de `bb_backward` mais 1 EP: 1,0004 + 0,1304 = **1,1308 p.p.** (mesmo limite do [doc 09](09-dispersao-regional.md), seção 4). Razão: o ganho de diagnóstico não pode custar previsão fora da amostra, que é o objetivo do projeto.

## 2. Filtro espacial de autovetores de Moran, ESF (células 5 a 7)

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
- **Decisão: ESF rejeitado.** O modelo final é `bb_disp` (`final_decisao.json`, `aceitas.esf = false`, `modelo_final = "bb_disp"`).

## 3. EPs espaciais de Conley (células 11 a 14)

Como o ESF não foi aceito, a dependência residual (I de Moran 0,339) permanece e é tratada só na inferência.

- **Estimador.** Covariância HAC espacial de Conley (1999): núcleo de Bartlett sobre a distância de círculo máximo entre centroides, aplicada ao vetor completo de 25 parâmetros (20 de β e 5 de γ da dispersão). Os pontos estimados não mudam.
- **Por que Conley é a correção principal e não o agrupamento.** Os dados não têm códigos de microrregião; agrupar por UF (27 aglomerados) é um limite conservador, não o desenho natural.
- **Verificação da implementação.** Com corte ≈ 0 a covariância iguala a HC0; a covariância por aglomerado iguala a do statsmodels (diferença de 1e-15).

### Corte, fixado antes de ver os EPs (célula 12)

Regra: o corte é o limite inferior da primeira faixa do correlograma cuja correlação dos resíduos quantílicos é < 0,05. Faixas em km: 0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000.

| faixa (km) | 0-10 | 10-25 | 25-50 | 50-75 | 75-100 | 100-150 | 150-200 | 200-300 | 300-400 |
|---|---|---|---|---|---|---|---|---|---|
| correlação | 0,380 | 0,348 | 0,250 | 0,184 | 0,143 | 0,107 | 0,077 | 0,043 | 0,009 |

A primeira faixa abaixo de 0,05 é 200-300 km: **corte = 200 km.** Sensibilidade: 100 km (×0,5) e 400 km (×2). Limite conservador: agrupamento por UF com correção de amostra pequena.

### Resultado (célula 13)

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

- O I de Moran dos resíduos permanece em 0,339 no modelo final: Conley corrige a inferência, não as estimativas pontuais.
- O corte depende do correlograma de uma única eleição.
- O p-valor do LRT do [doc 09](09-dispersao-regional.md) supõe municípios independentes, o que a correlação residual contradiz. A diferença de BIC (450) e o ganho de DP não dependem desse p-valor.

## Referências

- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*. Springer.
- Tiefelsdorf, M.; Griffith, D. A. (2007). Semiparametric filtering of spatial autocorrelation: the eigenvector approach. *Environment and Planning A*, 39(5), 1193-1221.
- Cameron, A. C.; Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
