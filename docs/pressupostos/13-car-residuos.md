# Efeito espacial CAR/BYM nos resíduos (Parte 19 do notebook 05)

Células citadas contam a partir de 0 em `notebooks/05-modelo-final.ipynb` (38 a 41). Tabela em `results/tables/final_car.csv`; previsões em `results/cv/{loso,kfold}_bb_slx_car.parquet`.

A krigagem dos resíduos e a decomposição do I de Moran estão em [12-krigagem-residuos.md](12-krigagem-residuos.md); o modelo final em [11-modelo-final.md](11-modelo-final.md).

## 1. Regra de aceitação, fixada em 2026-11-09 antes do resultado

Aceitar `bb_slx_car` (`bb_slx` mais efeito CAR/BYM nos resíduos) se e só se:

1. o I de Moran dos resíduos logit fora da amostra (LOSO, agrupados) cai pelo menos 0,05 em relação ao de `bb_slx`;
2. o MAE por UF no LOSO é no máximo 1,1308 p.p.

Razão: é a mesma regra da krigagem ([12](12-krigagem-residuos.md), seção 1), para que os dois remédios sejam comparáveis. O CAR é a alternativa que restava na pendência de autocorrelação residual ([07](07-pendencias.md)).

## 2. Procedimento (células 39 a 41)

- **Modelo.** Em dois estágios: `bb_slx` é ajustado em cada dobra e o resíduo logit é r = logit((y + 0,5)/(n + 1)) − η̂ (o mesmo da krigagem, [12](12-krigagem-residuos.md)). Depois r = u + ε, com u CAR de Leroux (precisão (1/τ)[(1 − λ)I + λ(D − A)], A a adjacência queen binária e simétrica) e ε ruído iid N(0, σ²). É a estrutura BYM: parte espacial mais parte sem estrutura, com λ entre elas.
- **Estimação.** (τ, λ, σ²) por máxima verossimilhança gaussiana só no subgrafo de treino, com λ limitado a 0,99. Não há dependência nova; as matrizes são esparsas.
- **Previsão.** Média a posteriori de u no grafo completo dado só r de treino. π̃ = expit(η̂ + û). Municípios sem vizinho recebem û = 0.
- **Por que vale sob LOSO.** Nem o ajuste nem a previsão usam resíduo da UF deixada de fora; a UF nova só recebe informação dos vizinhos de treino pela divisa. O 5-fold repete o procedimento e é só descritivo.
- **Critério.** A seção 1, aplicada mecanicamente (`aceitas.car` em `final_decisao.json`).

## 3. Resultado

| medida | `bb_slx_car` | `bb_slx` |
|---|---|---|
| MAE por UF, LOSO (p.p.) | 0,9430 | 1,0224 |
| MAE, 5 dobras (p.p.) | 0,4502 | 0,7546 |
| I de Moran dos resíduos LOSO | 0,3834 | 0,4201 |
| I de Moran dos resíduos, 5 dobras | −0,0496 | 0,3129 |
| MAE regional LOSO (p.p.) | 0,4818 | 0,3210 |
| erro nacional LOSO (p.p.) | 0,237 | 0,170 |

- Queda do I no LOSO: 0,0367, contra o mínimo de 0,05.

Parâmetros estimados no treino (por dobra):

| parâmetro | LOSO: média (mín a máx) | 5 dobras: média (mín a máx) |
|---|---|---|
| τ | 0,0147 (0,0135 a 0,0173) | 0,0103 (0,0078 a 0,0132) |
| λ | 0,870 (0,495 a 0,929) | 0,737 (0,495 a 0,908) |
| σ² | 0,00266 (0,00080 a 0,00313) | 0,00271 (0,00120 a 0,00403) |

## 4. Decisão

**Rejeitada.** O critério 1 falha (0,0367 < 0,05); o 2 passa (0,9430 ≤ 1,1308). `modelo_final` continua `bb_slx` (`final_decisao.json`, `aceitas.car = false`).

## 5. Leitura

- [fact] O CAR reduz o MAE por UF do LOSO em 0,08 p.p., mas piora o MAE regional (0,321 para 0,482) e o erro nacional (0,17 para 0,24). O ganho por UF não se soma nas regiões.
- [fact] O I de Moran LOSO cai só 0,037. Os resíduos de uma UF inteira deixada de fora continuam correlacionados entre si.
- [deduced] Diferente da krigagem (ρ̂ ≈ 0,02), o CAR acha estrutura forte nos resíduos de treino (λ médio 0,87; σ² pequeno diante de τ), porque aprende a correlação entre vizinhos da mesma UF. Essa correlação não atravessa a divisa, então só os municípios de fronteira da UF nova são corrigidos.
- [deduced] No 5-fold o CAR tem muito ganho (MAE por UF 0,755 para 0,450; I de −0,05) porque cada município de teste tem vizinhos de treino. Esse esquema não imita a previsão de uma UF nova e não entra na decisão.
- [deduced] Um λ mínimo de 0,495 em algumas dobras está perto do ponto de partida do otimizador (λ ≈ 0,5); a verossimilhança é plana em λ nessas dobras ou a busca parou cedo. Não foi investigado, e a decisão (queda de 0,037 contra 0,05) não é sensível a um ajuste fino de λ de forma conhecida. [assumed]
- **Consequência.** Com a última alternativa espacial rejeitada, a autocorrelação residual fica como limitação documentada, tratada por Conley ([10](10-dependencia-espacial.md), seção 3) e por σ_UF nos intervalos ([11](11-modelo-final.md)).

## Referências

- Besag, J.; York, J.; Mollié, A. (1991). Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-20. https://doi.org/10.1007/BF00116466
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Leroux, B. G.; Lei, X.; Breslow, N. (2000). Estimation of disease rates in small areas: a new mixed model for spatial dependence. In *Statistical Models in Epidemiology, the Environment, and Clinical Trials*. Springer, 179-191. https://doi.org/10.1007/978-1-4612-1284-3_4
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
