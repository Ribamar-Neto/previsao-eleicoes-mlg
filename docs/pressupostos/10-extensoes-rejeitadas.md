# 10 - Extensões rejeitadas: krigagem e CAR nos resíduos (Parte 17 do notebook 05)

Notebook `notebooks/05-modelo-final.ipynb`. Tabelas em `results/tables/final_moran_decomposto.csv`, `final_krigagem.csv` e `final_car.csv`; previsões em `results/cv/{loso,kfold}_bb_slx_krig.parquet` e `results/cv/{loso,kfold}_bb_slx_car.parquet`. A dependência espacial e o SLX estão em [08-dependencia-espacial.md](08-dependencia-espacial.md); as estimativas do modelo final em [09-modelo-final.md](09-modelo-final.md).

Os dois remédios para a correlação residual do modelo final foram testados sob a mesma regra e **rejeitados**. O modelo final segue `bb_slx`.

## 1. Regra de aceitação comum, fixada em 2026-11-09 antes do resultado

Aceitar a extensão (`bb_slx_krig` ou `bb_slx_car`, isto é, `bb_slx` mais um efeito espacial nos resíduos) se e só se:

1. o I de Moran dos resíduos logit fora da amostra (LOSO, agrupados) cai pelo menos 0,05 em relação ao de `bb_slx` (0,4224);
2. o MAE por UF no LOSO é no máximo 1,2217 p.p. (limite L de [08](08-dependencia-espacial.md), seção 1).

Razão da queda mínima: a regra do SLX não tinha uma, e um remédio que quase não muda a estrutura espacial foi aceito ([08](08-dependencia-espacial.md), seção 3). A regra é a mesma para os dois remédios, para que sejam comparáveis. O CAR era a alternativa que restava na pendência de autocorrelação residual ([12](12-pendencias.md)).

## 2. Decomposição do I de Moran (diagnóstico)

Resíduos de `bb_slx`, dentro da amostra:

| matriz de vizinhança | I de Moran |
|---|---|
| W completa | 0,3361 |
| só vínculos dentro da mesma UF | 0,3439 |
| só vínculos entre UFs | 0,1570 |
| W completa, depois de tirar a média da UF | 0,2683 |

## 3. Krigagem dos resíduos

- **Procedimento, por dobra do LOSO.** Ajusta-se `bb_slx` em 26 UFs. O resíduo logit é r = logit((y + 0,5)/(n + 1)) − η̂. A previsão do resíduo no município da UF deixada de fora é a média de Bartlett dos resíduos de treino a até 200 km (o corte de Conley, sem novo ajuste de parâmetro), multiplicada por um encolhimento ρ̂ entre 0 e 1, estimado no treino só com vizinhos de outra UF.
- **Por que vale sob LOSO.** Usa só resíduos observados de outras UFs, que existem quando se prevê uma UF nova. Não há vazamento da UF deixada de fora.

| medida | `bb_slx_krig` | `bb_slx` |
|---|---|---|
| MAE por UF, LOSO (p.p.) | 1,0771 | 1,0777 |
| MAE por UF, 5 dobras (p.p.) | 0,8435 | 0,8459 |
| I de Moran dos resíduos LOSO | 0,4222 | 0,4224 |

- Queda do I: 0,0003 (de 0,4224 para 0,4222), contra o mínimo de 0,05.
- ρ̂ no LOSO: média 0,005 (0 a 0,041). Nas 5 dobras: média 0,004, máximo 0,019.
- **Decisão: rejeitada.** O critério 1 falha; o 2 passa (1,0771 ≤ 1,2217). `aceitas.krig = false` em `final_decisao.json`.
- [deduced] Resíduos do outro lado da fronteira de uma UF quase não preveem a UF deixada de fora (ρ̂ ≈ 0), por isso a krigagem quase não corrige.
- [deduced] Os vínculos dentro da UF carregam mais correlação (0,34) que os entre UFs (0,16). Tirar a média da UF reduz o I em cerca de um quinto (0,336 para 0,268). Parte da correlação é, então, um choque no nível da UF, já coberto por σ_UF nos intervalos da Parte 16 ([09](09-modelo-final.md)). O resto é agrupamento local dentro das UFs, que as covariáveis e as UFs vizinhas não explicam.
- [deduced] O I dos resíduos LOSO (0,42) fica acima do de dentro da amostra (0,34) porque todos os municípios da UF deixada de fora compartilham o mesmo erro de extrapolação.

## 4. Efeito espacial CAR/BYM nos resíduos

- **Modelo.** Em dois estágios: `bb_slx` é ajustado em cada dobra e o resíduo logit é r (o mesmo da krigagem). Depois r = u + ε, com u CAR de Leroux (precisão (1/τ)[(1 − λ)I + λ(D − A)], A a adjacência rainha binária e simétrica) e ε ruído iid N(0, σ²). É a estrutura BYM: parte espacial mais parte sem estrutura, com λ entre elas.
- **Estimação.** (τ, λ, σ²) por máxima verossimilhança gaussiana só no subgrafo de treino, com λ limitado a 0,99.
- **Previsão.** Média a posteriori de u no grafo completo dado só r de treino. π̃ = expit(η̂ + û). Municípios sem vizinho recebem û = 0.
- **Por que vale sob LOSO.** Nem o ajuste nem a previsão usam resíduo da UF deixada de fora; a UF nova só recebe informação dos vizinhos de treino pela divisa. O 5-fold repete o procedimento e é só descritivo.

| medida | `bb_slx_car` | `bb_slx` |
|---|---|---|
| MAE por UF, LOSO (p.p.) | 1,0037 | 1,0777 |
| MAE por UF, 5 dobras (p.p.) | 0,4736 | 0,8459 |
| I de Moran dos resíduos LOSO | 0,3949 | 0,4224 |
| I de Moran dos resíduos, 5 dobras | −0,0429 | 0,3207 |
| MAE regional LOSO (p.p.) | 0,5634 | 0,4260 |
| erro nacional LOSO (p.p.) | 0,231 | 0,167 |

- Queda do I no LOSO: 0,0275, contra o mínimo de 0,05.

Parâmetros estimados no treino (por dobra):

| parâmetro | LOSO: média (mín a máx) | 5 dobras: média (mín a máx) |
|---|---|---|
| τ | 0,0153 (0,0136 a 0,0176) | 0,0096 (0,0079 a 0,0128) |
| λ | 0,798 (0,495 a 0,935) | 0,811 (0,495 a 0,914) |
| σ² | 0,0023 | 0,0031 |

- **Decisão: rejeitada.** O critério 1 falha (0,0275 < 0,05); o 2 passa (1,0037 ≤ 1,2217). `aceitas.car = false`.
- [fact] O CAR reduz o MAE por UF do LOSO em 0,07 p.p., mas piora o MAE regional (0,426 para 0,563) e o erro nacional (0,17 para 0,23). O ganho por UF não se soma nas regiões.
- [fact] O I de Moran LOSO cai só 0,027. Os resíduos de uma UF inteira deixada de fora continuam correlacionados entre si.
- [deduced] Diferente da krigagem (ρ̂ ≈ 0), o CAR acha estrutura forte nos resíduos de treino (λ médio 0,80), porque aprende a correlação entre vizinhos da mesma UF. Essa correlação não atravessa a divisa, então só os municípios de fronteira da UF nova são corrigidos.
- [deduced] No 5-fold o CAR tem muito ganho (MAE por UF 0,846 para 0,474; I de −0,04) porque cada município de teste tem vizinhos de treino. Esse esquema não imita a previsão de uma UF nova e não entra na decisão.
- [deduced] Um λ mínimo de 0,495 em algumas dobras está perto do ponto de partida do otimizador (λ ≈ 0,5); a verossimilhança é plana em λ nessas dobras ou a busca parou cedo. Não foi investigado, e a decisão (queda de 0,027 contra 0,05) não é sensível a um ajuste fino de λ de forma conhecida [assumed].

## 5. Consequência

A correlação residual não é removível com a informação disponível para uma UF nova. Fica como limitação documentada, tratada por Conley ([08](08-dependencia-espacial.md), seção 4) na inferência e por σ_UF nos intervalos ([09](09-modelo-final.md)). Não resta alternativa espacial na lista.

## Referências

- Besag, J.; York, J.; Mollié, A. (1991). Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-20. https://doi.org/10.1007/BF00116466
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Leroux, B. G.; Lei, X.; Breslow, N. (2000). Estimation of disease rates in small areas: a new mixed model for spatial dependence. In *Statistical Models in Epidemiology, the Environment, and Clinical Trials*. Springer, 179-191. https://doi.org/10.1007/978-1-4612-1284-3_4
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
