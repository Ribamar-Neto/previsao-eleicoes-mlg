# 10 - Extensões rejeitadas: krigagem e CAR nos resíduos (Parte 17 do `02-MLG`)

Notebook `notebooks/02-MLG.ipynb`. Tabelas em `results/tables/final_moran_decomposto.csv`, `final_krigagem.csv` e `final_car.csv`; previsões em `results/cv/{loso,kfold}_bb_slx_krig.parquet` e `results/cv/{loso,kfold}_bb_slx_car.parquet`. A dependência espacial e o SLX estão em [08-dependencia-espacial.md](08-dependencia-espacial.md); as estimativas do modelo final em [09-modelo-final.md](09-modelo-final.md).

Os dois remédios para a correlação residual do modelo final foram testados sob a mesma regra e **rejeitados**. O modelo final segue `bb_slx`.

## 1. Regra de aceitação comum, fixada em 2026-11-09 antes do resultado

Aceitar a extensão (`bb_slx_krig` ou `bb_slx_car`, isto é, `bb_slx` mais um efeito espacial nos resíduos) se e só se:

1. o I de Moran dos resíduos logit fora da amostra (LOSO, agrupados) cai pelo menos 0,05 em relação ao de `bb_slx` (0,2969);
2. o MAE por UF no LOSO é no máximo 0,2904 p.p. (limite L de [08](08-dependencia-espacial.md), seção 1).

Razão da queda mínima: a regra do SLX não tinha uma, e um remédio que quase não muda a estrutura espacial foi aceito ([08](08-dependencia-espacial.md), seção 3). A regra é a mesma para os dois remédios, para que sejam comparáveis. O CAR era a alternativa que restava na pendência de autocorrelação residual ([12](12-pendencias.md)).

## 2. Decomposição do I de Moran (diagnóstico)

Resíduos de `bb_slx`, dentro da amostra:

| matriz de vizinhança | I de Moran |
|---|---|
| W completa | 0,2697 |
| só vínculos dentro da mesma UF | 0,2809 |
| só vínculos entre UFs | 0,0818 |
| W completa, depois de tirar a média da UF | 0,2024 |

## 3. Krigagem dos resíduos

- **Procedimento, por dobra do LOSO.** Ajusta-se `bb_slx` em 26 UFs. O resíduo logit é r = logit((y + 0,5)/(n + 1)) − η̂. A previsão do resíduo no município da UF deixada de fora é a média de Bartlett dos resíduos de treino a até 150 km (o corte de Conley, sem novo ajuste de parâmetro), multiplicada por um encolhimento ρ̂ entre 0 e 1, estimado no treino só com vizinhos de outra UF.
- **Por que vale sob LOSO.** Usa só resíduos observados de outras UFs, que existem quando se prevê uma UF nova. Não há vazamento da UF deixada de fora.

| medida | `bb_slx_krig` | `bb_slx` |
|---|---|---|
| MAE por UF, LOSO (p.p.) | 0,1970 | 0,1940 |
| MAE por UF, 5 dobras (p.p.) | 0,1441 | 0,1397 |
| I de Moran dos resíduos LOSO | 0,2844 | 0,2969 |

- Queda do I: 0,0126 (de 0,2969 para 0,2844), contra o mínimo de 0,05.
- ρ̂ no LOSO: média 0,218 (0,074 a 0,305). Nas 5 dobras: média 0,211 (0,178 a 0,237).
- **Decisão: rejeitada.** O critério 1 falha; o 2 passa (0,1970 ≤ 0,2904). `aceitas.krig = false` em `final_decisao.json`.
- [deduced] Resíduos do outro lado da fronteira de uma UF preveem parte do erro da UF deixada de fora (ρ̂ ≈ 0,22), mas só os municípios de divisa têm vizinhos de treino a menos de 150 km com peso relevante, por isso a krigagem corrige pouco o I.
- [deduced] Os vínculos dentro da UF carregam mais correlação (0,28) que os entre UFs (0,08). Tirar a média da UF reduz o I em cerca de um quarto (0,270 para 0,202). Parte da correlação é, então, um choque no nível da UF, já coberto por σ_UF nos intervalos da Parte 16 ([09](09-modelo-final.md)). O resto é agrupamento local dentro das UFs, que as covariáveis e as UFs vizinhas não explicam.
- [deduced] O I dos resíduos LOSO (0,30) fica acima do de dentro da amostra (0,27) porque todos os municípios da UF deixada de fora compartilham o mesmo erro de extrapolação.

- **Limitação.** O corte do correlograma vem dos resíduos dentro da amostra com todos os
  dados (Parte 15, c. 20) e é reusado como alcance da krigagem (c. 33): vazamento leve
  que favorece aceitar a krigagem. Ela foi rejeitada, então a decisão não muda
  ([12](12-pendencias.md)).

## 4. Efeito espacial CAR/BYM nos resíduos

- **Modelo.** Em dois estágios: `bb_slx` é ajustado em cada dobra e o resíduo logit é r (o mesmo da krigagem). Depois r = u + ε, com u CAR de Leroux (precisão (1/τ)[(1 − λ)I + λ(D − A)], A a adjacência rainha binária e simétrica) e ε ruído iid N(0, σ²). É a estrutura BYM: parte espacial mais parte sem estrutura, com λ entre elas.
- **Estimação.** (τ, λ, σ²) por máxima verossimilhança gaussiana só no subgrafo de treino, com λ limitado a 0,99.
- **Previsão.** Média a posteriori de u no grafo completo dado só r de treino. π̃ = expit(η̂ + û). Municípios sem vizinho recebem û = 0.
- **Por que vale sob LOSO.** Nem o ajuste nem a previsão usam resíduo da UF deixada de fora; a UF nova só recebe informação dos vizinhos de treino pela divisa. O 5-fold repete o procedimento e é só descritivo.

| medida | `bb_slx_car` | `bb_slx` |
|---|---|---|
| MAE por UF, LOSO (p.p.) | 0,2166 | 0,1940 |
| MAE por UF, 5 dobras (p.p.) | 0,1777 | 0,1397 |
| I de Moran dos resíduos LOSO | 0,2609 | 0,2969 |
| I de Moran dos resíduos, 5 dobras | −0,0541 | 0,2370 |
| MAE regional LOSO (p.p.) | 0,1636 | 0,1451 |
| erro nacional LOSO (p.p.) | 0,192 | 0,181 |

- Queda do I no LOSO: 0,0360, contra o mínimo de 0,05.

Parâmetros estimados no treino (por dobra):

| parâmetro | LOSO: média (mín a máx) | 5 dobras: média (mín a máx) |
|---|---|---|
| τ | 0,0018 (0,0016 a 0,0026) | 0,0015 (0,0010 a 0,0018) |
| λ | 0,865 (0,495 a 0,960) | 0,664 (0,495 a 0,928) |
| σ² | 0,00065 (0,00030 a 0,00078) | 0,00058 (0,00040 a 0,00081) |

- **Decisão: rejeitada.** O critério 1 falha (0,0360 < 0,05); o 2 passa (0,2166 ≤ 0,2904). `aceitas.car = false`.
- [fact] O CAR não ajuda a previsão: o MAE por UF do LOSO sobe de 0,1940 para 0,2166, o MAE regional de 0,145 para 0,164 e o erro nacional de 0,181 para 0,192.
- [fact] O I de Moran LOSO cai 0,036, abaixo do mínimo de 0,05. Os resíduos de uma UF inteira deixada de fora continuam correlacionados entre si.
- [deduced] O CAR acha estrutura nos resíduos de treino (λ médio 0,87 no LOSO), porque aprende a correlação entre vizinhos da mesma UF. Essa correlação não atravessa a divisa, então só os municípios de fronteira da UF nova são corrigidos.
- [deduced] No 5-fold o CAR zera o I (−0,05), porque cada município de teste tem vizinhos de treino, mas o MAE por UF piora (0,140 para 0,178): remover a autocorrelação não reduz o erro. Esse esquema não imita a previsão de uma UF nova e não entra na decisão.
- [deduced] Um λ mínimo de 0,495 (em alguma dobra do LOSO e das 5 dobras) está perto do ponto de partida do otimizador (λ ≈ 0,5); a verossimilhança é plana em λ nessas dobras ou a busca parou cedo. Não foi investigado. Com a queda de 0,036 e o mínimo de 0,05, não se sabe se um λ afinado cruzaria o mínimo [assumed].

- **Limitação (krigagem e CAR).** ρ e o ajuste do CAR usam resíduos sem peso, embora a
  variância deles seja cerca de 1/n; o I de Moran fora da amostra também é sem peso. A
  sensibilidade ponderada está na seção 6: as decisões ficam.

## 5. Consequência

A correlação residual não é removível com a informação disponível para uma UF nova. Fica como limitação documentada, tratada por Conley ([08](08-dependencia-espacial.md), seção 4) na inferência e por σ_UF nos intervalos ([09](09-modelo-final.md)). Não resta alternativa espacial na lista.

## 6. Sensibilidade ponderada (informativa, 2026-11-22)

**Motivo.** A variância do resíduo da proporção é cerca de 1/n, então municípios pequenos pesam
como os grandes em ρ, no CAR e no I de Moran.

**Especificação (fixada antes do resultado).**

- Resíduo de Pearson beta-binomial z = (y/n − π̂)/sd, sd² = π̂(1 − π̂)[1 + (n − 1)θ]/n, com θ por
  município. Mesmo π̂ e mesmo n dos resíduos atuais (dentro da amostra no treino; fora da amostra
  no I de Moran). Os resíduos atuais são logit; aqui se padroniza a versão na escala da proporção.
- ρ, krigagem LOSO e CAR rodam sobre z, com o mesmo corte, W, dobras e sementes. A correção volta à
  proporção como ẑ·sd, somada a π̂.
- Mesmos critérios de antes (queda do I >= 0,05; MAE por UF LOSO <= limite 0,2904 p.p.).

**Resultado** (`results/tables/sensibilidade_ponderada_krig_car.csv`, LOSO).

| Métrica | Krigagem atual | Krigagem ponderada | CAR atual | CAR ponderado |
|---|---|---|---|---|
| ρ médio / λ médio | 0,2180 | 0,2207 | 0,8650 | 0,7574 |
| MAE por UF (p.p.) | 0,197 | 0,247 | 0,217 | 0,216 |
| MAE regional (p.p.) | 0,141 | 0,166 | 0,164 | 0,171 |
| I de Moran fora da amostra, base | 0,2969 | 0,2977 | 0,2969 | 0,2977 |
| Queda do I (exige 0,05) | 0,0126 | 0,0070 | 0,0360 | 0,0379 |

[fact] Nenhuma decisão muda: o critério do MAE passa e o do I de Moran falha, nas duas versões, para
krigagem e CAR. [fact] Padronizar quase não muda o I do modelo base (0,2969 para 0,2977): com a
covariável do 1º turno, o peso dos municípios pequenos deixou de importar para a correlação espacial.
[deduced] O MAE por UF piora com a correção ponderada na krigagem (0,197 para 0,247) e fica estável no CAR (0,217 para 0,216); o MAE regional piora nos dois (krigagem 0,141 para 0,166; CAR 0,164 para 0,171).

## Referências

- Besag, J.; York, J.; Mollié, A. (1991). Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-20. https://doi.org/10.1007/BF00116466
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Leroux, B. G.; Lei, X.; Breslow, N. (2000). Estimation of disease rates in small areas: a new mixed model for spatial dependence. In *Statistical Models in Epidemiology, the Environment, and Clinical Trials*. Springer, 179-191. https://doi.org/10.1007/978-1-4612-1284-3_4
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
