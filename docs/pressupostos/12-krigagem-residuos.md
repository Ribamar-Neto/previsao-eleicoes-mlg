# Correlação residual: decomposição e krigagem dos resíduos (Parte 18 do notebook 05)

Células citadas contam a partir de 0 em `notebooks/05-modelo-final.ipynb` (33 a 38). Tabelas em `results/tables/final_moran_decomposto.csv` e `final_krigagem.csv`. A dependência espacial e o SLX estão em [10-dependencia-espacial.md](10-dependencia-espacial.md); as estimativas do modelo final em [11-modelo-final.md](11-modelo-final.md).

## 1. Regra de aceitação, fixada em 2026-11-09 antes do resultado

Aceitar `bb_slx_krig` (`bb_slx` mais krigagem dos resíduos) se e só se:

1. o I de Moran dos resíduos logit fora da amostra (LOSO, agrupados) cai pelo menos 0,05 em relação ao de `bb_slx`;
2. o MAE por UF no LOSO é no máximo 1,1308 p.p.

Razão da queda mínima: a regra do SLX não tinha uma, e um remédio que quase não muda a estrutura espacial foi aceito ([10](10-dependencia-espacial.md), seção 5).

## 2. Decomposição do I de Moran (diagnóstico, células 33 e 34)

Resíduos de `bb_slx`, dentro da amostra:

| matriz de vizinhança | I de Moran |
|---|---|
| W completa | 0,3328 |
| só vínculos dentro da mesma UF | 0,3406 |
| só vínculos entre UFs | 0,1507 |
| W completa, depois de tirar a média da UF | 0,2630 |

## 3. Krigagem dos resíduos (células 35 a 37)

- **Procedimento, por dobra do LOSO.** Ajusta-se `bb_slx` em 26 UFs. O resíduo logit é r = logit((y + 0,5)/(n + 1)) − η̂. A previsão do resíduo no município da UF deixada de fora é a média de Bartlett dos resíduos de treino a até 200 km (o corte de Conley, sem novo ajuste de parâmetro), multiplicada por um encolhimento ρ̂ entre 0 e 1.
- **ρ̂** é estimado no treino, só com vizinhos de outra UF, e limitado a [0, 1].
- **Por que vale sob LOSO.** Usa só resíduos observados de outras UFs, que existem quando se prevê uma UF nova. Não há vazamento da UF deixada de fora.

| medida | `bb_slx_krig` | `bb_slx` |
|---|---|---|
| MAE por UF, LOSO (p.p.) | 1,0205 | 1,0224 |
| MAE, 5 dobras (p.p.) | 0,7459 | 0,7546 |
| I de Moran dos resíduos LOSO | 0,4191 | 0,4201 |

- Queda do I: 0,0009, contra o mínimo de 0,05.
- ρ̂ no LOSO: média 0,017 (0 a 0,071). Nas 5 dobras: média 0,020, máximo 0,049.

## 4. Decisão

**Rejeitada.** O critério 1 falha (0,0009 < 0,05); o 2 passa (1,0205 ≤ 1,1308). `modelo_final` continua `bb_slx` (`final_decisao.json`, `aceitas.krig = false`).

## 5. Leitura

- [deduced] Resíduos do outro lado da fronteira de uma UF quase não preveem a UF deixada de fora (ρ̂ ≈ 0,02), por isso a krigagem quase não corrige.
- [deduced] Os vínculos dentro da UF carregam mais correlação (0,34) que os entre UFs (0,15). Tirar a média da UF reduz o I em cerca de um quinto (0,333 para 0,263). Parte da correlação é, então, um choque no nível da UF, já coberto por σ_UF nos intervalos da Parte 17 ([11](11-modelo-final.md)). O resto é agrupamento local dentro das UFs, que as covariáveis e as UFs vizinhas não explicam.
- [deduced] O I dos resíduos LOSO (0,42) fica acima do de dentro da amostra (0,33) porque todos os municípios da UF deixada de fora compartilham o mesmo erro de extrapolação.
- **Consequência.** A correlação não é removível com a informação disponível para uma UF nova. A inferência fica com Conley ([10](10-dependencia-espacial.md), seção 3) e os intervalos com σ_UF. O item de pendência fica fechado como limitação ([07](07-pendencias.md)).

## Referências

- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
