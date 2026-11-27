# 07 - Dispersão regional do beta-binomial (Parte 13 do `02-MLG`)

Notebook `notebooks/02-MLG.ipynb`. Tabelas em `results/tables/final_dispersao.csv`, `final_dispersao_wald.csv`, `final_dp_regiao.csv`, `final_cv.csv` e `final_decisao.json`; figura em `results/figures/final_dp_regiao.png`. O filtro espacial (Parte 14) fica em [08](08-dependencia-espacial.md).

**Objetivo.** Testar se deixar a dispersão do beta-binomial variar com região (e com o porte do município) melhora o diagnóstico sem piorar a previsão fora da amostra. O ponto de partida é `bb_selecao` (18 contínuas, entre elas a defasagem de 2022 e a do 1º turno do mesmo ano, `lag1t`; média com a fusão Nordeste + Sudeste, isto é, três dummies de região; escolhido na Parte 11 pela regra do 1-EP, [05](05-validacao-cruzada.md)). Os termos de média não mudam; só a dispersão muda.

**Motivação.** Com θ constante, o desvio padrão (DP) do resíduo quantílico por região ia de 0,75 (Sul) a 1,19 (Nordeste): a dispersão não é constante. Ver [diagnóstico, seção 3](04-diagnostico.md) e [pendências](12-pendencias.md).

## 1. Modelo de dispersão

Modelo duplo (Smyth, 1989): a média segue o preditor de `bb_selecao` e a dispersão tem preditor próprio, log θᵢ = γ·zᵢ. A dispersão usa as **5 regiões**. Opções de zᵢ:

| opção | zᵢ | parâmetros extras |
|---|---|---|
| `const` | constante (o `bb_selecao` original) | 0 |
| `regiao` | constante e dummies de região | 4 |
| `regiao_logn` | constante, dummies de região e z(log nᵢ) | 5 |

- **Por que região.** A heterogeneidade de dispersão observada é regional (seção 3 do diagnóstico).
- **Por que log n.** A dispersão pode depender do porte do município (n); z(log n) é a versão padronizada. Foi candidato, testado e **não adotado** (seção 5: p = 0,278).
- **Escolha entre opções:** o teste de Wald robusto da seção 5 decide; BIC e LRT, ajustados a todos os dados, são reportados.

## 2. Regra de aceitação, escrita antes do resultado

Fixada em 2026-11-09, antes de ver os resultados da extensão. A dispersão escolhida é aceita se:

1. o teste da razão de verossimilhanças (LRT) contra θ constante tem p < 0,01;
2. o maior |DP − 1| dos resíduos quantílicos por região é menor que com θ constante (resíduos aleatorizados, semente 0, como no [diagnóstico](04-diagnostico.md));
3. o MAE por UF no LOSO é no máximo o limite L da seção 4.

Falhando qualquer uma, a dispersão volta a constante. Razão do limite 3: o ganho de diagnóstico não pode custar previsão fora da amostra, que é o objetivo do projeto.

**Registro separado.** As extensões ficam em `EXTENSOES`, fora de `CANDIDATOS`. Razão: a Parte 11 monta sua lista de candidatos a partir de `CANDIDATOS`; incluí-las ali mudaria as tabelas das Partes 5 a 11.

## 3. Resultado do ajuste

| opção | log-verossimilhança | k | BIC | LRT vs `const` | gl | p |
|---|---|---|---|---|---|---|
| `const` | -31182,2 | 24 | 62571,4 | n/a | n/a | n/a |
| `regiao` | -31006,9 | 28 | 62255,2 | 350,7 | 4 | 1,2e-74 |
| `regiao_logn` | -31003,6 | 29 | 62257,3 | 357,3 | 5 | 4,8e-75 |

- **Escolhida: `regiao`.** O BIC já a prefere a `regiao_logn` (diferença de 2,1, pequena). O LRT de `regiao_logn` contra `regiao` (6,5, 1 gl, p ≈ 0,011) fica na margem do corte de 0,01; quem decide é o Wald robusto da seção 5, que descarta o log n (p = 0,278). O log n foi testado e não adotado: não carrega dispersão além da região.
- **Condição 1:** LRT p = 1,2e-74 (< 0,01). Ressalva: a verossimilhança assume independência entre municípios, que o [diagnóstico](04-diagnostico.md) não confirma; o p-valor é indicativo, e a decisão pesa mais o DP e a CV.

DP do resíduo quantílico por região:

| opção | Centro-Oeste | Nordeste | Norte | Sudeste | Sul | max \|DP − 1\| |
|---|---|---|---|---|---|---|
| `const` | 0,898 | 1,187 | 0,958 | 0,903 | 0,754 | 0,246 |
| `regiao` | 1,009 | 0,993 | 1,008 | 0,982 | 0,929 | 0,071 |
| `regiao_logn` | 1,003 | 0,990 | 1,002 | 0,974 | 0,926 | 0,074 |

- **Condição 2:** o maior |DP − 1| cai de 0,246 para 0,071. Cumprida.
- **Leitura.** `regiao` e `regiao_logn` ficam empatadas no DP (0,071 e 0,074); o log n não melhora o diagnóstico. O Sul continua o pior ajuste (DP 0,93).

## 4. Validação cruzada da extensão

Candidato `bb_disp` = `bb_selecao` com a dispersão `regiao`. Roda em 5-fold (semente 0) e LOSO (uma UF fora por vez), sem alterar as tabelas da Parte 11.

**Limite L.** MAE por UF aninhado de `bb_selecao` (0,2497 p.p., seleção refeita em cada dobra; [05](05-validacao-cruzada.md), seção 8) + erro-padrão do melhor modelo LOSO da Parte 11 (`bin_logit`, 0,0407) = **0,2904 p.p.** (0,2541 com o 0,2134 sem aninhar). É o EP da regra do 1-EP usada para escolher `bb_selecao`, para que a tolerância seja a mesma da escolha original. O MAE por UF e o MAE regional (`mae_uf_pp`, `mae_reg_pp`) ponderam as previsões por n̂, como nos notebooks 03 e 05; `mae_pond_pp` e `erro_nacional_pp` ponderam pelo n real.

| esquema | modelo | MAE por UF (p.p.) | EP | MAE ponderado (p.p.) | erro nacional (p.p.) |
|---|---|---|---|---|---|
| LOSO | `bb_selecao` (seleção nos dados completos) | 0,2134 | 0,035 | 0,537 | 0,147 |
| LOSO | `bb_disp` | 0,2176 | 0,034 | 0,525 | 0,154 |
| 5-fold | `bb_selecao` | 0,1804 | 0,028 | 0,509 | 0,149 |
| 5-fold | `bb_disp` | 0,1866 | 0,028 | 0,499 | 0,160 |

- **Condição 3:** 0,2176 ≤ 0,2904. Cumprida.
- **Decisão.** As três condições passam: `bb_disp` é aceita (`final_decisao.json`, `aceitas.disp = true`). A previsão não muda de forma relevante: o MAE por UF piora 0,004 p.p. no LOSO (EP 0,03) e o MAE ponderado melhora 0,012 p.p.; a deviance no LOSO cai de 26429 para 25372. A aceitação se apoia no diagnóstico (DP regional) sem custo de previsão, não em ganho de previsão.
- **O que a dispersão muda.** Os termos de média são os mesmos de `bb_selecao`; os coeficientes são reestimados juntos com a dispersão.

## 5. Teste de Wald robusto para a dispersão

**Regra acrescentada em 2026-11-09, depois dos primeiros resultados do notebook** (as seções 2 a 4 foram escritas antes de ver os resultados; esta não). Razão: o LRT da seção 3 supõe municípios independentes, o que a correlação residual contradiz ([pendências](12-pendencias.md)). A regra foi fixada antes de calcular o resultado do teste.

- **Teste.** Wald sequencial sobre o vetor γ da dispersão, com a covariância de Conley do próprio ajuste (corte do correlograma de resíduos quantílicos daquele ajuste: limite inferior da primeira faixa com correlação < 0,05, regra da Parte 15, [doc 08](08-dependencia-espacial.md)). Passo A: 4 dummies de região na dispersão, que segue com as 5 regiões mesmo com a fusão na média (`regiao` contra `const`). Passo B: z(log n) (`regiao_logn` contra `regiao`), testado só se A for mantido. O corte do correlograma nesse ajuste é 300 km (×0,5 = 150 e ×2 = 600 km).
- **Decisão.** Mantém-se o passo se p < 0,01 sob Conley ×1. Também se reportam ×0,5, ×2 e agrupamento por UF (F com 26 gl no denominador). O LRT e o BIC continuam reportados, mas deixam de decidir sozinhos.

| passo | χ² (Conley ×1) | gl | F ×0,5 | F ×1 | F ×2 | F por UF |
|---|---|---|---|---|---|---|
| A (região) | 63,4, p = 5,6e-13 | 4 | p = 1,7e-7 | p = 1,1e-6 | p = 3,9e-6 | 17,6, p = 4,1e-7 |
| B (log n) | 1,18, p = 0,278 | 1 | p = 0,182 | p = 0,288 | p = 0,440 | 0,78, p = 0,386 |

- **Resultado.** O passo A é mantido em todas as covariâncias; o passo B é descartado em todas (p ≥ 0,182). A dispersão fica em `regiao`. O teste robusto concorda com o BIC e discorda do LRT só na margem (seção 3), então a escolha não depende da suposição de independência. Fechado o item de [pendências](12-pendencias.md) sobre o LRT.
- **Nota.** O modelo final deixou de ser `bb_disp` por causa do SLX ([doc 08](08-dependencia-espacial.md), seção 5).

## Referências

- Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
- Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
