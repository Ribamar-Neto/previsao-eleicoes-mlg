# 07 - Dispersão regional do beta-binomial (Parte 13 do notebook 05)

Notebook `notebooks/05-modelo-final.ipynb`. Tabelas em `results/tables/final_dispersao.csv`, `final_dp_regiao.csv`, `final_cv.csv` e `final_decisao.json`; figura em `results/figures/final_dp_regiao.html`. O filtro espacial (Parte 14) fica em [08](08-dependencia-espacial.md).

**Objetivo.** Testar se deixar a dispersão do beta-binomial variar com região (e com o porte do município) melhora o diagnóstico sem piorar a previsão fora da amostra. O ponto de partida é `bb_selecao` (11 contínuas e a defasagem, Nordeste e Norte fundidos na média; escolhido no notebook 03 pela regra do 1-EP, [05](05-validacao-cruzada.md)). Os termos de média não mudam; só a dispersão muda.

**Motivação.** Com θ constante, o desvio padrão (DP) do resíduo quantílico por região ia de 0,80 (Sudeste) a 1,43 (Norte): a dispersão não é constante. Ver [diagnóstico, seção 3](04-diagnostico.md) e [pendências](12-pendencias.md).

## 1. Modelo de dispersão

Modelo duplo (Smyth, 1989): a média segue o preditor de `bb_selecao` e a dispersão tem preditor próprio, log θᵢ = γ·zᵢ. A dispersão usa as **5 regiões**; a fusão de Nordeste e Norte vale só para a média. Opções de zᵢ:

| opção | zᵢ | parâmetros extras |
|---|---|---|
| `const` | constante (o `bb_selecao` original) | 0 |
| `regiao` | constante e dummies de região | 4 |
| `regiao_logn` | constante, dummies de região e z(log nᵢ) | 5 |

- **Por que região.** A heterogeneidade de dispersão observada é regional (seção 3 do diagnóstico).
- **Por que log n.** A dispersão pode depender do porte do município (n); z(log n) é a versão padronizada.
- **Escolha entre opções:** menor BIC, ajustado a todos os dados.

## 2. Regra de aceitação, escrita antes do resultado

Fixada em 2026-11-09, antes de ver os resultados da extensão. A dispersão escolhida é aceita se:

1. o teste da razão de verossimilhanças (LRT) contra θ constante tem p < 0,01;
2. o maior |DP − 1| dos resíduos quantílicos por região é menor que com θ constante (resíduos aleatorizados, semente 0, como no [diagnóstico](04-diagnostico.md));
3. o MAE por UF no LOSO é no máximo o limite L da seção 4.

Falhando qualquer uma, a dispersão volta a constante. Razão do limite 3: o ganho de diagnóstico não pode custar previsão fora da amostra, que é o objetivo do projeto.

**Registro separado.** As extensões ficam em `EXTENSOES`, fora de `CANDIDATOS`. Razão: o notebook 03 monta sua lista de candidatos a partir de `CANDIDATOS`; incluí-las ali mudaria as tabelas dos notebooks 02 a 04.

## 3. Resultado do ajuste

| opção | log-verossimilhança | k | BIC | LRT vs `const` | gl | p |
|---|---|---|---|---|---|---|
| `const` | -36340,5 | 19 | 72844,9 | n/a | n/a | n/a |
| `regiao` | -36163,3 | 23 | 72525,0 | 354,4 | 4 | 2e-75 |
| `regiao_logn` | -36087,5 | 24 | 72382,1 | 505,9 | 5 | 4e-107 |

- **Escolhida por BIC:** `regiao_logn`. Vence também `regiao` (diferença de 142,9 no BIC), então o porte do município carrega informação além da região.
- **Condição 1:** LRT p = 4e-107 (< 0,01). Ressalva: a verossimilhança assume independência entre municípios, que o [diagnóstico](04-diagnostico.md) não confirma; o p-valor é indicativo, e a decisão pesa mais o DP e a CV.

DP do resíduo quantílico por região:

| opção | Centro-Oeste | Nordeste | Norte | Sudeste | Sul | max \|DP − 1\| |
|---|---|---|---|---|---|---|
| `const` | 0,951 | 1,183 | 1,434 | 0,804 | 0,925 | 0,434 |
| `regiao` | 1,005 | 1,034 | 1,072 | 1,023 | 1,023 | 0,072 |
| `regiao_logn` | 0,997 | 1,027 | 1,087 | 1,010 | 1,010 | 0,087 |

- **Condição 2:** o maior |DP − 1| cai de 0,434 para 0,087. Cumprida.
- **Leitura.** `regiao` tem DP um pouco mais perto de 1 no pior caso (0,072), mas a regra escolhe por BIC, não por DP; a diferença (0,015) é pequena frente ao ganho de 0,434. O Norte continua o pior ajuste (DP 1,09).

## 4. Validação cruzada da extensão

Candidato `bb_disp` = `bb_selecao` com a dispersão `regiao_logn`. Roda em 5-fold (semente 0) e LOSO (uma UF fora por vez), sem alterar as tabelas do notebook 03.

**Limite L.** MAE por UF aninhado de `bb_selecao` (1,1168 p.p., seleção refeita em cada dobra; [05](05-validacao-cruzada.md), seção 8) + erro-padrão do melhor modelo LOSO do notebook 03 (`bin_probit`, 0,1347) = **1,2515 p.p.** (1,1695 com o 1,0348 sem aninhar; 1,1308 no LOSO que vazava o 2º turno da UF retida). É o EP da regra do 1-EP usada para escolher `bb_selecao`, para que a tolerância seja a mesma da escolha original. O MAE por UF e o MAE regional (`mae_uf_pp`, `mae_reg_pp`) ponderam as previsões por n̂, como nos notebooks 03 e 05; `mae_pond_pp` e `erro_nacional_pp` ponderam pelo n real.

| esquema | modelo | MAE por UF (p.p.) | EP | MAE ponderado (p.p.) | erro nacional (p.p.) |
|---|---|---|---|---|---|
| LOSO | `bb_selecao` (seleção nos dados completos) | 1,0348 | 0,156 | 1,304 | 0,076 |
| LOSO | `bb_disp` | 1,0345 | 0,161 | 1,271 | 0,051 |
| 5-fold | `bb_selecao` | 0,8550 | 0,148 | 1,150 | -0,066 |
| 5-fold | `bb_disp` | 0,8630 | 0,159 | 1,139 | -0,078 |

- **Condição 3:** 1,0345 ≤ 1,2515. Cumprida.
- **Decisão.** As três condições passam: `bb_disp` é aceita (`final_decisao.json`, `aceitas.disp = true`). O ganho em previsão é pequeno e dentro do ruído (0,0003 p.p. no MAE por UF, EP 0,16); o MAE ponderado e o erro nacional melhoram mais no LOSO. A aceitação se apoia principalmente no diagnóstico (DP regional) sem custo de previsão, não em ganho de previsão.
- **O que a dispersão muda.** Os termos de média são os mesmos de `bb_selecao`; os coeficientes são reestimados juntos com a dispersão.

## 5. Teste de Wald robusto para a dispersão

**Regra acrescentada em 2026-11-09, depois dos primeiros resultados do notebook** (as seções 2 a 4 foram escritas antes de ver os resultados; esta não). Razão: o LRT da seção 3 supõe municípios independentes, o que a correlação residual contradiz ([pendências](12-pendencias.md)). A regra foi fixada antes de calcular o resultado do teste.

- **Teste.** Wald sequencial sobre o vetor γ da dispersão, com a covariância de Conley do próprio ajuste (corte do correlograma de resíduos quantílicos daquele ajuste: limite inferior da primeira faixa com correlação < 0,05, regra da Parte 15, [doc 08](08-dependencia-espacial.md)). Passo A: 4 dummies de região (`regiao` contra `const`). Passo B: z(log n) (`regiao_logn` contra `regiao`), testado só se A for mantido.
- **Decisão.** Mantém-se o passo se p < 0,01 sob Conley ×1. Também se reportam ×0,5, ×2 e agrupamento por UF (F com 26 gl no denominador). O LRT e o BIC continuam reportados, mas deixam de decidir sozinhos.

| passo | χ² (Conley ×1) | gl | F ×0,5 | F ×1 | F ×2 | F por UF |
|---|---|---|---|---|---|---|
| A (região) | 57,4, p = 1e-11 | 4 | p = 3,5e-8 | p = 2,6e-6 | p = 1,8e-6 | 9,26, p = 8,7e-5 |
| B (log n) | 64,8, p = 8e-16 | 1 | p = 5,4e-10 | p = 1,6e-8 | p = 6,0e-7 | 40,1, p = 1,0e-6 |

- **Resultado.** Os dois passos são mantidos em todas as covariâncias; a dispersão segue `regiao_logn`. O teste robusto concorda com o LRT e o BIC, então a escolha da seção 3 não dependia da suposição de independência. Fechado o item de [pendências](12-pendencias.md) sobre o LRT.
- **Nota.** O modelo final deixou de ser `bb_disp` por causa do SLX ([doc 08](08-dependencia-espacial.md), seção 5).

## Referências

- Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
- Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
