# Dispersão regional do beta-binomial (Parte 14 do notebook 05)

Células citadas contam a partir de 0 em `notebooks/05-modelo-final.ipynb` (0 a 4 e 10 a 13). Tabelas em `results/tables/final_dispersao.csv`, `final_dp_regiao.csv`, `final_cv.csv` e `final_decisao.json`; figura em `results/figures/final_dp_regiao.html`. A parte de filtro espacial (Parte 15) fica em outro documento.

**Objetivo.** Testar se deixar a dispersão do beta-binomial variar com região (e com o porte do município) melhora o diagnóstico sem piorar a previsão fora da amostra. O ponto de partida é `bb_backward` (12 termos, escolhido no [notebook 03](03-modelos-candidatos.md) pela regra do 1-EP). O β de `bb_backward` não muda; só a dispersão muda.

**Motivação.** Com θ constante, o desvio padrão (DP) do resíduo quantílico por região ia de 0,80 (Sudeste) a 1,40 (Norte): a dispersão não é constante. Ver [diagnóstico, seção 3](04-diagnostico.md) e [pendências](07-pendencias.md).

## 1. Modelo de dispersão (célula 2)

Modelo duplo (Smyth, 1989): a média segue o preditor de `bb_backward` e a dispersão tem preditor próprio, log θᵢ = γ·zᵢ. Opções de zᵢ:

| opção | zᵢ | parâmetros extras |
|---|---|---|
| `const` | constante (o `bb_backward` original) | 0 |
| `regiao` | constante e dummies de região | 4 |
| `regiao_logn` | constante, dummies de região e z(log nᵢ) | 5 |

- **Por que região.** A heterogeneidade de dispersão observada é regional (seção 3 do diagnóstico).
- **Por que log n.** A dispersão pode depender do porte do município (n); z(log n) é a versão padronizada.
- **Escolha entre opções:** menor BIC, ajustado a todos os dados.

## 2. Regra de aceitação, escrita antes do resultado (células 2 e 16)

Fixada em 2026-11-09, antes de ver os resultados da extensão. A dispersão escolhida é aceita se:

1. o teste da razão de verossimilhanças (LRT) contra θ constante tem p < 0,01;
2. o maior |DP − 1| dos resíduos quantílicos por região é menor que com θ constante (resíduos aleatorizados, semente 0, como no [notebook 02](04-diagnostico.md));
3. o MAE por UF no LOSO é no máximo o de `bb_backward` mais 1 erro-padrão.

Falhando qualquer uma, a dispersão volta a constante. Razão do limite 3: o ganho de diagnóstico não pode custar previsão fora da amostra, que é o objetivo do projeto.

**Registro separado.** As extensões ficam em `EXTENSOES`, fora de `CANDIDATOS`. Razão: o notebook 03 monta sua lista de candidatos a partir de `CANDIDATOS`; incluí-las ali mudaria as tabelas dos notebooks 02 a 04.

## 3. Resultado do ajuste (células 3 e 4)

| opção | log-verossimilhança | k | BIC | LRT vs `const` | gl | p |
|---|---|---|---|---|---|---|
| `const` | -36311,2 | 20 | 72794,9 | n/a | n/a | n/a |
| `regiao` | -36140,7 | 24 | 72488,5 | 340,9 | 4 | 1,6e-72 |
| `regiao_logn` | -36064,6 | 25 | 72344,9 | 493,1 | 5 | 2,4e-104 |

- **Escolhida por BIC:** `regiao_logn`. Vence também `regiao` (diferença de 143,6 no BIC), então o porte do município carrega informação além da região.
- **Condição 1:** LRT p = 2,4e-104 (< 0,01). Ressalva: a verossimilhança assume independência entre municípios, que o [diagnóstico](04-diagnostico.md) não confirma; o p-valor é indicativo, e a decisão pesa mais o DP e a CV.

DP do resíduo quantílico por região:

| opção | Centro-Oeste | Nordeste | Norte | Sudeste | Sul | max \|DP − 1\| |
|---|---|---|---|---|---|---|
| `const` | 0,961 | 1,182 | 1,409 | 0,800 | 0,938 | 0,409 |
| `regiao` | 1,006 | 1,031 | 1,077 | 1,023 | 1,024 | 0,077 |
| `regiao_logn` | 0,998 | 1,025 | 1,095 | 1,009 | 1,011 | 0,095 |

- **Condição 2:** o maior |DP − 1| cai de 0,409 para 0,095. Cumprida.
- **Leitura.** `regiao` tem DP um pouco mais perto de 1 no pior caso (0,077), mas a regra escolhe por BIC, não por DP; a diferença (0,018) é pequena frente ao ganho de 0,409. O Norte continua o pior ajuste (DP 1,10).

## 4. Validação cruzada da extensão (células 16 a 19)

Candidato `bb_disp` = `bb_backward` com a dispersão `regiao_logn`. Roda em 5-fold (semente 0) e LOSO (uma UF fora por vez), sem alterar as tabelas do notebook 03.

**Limite.** MAE por UF de `bb_backward` (1,0004 p.p.) + erro-padrão do melhor modelo LOSO do notebook 03 (`bin_probit`, 0,1304) = **1,1308 p.p.** É o EP da regra do 1-EP usada para escolher `bb_backward`, para que a tolerância seja a mesma da escolha original. Informativo: o limite fica dentro da faixa de empate original (1,0605).

| esquema | modelo | MAE por UF (p.p.) | EP | MAE ponderado (p.p.) | erro nacional (p.p.) |
|---|---|---|---|---|---|
| LOSO | `bb_backward` | 1,0004 | 0,142 | 1,315 | 0,127 |
| LOSO | `bb_disp` | 0,9951 | 0,145 | 1,270 | 0,062 |
| 5-fold | `bb_backward` | 0,7886 | 0,134 | 1,159 | -0,019 |
| 5-fold | `bb_disp` | 0,7975 | 0,141 | 1,141 | -0,067 |

- **Condição 3:** 0,9951 ≤ 1,1308. Cumprida.
- **Decisão.** As três condições passam: `bb_disp` é aceita (`final_decisao.json`, `aceitas.disp = true`). O ganho em previsão é pequeno e dentro do ruído (0,005 p.p. no MAE por UF, EP 0,14); o MAE ponderado e o erro nacional melhoram mais no LOSO. A aceitação se apoia principalmente no diagnóstico (DP regional) sem custo de previsão, não em ganho de previsão.
- **O que a dispersão muda.** Os termos de média são os mesmos 12 de `bb_backward`; os coeficientes são reestimados juntos com a dispersão.

## 5. Teste de Wald robusto para a dispersão (adendo, células 6 e 7)

**Regra acrescentada em 2026-11-09, depois dos primeiros resultados do notebook** (as seções 2 a 4 foram escritas antes de ver os resultados; esta não). Razão: o LRT da seção 3 supõe municípios independentes, o que a correlação residual contradiz ([pendências](07-pendencias.md)). A regra foi fixada antes de calcular o resultado do teste.

- **Teste.** Wald sequencial sobre o vetor γ da dispersão, com a covariância de Conley do próprio ajuste (corte do correlograma de resíduos quantílicos daquele ajuste: limite inferior da primeira faixa com correlação < 0,05, regra da Parte 16, [doc 10](10-dependencia-espacial.md)). Passo A: 4 dummies de região (`regiao` contra `const`). Passo B: z(log n) (`regiao_logn` contra `regiao`), testado só se A for mantido.
- **Decisão.** Mantém-se o passo se p < 0,01 sob Conley ×1. Também se reportam ×0,5, ×2 e agrupamento por UF (F com 26 gl no denominador). O LRT e o BIC continuam reportados, mas deixam de decidir sozinhos.

| passo | χ² (Conley ×1) | gl | F ×0,5 | F ×1 | F ×2 | F por UF |
|---|---|---|---|---|---|---|
| A (região) | 56,63, p = 1,5e-11 | 4 | p = 4,3e-8 | p = 2,9e-6 | p = 2,2e-6 | 8,61, p = 1,5e-4 |
| B (log n) | 66,15, p = 4,2e-16 | 1 | p = 4,4e-10 | p = 1,3e-8 | p = 4,4e-7 | 39,9, p = 1,1e-6 |

- **Resultado.** Os dois passos são mantidos em todas as covariâncias; a dispersão segue `regiao_logn`. O teste robusto concorda com o LRT e o BIC, então a escolha da seção 3 não dependia da suposição de independência. Fechado o item de [pendências](07-pendencias.md) sobre o LRT.
- **Nota.** A tabela da seção 4 e a aceitação de `bb_disp` seguem como estavam; o modelo final deixou de ser `bb_disp` por causa do SLX ([doc 10](10-dependencia-espacial.md), seção 5).

## Referências

- Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
- Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
