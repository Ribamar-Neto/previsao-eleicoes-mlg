# Diagnóstico dos modelos binomiais (Parte 10 do notebook 02)

Células citadas contam a partir de 0 em `notebooks/02-MLG.ipynb` (25 a 38). Tabelas em `results/tables/diag_*.csv`, figuras em `results/figures/diag_*.html`, funções gráficas em `notebooks/graficos_diagnostico.py`.

**Objetivo.** Verificar os pressupostos de `bin_logit`, `quasi_reglag`, `bb_logit`, `bb_reglag`, `bb_spline` e `glmm_uf` antes da validação cruzada (notebook 03). Este documento não escolhe modelo.

**Pressupostos testados.**
1. Distribuição da resposta (binomial ou beta-binomial com θ constante).
2. Nenhum município domina o ajuste.
3. Preditor linear e variância bem especificados.
4. Independência entre municípios.

## 1. Resíduos quantílicos aleatorizados (célula 26)

- **Função:** `G.residuos_quantilicos_rapido(y, n, pi, theta, rng)`, equivalente a `modelos.residuos_quantilicos`. Sorteia u ~ U(F(y-1), F(y)) e devolve Φ⁻¹(u).
- **Parâmetros:** `theta=None` para binomial; `theta` do ajuste para beta-binomial; `rng=0` fixo (resultado reprodutível).
- **Pressuposto:** F é a distribuição ajustada. Se o modelo está certo, os resíduos são N(0,1) exatos, mesmo com resposta discreta (Dunn e Smyth, 1996). Resíduos de Pearson e de deviance não têm essa propriedade.
- **Decisão:** a FDA beta-binomial da scipy leva ~9 s por chamada para 5.570 municípios; no envelope seriam 50 chamadas. Usamos quadratura sobre a beta, com número de pontos crescente em n (a binomial vira um degrau estreito em p quando n é grande). Erro máximo da FDA ~2e-4; 99% dos resíduos diferem menos de 0,04 da versão exata, e as diferenças grandes ficam em |r| > 4. A versão rápida está só em `graficos_diagnostico.py` porque `src/` não foi alterado.
- **Para quase-binomial e GLMM** não há verossimilhança completa; usamos a binomial com o π̂ do modelo. O desvio padrão acima de 1 mede a sobredispersão que esses modelos absorvem só no erro-padrão (quase) ou no efeito de UF (GLMM).

| modelo | dp | assimetria | curtose em excesso | % \|r\|>2 | % \|r\|>3 |
|---|---|---|---|---|---|
| esperado N(0,1) | 1 | 0 | 0 | 4,55 | 0,27 |
| bin_logit | 3,43 | -0,03 | -0,61 | 55,0 | 38,5 |
| quasi_reglag | 3,42 | -0,04 | -0,60 | 55,4 | 38,7 |
| bb_logit | 1,03 | -0,43 | 3,93 | 5,0 | 1,06 |
| bb_reglag | 1,03 | -0,41 | 3,82 | 4,9 | 1,10 |
| bb_spline | 1,03 | -0,44 | 3,84 | 5,0 | 1,10 |
| glmm_uf | 3,25 | 0,04 | -0,44 | 52,0 | 35,0 |

**Conclusão.** Binomial, quase-binomial e GLMM não descrevem a dispersão (dp 3,3 a 3,4). Os beta-binomiais acertam a escala média, mas têm cauda pesada e assimetria negativa. A média do resíduo é ~0 em todos (GLMM: -0,11).

## 2. QQ-plot (células 27 e 28)

- **Função:** `G.qq_normal`. Resíduos ordenados contra quantis N(0,1); no máximo 1.500 pontos por curva.
- **Leitura:** inclinação > 1 é variância subestimada; curvatura nas caudas é cauda mais pesada que o modelo.
- **Conclusão:** coerente com a tabela acima: inclinação ~3,4 nos modelos sem θ; beta-binomial sobre a diagonal no centro, com caudas abertas.

## 3. Half-normal com envelope simulado (células 29 e 30)

- **Função:** `M.envelope_meio_normal` com `ajustar_fn`, `residuos_fn` e `simular_fn` definidos na célula 30; figura `G.meio_normal`.
- **Método:** bootstrap paramétrico. Simula y sob o modelo ajustado (beta-binomial: p ~ Beta, depois binomial), reajusta o modelo inteiro, ordena |r|, e toma os quantis 2,5% e 97,5% ponto a ponto (Atkinson, 1985; Moral et al., 2017). O reajuste leva em conta que os parâmetros foram estimados.
- **Parâmetros:** `bin_logit` B = 99 (reajuste de 0,04 s); `bb_reglag` B = 49, porque cada reajuste beta-binomial leva ~4 s (envelope em ~6 min). Com B = 49 a banda é mais ruidosa (extremos = 2º menor e 2º maior valor).
- **Resultado:**

| modelo | B | % fora | % acima | % abaixo |
|---|---|---|---|---|
| bin_logit | 99 | 99,96 | 99,96 | 0 |
| bb_reglag | 49 | 93,9 | 4,65 | 89,2 |

- **Conclusão.** `bin_logit` está fora da banda em toda a extensão (razão máxima 4,06 entre observado e limite superior). Em `bb_reglag` o corpo está abaixo da banda (resíduos pequenos demais) e as caudas acima (4,65%). Esse padrão é o de uma mistura de dispersões: [deduced] θ não é constante. O dp do resíduo por região apoia isso (`diag_dp_regiao.csv`: Sudeste 0,80; Sul 0,94; Centro-Oeste 0,95; Nordeste 1,19; Norte 1,40). Pontos fora de banda ponto a ponto não são p-valores; mesmo com o modelo certo, alguns por cento saem.

## 4. Alavanca e distância de Cook (células 31 a 33)

- **Função:** cálculo manual na célula 32; figura `G.alavanca_cook` (escala log em Cook).
- **Fórmulas:** h = diag(W^½ X (X'WX)⁻¹ X' W^½) com W = nπ̂(1-π̂); D = r_P² h / (φ̂ p (1-h)²), r_P resíduo de Pearson, φ̂ = 21,49, p = 29 (Pregibon, 1981; Cook, 1977; Williams, 1987). Feito à mão porque `get_influence()` não trata a resposta de duas colunas com φ de Pearson própria.
- **Pressuposto:** o ajuste não depende de poucas observações. Referências 2p/N e 4/N sinalizam 481 e 424 municípios; num N desse tamanho servem só de triagem. A prova é o reajuste.
- **Resultado.** Top 10 por Cook: Brasília (D = 5,12; h = 0,67), São Paulo (3,93; h = 0,56), Manaus (1,10), Fortaleza (0,72), Recife (0,39), Goiânia (0,29), Rio de Janeiro (0,23), Belém (0,10), Porto Alegre (0,09), Feira de Santana (0,09). Todos são capitais ou grandes cidades, porque W cresce com n.
- **Reajuste sem esses 10** (`diag_sem_influentes.csv`; a padronização é refeita no subconjunto, efeito desprezível):

| modelo | maior \|mudança\| | termos com \|mudança\| > 0,5 EP | trocas de sinal |
|---|---|---|---|
| quasi_reglag | 8,43 EP (`bloco_outro`) | 22 de 29 | 2 |
| bb_reglag | 0,14 EP (`bloco_outro`) | 0 de 30 | 0 |

- **Conclusão.** O quase-binomial depende das grandes cidades; o beta-binomial não, porque θ finito limita o peso de n grande. Isso favorece o beta-binomial para inferência sobre coeficientes; não diz nada sobre previsão.

## 5. Resíduos contra ajustado, `lag` e região (células 34 e 35)

- **Funções:** `G.residuos_contra` (média do resíduo em 20 faixas de mesmo tamanho, em vermelho) e `G.caixa_regiao`.
- **Pressuposto:** média 0 e dispersão 1 em qualquer faixa de η̂, de `lag` e em cada região.
- **Resultado:** Spearman do resíduo com η̂ e com `lag` entre -0,003 e 0,023 em todos os modelos; Spearman de |r| com η̂ entre 0,086 e 0,100 (a dispersão cresce levemente com a proporção ajustada). Médias regionais dos beta-binomiais ≤ 0,02 em módulo; em `bin_logit` chegam a ±0,42 e no GLMM a -0,54 (Sul).
- **Conclusão.** Não há tendência na média: o preditor linear (incluindo `bb_spline` e `bb_reglag`) não deixa forma funcional óbvia. O problema está na variância, não na média, e `bb_spline`/`bb_reglag` não melhoram os resíduos frente a `bb_logit`. A heterogeneidade regional de dispersão (seção 3) aparece de novo no box plot.

## 6. I de Moran dos resíduos (células 36 e 37)

- **Função:** `esda.Moran(r, w, permutations=999)`. Pesos: rainha, padronizados por linha, ilhas ligadas ao vizinho mais próximo (mesma receita do notebook 01; 0 ilhas restantes, 5,79 vizinhos em média). Mapa de `bb_reglag` só em HTML, geometria simplificada a 0,05°.
- **Pressuposto:** resíduos independentes. Se o preditor absorve a estrutura espacial, I ≈ 0.
- **Resultado** (`diag_moran.csv`; p_sim = 0,001 em todos, o mínimo com 999 permutações):

| modelo | I | z_sim |
|---|---|---|
| bin_logit | 0,345 | 42,7 |
| quasi_reglag | 0,352 | 43,5 |
| bb_logit | 0,317 | 40,4 |
| bb_reglag | 0,322 | 41,2 |
| bb_spline | 0,317 | 38,6 |
| glmm_uf | 0,286 | 36,3 |
| bin_logit, resíduo de Pearson | 0,334 | 41,7 |

- **Conclusão.** Há forte autocorrelação espacial residual em todos. A EDA tinha I = 0,90 na proporção observada e 0,54 no *swing*; os preditores explicam a maior parte, mas não tudo. A informação efetiva é menor que N: **os erros-padrão e intervalos de todos os modelos continuam pequenos demais**, mesmo no beta-binomial, e valores-p de coeficientes não devem ser lidos literalmente. A estimativa pontual de π não fica necessariamente enviesada. A UF aleatória é o que mais reduz I (0,29), pouco. A checagem honesta de desempenho é a validação deixando uma UF de fora (notebook 03); ela deixa vizinhos de fronteira no treino, então ainda é levemente otimista.

## 7. O que se pode concluir (célula 38)

| modelo | distribuição | preditor | influentes | independência |
|---|---|---|---|---|
| bin_logit | falha | ok | não avaliado | falha |
| quasi_reglag | falha na forma | ok | falha | falha |
| bb_logit, bb_spline | escala ok; formato e θ constante falham | ok | ok (irmão `bb_reglag`) | falha |
| bb_reglag | escala ok; formato e θ constante falham | ok | ok | falha |
| glmm_uf | falha | ok | não avaliado | falha (menor I) |

**Decisões registradas.**
- Nenhum modelo é escolhido aqui: os diagnósticos medem pressupostos, e a escolha é por validação cruzada (notebook 03).
- O beta-binomial é o único com resíduos de escala correta e coeficientes estáveis; o que falta nele é dispersão por região e dependência espacial.
- Remédios candidatos, não implementados: θ dependente da região; termo espacial; erros-padrão robustos com agrupamento por UF.
- Limitação: o envelope de `bb_reglag` usa B = 49 por custo; os resíduos são aleatorizados (uma realização, `rng=0`), então as contagens por faixa variam na segunda casa decimal entre sementes.

## Referências

- Atkinson, A. C. (1985). *Plots, Transformations and Regression*. Oxford University Press.
- Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15-18.
- Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
- Moral, R. A.; Hinde, J.; Demétrio, C. G. B. (2017). Half-normal plots and overdispersed models in R: the hnp package. *Journal of Statistical Software*, 81(10), 1-23.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
- Pregibon, D. (1981). Logistic regression diagnostics. *The Annals of Statistics*, 9(4), 705-724.
- Williams, D. A. (1987). Generalized linear model diagnostics using the deviance and single case deletions. *Applied Statistics (JRSS C)*, 36(2), 181-191.
