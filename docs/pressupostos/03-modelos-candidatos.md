# Modelos candidatos (notebook `02-MLG.ipynb`, Partes 5 a 9)

Os índices de célula começam em 0. As definições vivem em `src/especificacoes.py` e são reutilizadas pelos notebooks 03 (validação cruzada) e 04 (backtest).
Este notebook compara e **não escolhe** o modelo final: a escolha é por validação cruzada (notebook 03).

## Estrutura

| Local | Função |
|---|---|
| `src/especificacoes.py` | Papéis de coluna (`MAPA_2026`), padronização, matriz de desenho, `CANDIDATOS`, tabelas |
| `tests/test_especificacoes.py` | Testes de vazamento, alinhamento de dummies, offset, spline, ajuste e previsão |
| `notebooks/graficos_diagnostico.py` | Helpers plotly (QQ-plot, floresta de coeficientes) |
| `results/tables/comparacao_modelos.csv` | Uma linha por candidato |
| `results/tables/coef_*.csv` | Coeficientes e EPs por modelo; `coef_binomial_quasi_bb.csv` lado a lado |
| `results/tables/selecao_variaveis.json` | Variáveis do backward por BIC e do L1 |
| `results/figures/mlg_*.html` | QQ-plot e floresta (binomial contra beta-binomial) |

## Regras de negócio e de modelagem

- Resposta: Yᵢ = votos de Lula, nᵢ = votos de Lula + Flávio, Yᵢ ~ Bin(nᵢ, πᵢ). O share de Flávio é 1 − πᵢ.
- Todas as métricas de objetivo são ponderadas por votos (erro nacional é soma de nᵢπᵢ). Por isso modelos não ponderados (regressão beta, OLS, laço) servem como comparação, não como alvo.
- Papéis genéricos de coluna (`lag`, `y`, `n`, `nota_dep`, ...) permitem ao backtest trocar os anos sem mudar o código.
- Padronização (z-score) e níveis das dummies são fixados no treino. As dummies têm colunas fixas (região e UF), então folds de teste com níveis ausentes alinham.

## Parte 5 - Formulação (células 1 a 4)

**Função.** Define o binomial agrupado e mostra a identidade dos totais regionais.

**Pressupostos.** Os eleitores do município compartilham as covariáveis municipais, e votam independentemente dado πᵢ.

**Decisões e razões.**
- O binomial agrupado em (yᵢ, nᵢ) equivale, a menos da constante binomial, à logística de voto individual com covariáveis municipais. Os votos pesam por nᵢ de forma natural.
- Gamma, normal inversa e Gaussiana ficam fora: o suporte (0, ∞) ou ℝ não respeita uma resposta limitada em (0, nᵢ) e a variância πᵢ(1−πᵢ)/nᵢ. A Gaussiana só volta como linha de base sobre o logit empírico.
- Falácia ecológica: β descreve associações entre municípios; não se infere comportamento individual.

**O que a célula 4 mostra.** Com logit e dummies de região, as equações de escore igualam Σ nᵢπ̂ᵢ e Σ yᵢ em cada região. A diferença máxima é da ordem de 1e-7 votos.

**Conclusão.** O ajuste dentro da amostra reproduz os totais regionais por construção. Estimativas regionais honestas vêm só de previsão fora da amostra (notebook 03).

## Parte 6 - Binomial e ligações (células 5 a 9)

**Função.** VIF das covariáveis, e GLM binomial (`statsmodels.GLM`, `Binomial`) com cinco ligações: logit, probit, cloglog, loglog, cauchit.

**Parâmetros.** Mesmo preditor linear nas cinco; 24 colunas além da constante (18 contínuas, 2 dummies de bloco do prefeito, 4 de região).

**Pressupostos.** Independência condicional e variância binomial. Verossimilhanças com a mesma resposta e o mesmo k são comparáveis por AIC/BIC.

**Decisões e razões.**
- Nenhum VIF passou de 10 (máximo 7,6, `pct_16_24`), então nenhuma covariável foi removida.
- `bloco_prefeito_2024`: 'centro' (49) e 'sem_prefeito' (16) foram fundidos em 'outro' (65) para a dummy ter estimativa estável. Referências: 'direita' e 'Sudeste'.
- `pib_per_capita_2023` entra em log; o log é aplicado dentro de `montar_X`.

**Resultado e conclusão.**
- O logit tem o menor AIC/BIC entre as cinco ligações (AIC ≈ 171,98 mil; probit ≈ 175,4 mil; as demais ≥ 213 mil).
- Deviance/gl ≈ 21,6 e φ de Pearson ≈ 21,7: forte sobredispersão. Os EPs do binomial puro são pequenos demais e seus testes não valem.
- A ligação logit fica; probit, cloglog, loglog e cauchit saem dos candidatos.

## Parte 7 - Sobredispersão (células 10 a 13)

**Função.** Compara tratamentos da sobredispersão.

| Modelo | Função | Observação |
|---|---|---|
| Quase-binomial | `GLM` com φ de Pearson aplicado aos EPs | Sem verossimilhança; reporta-se QAIC |
| Beta-binomial | `modelos.BetaBinomial` | Verossimilhança; θ e ρ = 1/(1+θ) |
| Regressão beta | `BetaModel`, sem pesos | Município de 1.000 eleitores pesa como São Paulo |
| OLS e WLS sobre o logit empírico | `log((y+½)/(n−y+½))`; WLS com pesos n·p(1−p) | Linhas de base |

**Pressupostos.** No beta-binomial, Var(Y) = nπ(1−π)[1 + (n−1)ρ] e πᵢ ~ Beta com precisão θ.

**Decisão técnica.** O ajuste padrão por BFGS de `BetaBinomial` para cedo (gtol frouxo): a log-verossimilhança ficou cerca de 390 abaixo do ótimo. O `Candidato.ajustar` faz um polimento com L-BFGS e mantém o melhor ajuste. Os resultados abaixo são do ajuste polido.

**Resultado.**
- Quase-binomial: φ ≈ 21,7, EPs ≈ 4,65 vezes maiores; coeficientes idênticos aos do binomial.
- Beta-binomial: θ ≈ 630, ρ ≈ 0,0016. A correlação é pequena, mas com nᵢ de milhares 1 + (n−1)ρ é da ordem de 20.
- Os resíduos quantílicos têm desvio 3,4 no binomial e ≈ 1,00 no beta-binomial.
- Coeficientes do beta-binomial diferem dos do binomial, porque a variância extra tira o domínio dos municípios grandes. Exemplo: `bloco_outro` cai de 0,079 para 0,022, e `pct_vab_agro` de −0,019 para −0,007.

**Conclusão.** O beta-binomial é o único tratamento que dá verossimilhança utilizável e descreve a dispersão. O quase-binomial mantém os pesos por votos e serve para ponto e EP corrigido. As células 11 a 12 geram `coef_binomial_quasi_bb.csv` e as figuras `mlg_*`.

**Cuidado.** A deviance/gl e o φ do beta-binomial usam a variância binomial. Não medem o ajuste dele; para isso vale o QQ-plot quantílico.

## Parte 8 - Preditor linear (células 14 a 20)

**Função.** Altera o preditor linear sobre o beta-binomial e o quase-binomial.

| Variante | Chave | Parâmetro |
|---|---|---|
| Lag como offset (swing) | `bb_offset`, `quasi_offset` | Coeficiente do lag fixo em 1 (lag bruto) |
| Spline no lag | `bb_spline`, `quasi_spline` | `patsy.bs`, 4 gl; lag do teste limitado à faixa do treino |
| Região × lag | `bb_reglag`, `quasi_reglag` | Quatro interações com o lag padronizado |
| Efeitos fixos de UF | `bb_uf`, `quasi_uf` | UF no lugar da região, referência SP |
| GLMM, intercepto por UF | `glmm_uf` | PQL com variância por EM; pesos divididos por φ |
| Backward por BIC | `bb_backward` | Remove a contínua de menor \|z\| enquanto o BIC cai |
| L1 | `l1_logit` | C por CV de 5 grupos, regra de 1 EP |

**Pressupostos.**
- O offset pressupõe persistência pura do logit de 2022 (swing uniforme).
- O backward mantém o lag, o bloco do prefeito e a região sempre, e só testa as contínuas.
- O L1 usa duas linhas por município (pesos p e 1−p, cada município com peso 1), X padronizado, e não pondera por votos.

**Resultado e conclusões.**
- **Offset rejeitado.** O coeficiente livre do lag é ≈ 0,90 (z ≈ −28 contra 1). LRT do offset contra livre: 2ΔlogL ≈ 755 (gl = 1). O AIC piora ≈ 750. Os municípios regridem à média em relação a 2022.
- Spline do lag (LRT ≈ 31, 3 gl) e região × lag (LRT ≈ 58, 4 gl) melhoram a verossimilhança de modo significativo, mas pouco o erro ponderado. Interações positivas e significativas no Norte e no Centro-Oeste.
- UF fixa melhora muito: ΔAIC ≈ −770 contra a região no beta-binomial. No quase-binomial o erro ponderado cai de 1,06 para 0,92 p.p. (F ≈ 44, 22 gl).
- GLMM: σ_UF ≈ 0,05 no logit; efeitos de UF de −0,10 (GO) a +0,09 (AC). Dá o mesmo erro ponderado que o efeito fixo (0,92 p.p.) e aceita UF nova (efeito 0).
- `BinomialBayesMixedGLM` do statsmodels exige resposta 0/1 e recusa dados agrupados (célula 18). Por isso o GLMM é o PQL próprio de `especificacoes.py`.
- Backward por BIC remove `nota_dep`, `delta_comp`, `log_pop`, `pct_60_mais`, `pib_pc` e `pct_urbana`; mantém 12 contínuas. O BIC cai pouco (≈ 33).
- L1: C(1 EP) = 0,03 mantém 6 de 24 (lag, log_dens, pct_mulheres, cobertura_pbf, pct_vab_agro, pct_pretos_pardos). O C de menor perda de CV (100) está na borda da grade e não encolhe nada: a CV pede pouca penalização. A versão esparsa ajusta mal (erro ponderado ≈ 2,4 p.p.).

## Parte 9 - Comparação (células 21 a 24)

**Função.** Reúne todos os candidatos em `comparacao_modelos.csv`.

**Pressupostos.** Colunas `aic`/`bic` só existem para verossimilhanças binomiais sobre (y, n): binomial, beta-binomial e L1 (aproximação sem a penalização). Quase-binomial e GLMM têm `qaic` (só o quase-binomial) ou `NaN`. Regressão beta, OLS e WLS têm outra resposta, então ficam em `NaN`.

**Resultado (dentro da amostra; modelo: AIC / BIC / φ de Pearson / erro ponderado em p.p.).**
- `bb_uf`: 71.886 / 72.204 / n.a. / 1,19
- `bb_reglag`: 72.605 / 72.804 / n.a. / 1,15
- `bb_spline`: 72.630 / 72.822 / n.a. / 1,18
- `bb_logit`: 72.655 / 72.827 / n.a. / 1,17
- `bb_backward`: 72.662 / 72.795 / n.a. / 1,16
- `bb_offset`: 73.408 / 73.573 / n.a. / 1,37
- `bin_logit`: 171.981 / 172.146 / 21,7 / 1,06
- `quasi_*` (reglag, uf): QAIC ≈ 8,0 mil e 8,5 mil / n.a. / 21,5 e 18,0 / 1,05 e 0,92

**Atenção.** O erro ponderado dentro da amostra favorece o binomial e o quase-binomial, porque eles pesam por votos. O beta-binomial dá menos peso a municípios grandes e tem erro ponderado maior. Isso não ordena os modelos fora da amostra.

**Candidatos para o notebook 03 (chave em `CANDIDATOS`).**
- `bb_reglag`: menor AIC sem UF; verossimilhança completa.
- `bb_backward`: menor BIC sem UF; parcimônia. Registrar com `registrar_backward(vs)` e `selecao_variaveis.json`.
- `bb_uf`: menor AIC/BIC global; só serve em k-fold, porque no leave-one-UF-out a UF de teste não existe no treino.
- `quasi_reglag`: pesa por votos, que é o objetivo.
- `glmm_uf`: encolhimento por UF; válido no leave-one-UF-out.

Linhas de base: `bb_logit` e `quasi_logit`.

## Limites e cuidados

- Os coeficientes são associações ecológicas entre municípios.
- Todos os critérios deste notebook são dentro da amostra; a ordem final sai da CV.
- O L1 é referência de parcimônia, não candidato (não pondera por votos).
- O ajuste padrão de `modelos.BetaBinomial.fit` não converge por completo (ver Parte 7). Usar sempre `Candidato.ajustar`, que faz o polimento.

## Referências

- Agresti, A. (2013). *Categorical Data Analysis* (3ª ed.). Wiley.
- McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models* (2ª ed.). Chapman and Hall.
- Williams, D. A. (1982). Extra-binomial variation in logistic linear models. *Applied Statistics*, 31(2), 144-148.
- Ferrari, S.; Cribari-Neto, F. (2004). Beta regression for modelling rates and proportions. *Journal of Applied Statistics*, 31(7), 799-815.
- Papke, L. E.; Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics*, 11(6), 619-632.
- Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
- Breslow, N. E.; Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *Journal of the American Statistical Association*, 88(421), 9-25.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society B*, 58(1), 267-288.
- Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review*, 15(3), 351-357.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
