# Modelos candidatos (notebook `02-MLG.ipynb`, Partes 5 a 9)

As definições vivem em `src/especificacoes.py` e são reutilizadas pelos notebooks 03 (validação cruzada) e 04 (backtest).
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

## Parte 5 - Formulação

**Função.** Define o binomial agrupado e mostra a identidade dos totais regionais.

**Pressupostos.** Os eleitores do município compartilham as covariáveis municipais, e votam independentemente dado πᵢ.

**Decisões e razões.**
- O binomial agrupado em (yᵢ, nᵢ) equivale, a menos da constante binomial, à logística de voto individual com covariáveis municipais. Os votos pesam por nᵢ de forma natural.
- Gamma, normal inversa e Gaussiana ficam fora: o suporte (0, ∞) ou ℝ não respeita uma resposta limitada em (0, nᵢ) e a variância πᵢ(1−πᵢ)/nᵢ. A Gaussiana só volta como linha de base sobre o logit empírico.
- Falácia ecológica: β descreve associações entre municípios; não se infere comportamento individual.

**O que a identidade mostra.** Com logit e dummies de região, as equações de escore igualam Σ nᵢπ̂ᵢ e Σ yᵢ em cada região. A diferença máxima é da ordem de 1e-7 votos.

**Conclusão.** O ajuste dentro da amostra reproduz os totais regionais por construção. Estimativas regionais honestas vêm só de previsão fora da amostra (notebook 03).

## Parte 6 - Binomial e ligações

**Função.** VIF das covariáveis, e GLM binomial (`statsmodels.GLM`, `Binomial`) com cinco ligações: logit, probit, cloglog, loglog, cauchit.

**Parâmetros.** Mesmo preditor linear nas cinco; 24 colunas além da constante (18 contínuas, 2 dummies de bloco do prefeito, 4 de região).

**Pressupostos.** Independência condicional e variância binomial. Verossimilhanças com a mesma resposta e o mesmo k são comparáveis por AIC/BIC.

**Decisões e razões.**
- Nenhum VIF passou de 10 (máximo 7,6, `pct_16_24`), então nenhuma covariável foi removida.
- `bloco_prefeito_2024`: 'centro' (49) e 'sem_prefeito' (16) foram fundidos em 'outro' (65) para a dummy ter estimativa estável. Referências: 'direita' e 'Sudeste'.
- `pib_per_capita_2023` entra em log; o log é aplicado dentro de `montar_X`.

**Resultado e conclusão.**
- O logit tem o menor AIC/BIC entre as cinco ligações (AIC ≈ 171,98 mil; probit ≈ 175,4 mil; as demais ≥ 213 mil).
- Deviance/gl ≈ 21,8 e φ de Pearson ≈ 21,8: forte sobredispersão. Os EPs do binomial puro são pequenos demais e seus testes não valem.
- A ligação logit fica; probit, cloglog, loglog e cauchit saem dos candidatos.

## Parte 7 - Sobredispersão

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
- Quase-binomial: φ ≈ 21,8, EPs ≈ 4,65 vezes maiores; coeficientes idênticos aos do binomial.
- Beta-binomial: θ ≈ 624, ρ ≈ 0,0016. A correlação é pequena, mas com nᵢ de milhares 1 + (n−1)ρ é da ordem de 20.
- Os resíduos quantílicos têm desvio 3,4 no binomial e ≈ 1,00 no beta-binomial.
- Coeficientes do beta-binomial diferem dos do binomial, porque a variância extra tira o domínio dos municípios grandes. Exemplo: `bloco_outro` cai de 0,079 para 0,022, e `pct_vab_agro` de −0,019 para −0,007.

**Conclusão.** O beta-binomial é o único tratamento que dá verossimilhança utilizável e descreve a dispersão. O quase-binomial mantém os pesos por votos e serve para ponto e EP corrigido. A comparação gera `coef_binomial_quasi_bb.csv` e as figuras `mlg_*`.

**Cuidado.** A deviance/gl e o φ do beta-binomial usam a variância binomial. Não medem o ajuste dele; para isso vale o QQ-plot quantílico.

## Parte 8 - Preditor linear

**Função.** Altera o preditor linear sobre o beta-binomial e o quase-binomial.

| Variante | Chave | Parâmetro |
|---|---|---|
| Lag como offset (swing) | `bb_offset`, `quasi_offset` | Coeficiente do lag fixo em 1 (lag bruto) |
| Spline no lag | `bb_spline`, `quasi_spline` | `patsy.bs`, 4 gl; lag do teste limitado à faixa do treino |
| Região × lag | `bb_reglag`, `quasi_reglag` | Quatro interações com o lag padronizado |
| Efeitos fixos de UF | `bb_uf`, `quasi_uf` | UF no lugar da região, referência SP |
| GLMM, intercepto por UF | `glmm_uf` | PQL com variância por EM; pesos divididos por φ |
| Seleção de variáveis por BIC | `bb_selecao` | Backward, forward e both por BIC dão o mesmo conjunto; Nordeste e Norte fundidos na média |
| L1 | `l1_logit` | C por CV de 5 grupos, regra de 1 EP |

**Pressupostos.**
- O offset pressupõe persistência pura do logit de 2022 (swing uniforme).
- A seleção mantém o lag, o bloco do prefeito e a região sempre, e só testa as contínuas. A fusão de níveis da região testa a igualdade de dois coeficientes por razão de verossimilhanças.
- O L1 usa duas linhas por município (pesos p e 1−p, cada município com peso 1), X padronizado, e não pondera por votos.

**Resultado e conclusões.**
- **Offset rejeitado.** O coeficiente livre do lag é ≈ 0,90 (z ≈ −28 contra 1). LRT do offset contra livre: 2ΔlogL ≈ 755 (gl = 1). O AIC piora ≈ 750. Os municípios regridem à média em relação a 2022.
- Spline do lag (LRT ≈ 31, 3 gl) e região × lag (LRT ≈ 58, 4 gl) melhoram a verossimilhança de modo significativo, mas pouco o erro ponderado. Interações positivas e significativas no Norte e no Centro-Oeste.
- UF fixa melhora muito: ΔAIC ≈ −770 contra a região no beta-binomial. No quase-binomial o erro ponderado cai de 1,06 para 0,92 p.p. (F ≈ 44, 22 gl).
- GLMM: σ_UF ≈ 0,05 no logit; efeitos de UF de −0,10 (GO) a +0,09 (AC). Dá o mesmo erro ponderado que o efeito fixo (0,92 p.p.) e aceita UF nova (efeito 0).
- `BinomialBayesMixedGLM` do statsmodels exige resposta 0/1 e recusa dados agrupados. Por isso o GLMM é o PQL próprio de `especificacoes.py`.
- **Fusão de regiões na média.** Nordeste e Norte são fundidos (teste de igualdade, p = 0,921); o bloco do prefeito não é fundido (p = 8e-4). A dispersão continua com as 5 regiões: a fusão vale só para a média.
- **Seleção de variáveis.** Backward, forward e both por BIC dão o mesmo conjunto: 11 contínuas além da defasagem (saem `nota_dep`, `delta_comp`, `log_pop`, `pct_60_mais`, `pib_pc` e `pct_urbana`); BIC 72.844,87 e MAE por UF no LOSO 1,035. Modelo `bb_selecao` (nome anterior: backward).
- **Outros critérios, para comparar.** AIC (os três sentidos): 15 variáveis, LOSO 1,003. Hosmer-Lemeshow (Paula, seção 4.6.9; P_E = 0,20 e P_S = 0,25): o primeiro estágio termina em 15 variáveis. Todos os subconjuntos (triagem pelo BIC quase-binomial e reajuste beta-binomial dos 200 melhores): melhor BIC 72.848,8 com 13 variáveis, **que não achou o ótimo do stepwise** (72.844,87). É uma fraqueza da triagem, registrada, e não muda a escolha.
- [deduced] O AIC e os subconjuntos têm LOSO melhor que o BIC (1,003 e 1,000 contra 1,035), mas a diferença fica dentro de 1 EP ([05](05-validacao-cruzada.md)); o BIC segue como critério por parcimônia.
- L1: C(1 EP) = 0,03 mantém 6 de 24 (lag, log_dens, pct_mulheres, cobertura_pbf, pct_vab_agro, pct_pretos_pardos). O C de menor perda de CV (100) está na borda da grade e não encolhe nada: a CV pede pouca penalização. A versão esparsa ajusta mal (erro ponderado ≈ 2,4 p.p.).

### 8.1 Categóricas e binária: regra de fusão de níveis
Roteiro de Paula (2013, seções 2.10.2, 4.2 a 4.4 e 4.6.5). A regra foi fixada antes de ver os resultados, para a fusão não depender do que se queria achar. As tabelas por nível que propõem os testes estão em [02](02-aed.md), seção 3.1.
- **Regra.** Cada fator parte dos níveis brutos (região: referência Sudeste; bloco: referência direita). A cada passo os candidatos são todos os pares de níveis fora da referência e cada nível fora da referência fundido na referência. Funde-se o de maior p do LRT se p > 0,05, reajusta-se e repete-se até nenhum candidato ter p > 0,05.
- **Binária.** `prefeito_suplementar_2024` entra só se o LRT de adicioná-la, depois das fusões, der p ≤ 0,05.
- **Onde testar.** Todos os LRT são no beta-binomial completo (18 contínuas), porque o ψ bruto e o LRT univariado confundem o fator com as demais covariáveis. Os EP de Wald ignoram a correlação espacial; a significância sob essa correlação é checada com EP de Conley ([08](08-dependencia-espacial.md)).
- **Resultado nos dados completos.**
  - Região: retirar o fator inteiro custa Δ = 233,4 (4 gl). Nordeste = Norte não é rejeitada (p = 0,921) e é o maior p do passo, então se fundem. Depois, Nordeste+Norte contra o Sudeste dá p = 0,018 e todos os outros candidatos têm p < 1e-8: a regra para. A log-verossimilhança não muda (-36.327,71 para -36.327,72).
  - Bloco: esquerda fica separada (OR 0,984; esquerda = direita rejeitada, p < 1e-4). Centro = sem_prefeito não é rejeitada (p = 0,466), então se fundem; o grupo contra a direita dá p = 0,046 e a regra para. Isso reproduz o agrupamento anterior (esquerda; centro+sem_prefeito). Com 49 e 16 municípios o teste tem pouco poder para separar centro de sem_prefeito.
  - Suplementar: OR 0,992, p = 0,649: fica fora do modelo.
  - Consequência: `CANDIDATOS` e os notebooks 04 a 06 não mudam.
- **Seleção aninhada no LOSO.** As fusões são refeitas em cada dobra, para o erro de validação não depender de uma fusão escolhida com a região de teste; contagens por dobra em [05](05-validacao-cruzada.md).

## Parte 9 - Comparação

**Função.** Reúne todos os candidatos em `comparacao_modelos.csv`.

**Pressupostos.** Colunas `aic`/`bic` só existem para verossimilhanças binomiais sobre (y, n): binomial, beta-binomial e L1 (aproximação sem a penalização). Quase-binomial e GLMM têm `qaic` (só o quase-binomial) ou `NaN`. Regressão beta, OLS e WLS têm outra resposta, então ficam em `NaN`.

**Resultado (dentro da amostra; modelo: AIC / BIC / φ de Pearson / erro ponderado em p.p.).**
- `bb_uf`: 71.919 / 72.238 / n.a. / 1,17
- `bb_reglag`: 72.656 / 72.855 / n.a. / 1,14
- `bb_spline`: 72.680 / 72.872 / n.a. / 1,17
- `bb_logit`: 72.707 / 72.880 / n.a. / 1,15
- `bb_selecao`: 72.719 / 72.845 / n.a. / 1,15
- `bb_offset`: 73.480 / 73.646 / n.a. / 1,35
- `bin_logit`: 172.593 / 172.759 / 21,8 / 1,06
- `quasi_*` (reglag, uf): QAIC ≈ 8,0 mil e 8,5 mil / n.a. / 21,6 e 17,9 / 1,06 e 0,91

**Atenção.** O erro ponderado dentro da amostra favorece o binomial e o quase-binomial, porque eles pesam por votos. O beta-binomial dá menos peso a municípios grandes e tem erro ponderado maior. Isso não ordena os modelos fora da amostra.

**Candidatos para o notebook 03 (chave em `CANDIDATOS`).**
- `bb_reglag`: menor AIC sem UF; verossimilhança completa.
- `bb_selecao`: menor BIC sem UF; parcimônia (variáveis em `selecao_variaveis.json`).
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
