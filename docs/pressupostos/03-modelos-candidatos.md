# Modelos candidatos (notebook `02-MLG.ipynb`, Partes 5 a 9)

As definições vivem em `src/especificacoes.py` e são reutilizadas pela Parte 11 (validação cruzada) do `02-MLG` e por `src/backtest.py`.
A Parte 5 em diante compara e **não escolhe** o modelo final: a escolha é por validação cruzada (Parte 11).

## Estrutura

| Local | Função |
|---|---|
| `src/especificacoes.py` | Papéis de coluna (`MAPA_2026`), padronização, matriz de desenho, `CANDIDATOS`, tabelas |
| `tests/test_especificacoes.py` | Testes de vazamento, alinhamento de dummies, offset, spline, ajuste e previsão |
| `notebooks/graficos_diagnostico.py` | Helpers plotly (QQ-plot, floresta de coeficientes) |
| `results/tables/comparacao_modelos.csv` | Uma linha por candidato |
| `results/tables/coef_*.csv` | Coeficientes e EPs por modelo; `coef_binomial_quasi_bb.csv` lado a lado |
| `results/tables/selecao_variaveis.json` | Variáveis do backward por BIC e do L1 |
| `results/figures/mlg_*.png` | QQ-plot e floresta (binomial contra beta-binomial) |

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

**Conclusão.** O ajuste dentro da amostra reproduz os totais regionais por construção. Estimativas regionais honestas vêm só de previsão fora da amostra (Parte 11).

## Parte 6 - Binomial e ligações

**Função.** VIF das covariáveis, e GLM binomial (`statsmodels.GLM`, `Binomial`) com cinco ligações: logit, probit, cloglog, loglog, cauchit.

**Parâmetros.** Mesmo preditor linear nas cinco; 27 colunas além da constante (21 contínuas, com a defasagem; 2 dummies de bloco do prefeito; 4 de região).

**Pressupostos.** Independência condicional e variância binomial. Verossimilhanças com a mesma resposta e o mesmo k são comparáveis por AIC/BIC.

**Decisões e razões.**
- Nenhum VIF passou de 10 (máximo 7,84, `pct_16_24`), então nenhuma covariável foi removida.
- `bloco_prefeito_2024`: 'centro' (49) e 'sem_prefeito' (16) foram fundidos em 'outro' (65) para a dummy ter estimativa estável. Referências: 'direita' e 'Sudeste'.
- `pib_per_capita_2023` entra em log; o log é aplicado dentro de `montar_X`.

**Resultado e conclusão.**
- O logit tem o menor AIC/BIC entre as cinco ligações (AIC ≈ 169,6 mil; probit ≈ 173,2 mil; as demais ≥ 210,0 mil).
- Deviance/gl ≈ 21,21 e φ de Pearson ≈ 21,24: forte sobredispersão. Os EPs do binomial puro são pequenos demais e seus testes não valem.
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

**Decisão técnica.** O ajuste por BFGS de `BetaBinomial` já converge: no `bb_selecao` a diferença de log-verossimilhança entre o BFGS puro e o `Candidato.ajustar` é de 1e-5 (antes do ajuste do otimizador era de ~390). O polimento com L-BFGS de `Candidato.ajustar` fica como salvaguarda, e o ajuste dos candidatos sempre passa por ele.

**Resultado.**
- Quase-binomial: φ ≈ 21,24, EPs ≈ 4,61 vezes maiores; coeficientes idênticos aos do binomial.
- Beta-binomial: θ ≈ 651, ρ ≈ 0,0015. A correlação é pequena, mas com nᵢ de milhares 1 + (n−1)ρ é da ordem de 20.
- Os resíduos quantílicos têm desvio 3,4 no binomial e ≈ 1,03 no beta-binomial.
- Coeficientes do beta-binomial diferem dos do binomial, porque a variância extra tira o domínio dos municípios grandes. Exemplo (bloco com `BLOCOS_PADRAO`, como nas linhas de base): `bloco_outro` cai de 0,087 para 0,016, e `pct_vab_agro` de −0,018 para −0,008.

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
- **Offset rejeitado.** O coeficiente livre do lag é ≈ 0,89 (z ≈ −30 contra 1). LRT do offset contra livre: 2ΔlogL ≈ 831 (gl = 1). O AIC piora ≈ 829. Os municípios regridem à média em relação a 2022.
- Spline do lag (LRT ≈ 21, 3 gl) e região × lag (LRT ≈ 51, 4 gl) melhoram a verossimilhança de modo significativo, mas pouco o erro ponderado. Interações positivas no Norte (0,040) e no Centro-Oeste (0,019; p = 0,045).
- UF fixa melhora muito: ΔAIC ≈ −934 contra a região no beta-binomial. No quase-binomial o erro ponderado cai de 1,04 para 0,88 p.p. (F ≈ 53, 22 gl).
- GLMM: σ_UF ≈ 0,054 no logit; efeitos de UF de −0,09 (RS) a +0,10. Dá o mesmo erro ponderado que o efeito fixo (0,88 p.p.) e aceita UF nova (efeito 0).
- `BinomialBayesMixedGLM` do statsmodels exige resposta 0/1 e recusa dados agrupados. Por isso o GLMM é o PQL próprio de `especificacoes.py`.
- **Fusão de regiões na média.** Nordeste e Norte são fundidos (a igualdade não é rejeitada); no bloco, a regra de 8.1 funde centro e sem prefeito na direita e mantém só a esquerda separada. A fusão vale só para a linhagem `bb_selecao`; a Parte 6 (comparação de famílias) mantém `BLOCOS_PADRAO` (`bloco_outro` = centro + sem prefeito), porque compara famílias numa especificação fixa, e a fusão é um passo da seleção do `bb_selecao`. A dispersão continua com as 5 regiões: a fusão vale só para a média.
- **Seleção de variáveis.** Backward, forward e both por BIC dão o mesmo conjunto: 12 contínuas além da defasagem (`pct_dep_esq`, `comparecimento`, `delta_comp`, `log_dens`, `pct_mulheres`, `pct_16_24`, `cobertura_pbf`, `pct_vab_agro`, `pct_pretos_pardos`, `pct_fund_inc`, `pct_sem_religiao`, `pct_indigenas`); saem `nota_dep`, `log_pop`, `pct_60_mais`, `pct_superior`, `pct_analfabeto`, `pib_pc`, `pct_evangelicos` e `pct_urbana`, de 20 candidatas (o modelo cheio tem BIC 48,7 acima); BIC 72.624,48 e MAE por UF no LOSO 1,005. Modelo `bb_selecao` (19 parâmetros, nome anterior: backward). O conjunto antigo de 11 contínuas fica 215 pontos de BIC acima.
- **Outros critérios, para comparar.** AIC: backward mantém 18 variáveis (BIC 72.656,9; LOSO 1,041); forward e both mantêm 16 (LOSO 1,037). Hosmer-Lemeshow (Paula, seção 4.6.9; P_E = 0,20 e P_S = 0,25): o primeiro estágio termina em 18 variáveis. Como no passo 4 do livro (p. 360), antes das interações cada coeficiente passa por um teste de Wald e sai uma variável por vez, a de maior p acima de 0,05, porque o limiar frouxo do LRT deixa entrar variáveis sem efeito claro; saem `pct_urbana` (p = 0,130), `pib_pc` (0,169), `log_pop` (0,060) e `nota_dep` (0,053), restam 14 (BIC 72.633,8; LOSO 1,055). A tabela do π de todas as candidatas por iteração segue as Tabelas 4.8 e 4.9. Todos os subconjuntos (triagem pelo BIC quase-binomial e reajuste beta-binomial dos 200 melhores): melhor BIC 72.630,6 com 13 variáveis (LOSO 1,021), **que não achou o ótimo do stepwise** (72.624,48). É uma fraqueza da triagem, registrada, e não muda a escolha.
- [deduced] O BIC tem o menor LOSO (1,005, contra 1,021 dos subconjuntos, 1,037 a 1,041 do AIC e 1,055 do Hosmer-Lemeshow com Wald), e também a maior parcimônia; as diferenças ficam dentro de 1 EP ([05](05-validacao-cruzada.md)).
- L1: C(1 EP) = 0,0316 (o menor C de `logspace(-3, 2, 11)` dentro de 1 EP do mínimo da perda de CV de 5 grupos) mantém 6 de 27 colunas (lag, log_dens, pct_mulheres, cobertura_pbf, pct_vab_agro, pct_pretos_pardos; `coef_l1_logit.csv`). O C de menor perda de CV é 31,6 e quase não encolhe: a CV pede pouca penalização. A versão esparsa ajusta mal (erro ponderado ≈ 2,4 p.p.).

### 8.1 Categóricas e binária: regra de fusão de níveis
Roteiro de Paula (2013, seções 2.10.2, 4.2 a 4.4 e 4.6.5). A regra foi fixada antes de ver os resultados, para a fusão não depender do que se queria achar. As tabelas por nível que propõem os testes estão em [02](02-aed.md), seção 3.1.
- **Regra.** Cada fator parte dos níveis brutos (região: referência Sudeste; bloco: referência direita). A cada passo os candidatos são todos os pares de níveis fora da referência e cada nível fora da referência fundido na referência. Funde-se o de maior p do LRT se p > 0,05, reajusta-se e repete-se até nenhum candidato ter p > 0,05.
- **Binária.** `prefeito_suplementar_2024` entra só se o LRT de adicioná-la, depois das fusões, der p ≤ 0,05.
- **Onde testar.** Todos os LRT são no beta-binomial completo (20 contínuas, além da defasagem), porque o ψ bruto e o LRT univariado confundem o fator com as demais covariáveis. Os EP de Wald ignoram a correlação espacial; a significância sob essa correlação é checada com EP de Conley ([08](08-dependencia-espacial.md)).
- **Resultado nos dados completos.**
  - Região: retirar o fator inteiro, já com Nordeste e Norte fundidos, custa Δ = 208,7 (3 gl). A fusão sai do passo seguinte: Nordeste = Norte não é rejeitada e é o maior p do passo, então se fundem. Depois, Nordeste+Norte contra o Sudeste dá p = 0,003 (OR 1,017; IC95% 1,006 a 1,029) e a regra para. Os p dos passos intermediários eram de uma rodada anterior e não estão nas saídas atuais (pendência em [12](12-pendencias.md)).
  - Bloco (n: direita 4.770, esquerda 735, centro 49, sem prefeito 16): centro e sem prefeito são fundidos na direita (nenhum se distingue da referência) e só a esquerda fica separada. Retirar o bloco custa Δ = 20,0 (1 gl; p = 7,9e-6); esquerda tem OR 0,982 (IC95% 0,975 a 0,990). Com 49 e 16 municípios o teste tem pouco poder para separar centro e sem prefeito da direita. Os p dos passos de fusão (0,236, 0,474 e 0,083 de uma rodada anterior) não estão nas saídas atuais.
  - Suplementar (27 municípios): OR 0,989 (IC95% 0,957 a 1,021); Δ = 0,466 (1 gl), p = 0,495: fica fora do modelo.
  - Consequência: `bloco_outro` deixa de existir na linhagem `bb_selecao` (bloco = só `bloco_esquerda`). A Parte 6 mantém `BLOCOS_PADRAO` (centro + sem prefeito = `outro`) de propósito: ela compara famílias numa especificação fixa, e a fusão é um passo de seleção do `bb_selecao`. Decisão registrada também em [09](09-modelo-final.md).
- **Seleção aninhada no LOSO.** As fusões são refeitas em cada dobra, para o erro de validação não depender de uma fusão escolhida com a região de teste; contagens por dobra em [05](05-validacao-cruzada.md).

## Parte 9 - Comparação

**Função.** Reúne todos os candidatos em `comparacao_modelos.csv`.

**Pressupostos.** Colunas `aic`/`bic` só existem para verossimilhanças binomiais sobre (y, n): binomial, beta-binomial e L1 (aproximação sem a penalização). Quase-binomial e GLMM têm `qaic` (só o quase-binomial) ou `NaN`. Regressão beta, OLS e WLS têm outra resposta, então ficam em `NaN`.

**Resultado (dentro da amostra; modelo: AIC / BIC / φ de Pearson / erro ponderado em p.p.).**
- `bb_uf`: 71.561 / 71.899 / n.a. / 1,17
- `bb_reglag`: 72.452 / 72.670 / n.a. / 1,19
- `bb_spline`: 72.480 / 72.692 / n.a. / 1,21
- `bb_logit`: 72.495 / 72.687 / n.a. / 1,20
- `bb_selecao`: 72.499 / 72.624 / n.a. / 1,20
- `bb_offset`: 73.324 / 73.510 / n.a. / 1,40
- `bin_logit`: 169.553 / 169.739 / 21,2 / 1,04
- `quasi_*` (reglag, uf): QAIC ≈ 8,1 mil e 8,7 mil / n.a. / 21,0 e 16,9 / 1,03 e 0,88

**Atenção.** O erro ponderado dentro da amostra favorece o binomial e o quase-binomial, porque eles pesam por votos. O beta-binomial dá menos peso a municípios grandes e tem erro ponderado maior. Isso não ordena os modelos fora da amostra.

**Candidatos para a Parte 11 (chave em `CANDIDATOS`).**
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
- Usar sempre `Candidato.ajustar` para ajustar candidatos: o BFGS puro hoje converge (ver Parte 7), mas o polimento é a salvaguarda.

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
