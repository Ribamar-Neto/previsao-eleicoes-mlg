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
- **Covariáveis do 1º turno do mesmo ciclo.** `lag1t` (logit da parcela do PT entre os dois finalistas no 1º turno do próprio ano) e `outros_1t` (parcela dos válidos do 1º turno que foi para os demais candidatos) são candidatas legítimas: o 1º turno é conhecido antes do 2º, e a regra de informação ([09](09-modelo-final.md), seção 1) só proíbe usar o 2º turno da UF prevista. Denominador zero dá logit 0 e `outros_1t` 0. A `lag` (2º turno do ciclo anterior) continua no modelo.
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

**Parâmetros.** Mesmo preditor linear nas cinco; 29 colunas além da constante (23 contínuas, com `lag`, `lag1t` e `outros_1t`; 2 dummies de bloco do prefeito; 4 de região).

**Pressupostos.** Independência condicional e variância binomial. Verossimilhanças com a mesma resposta e o mesmo k são comparáveis por AIC/BIC.

**Decisões e razões.**
- Só `lag1t` (79,3) e `lag` (73,2) passam de 10: medem quase a mesma coisa (o PT entre os finalistas, em 2026 e em 2022). Pela regra da [02](02-aed.md), a colinearidade é reportada e não decide descarte; nenhuma covariável foi removida. Entre as demais o máximo é 8,02 (`pct_16_24`).
- `bloco_prefeito_2024`: 'centro' (49) e 'sem_prefeito' (16) foram fundidos em 'outro' (65) para a dummy ter estimativa estável. Referências: 'direita' e 'Sudeste'.
- `pib_per_capita_2023` entra em log; o log é aplicado dentro de `montar_X`.

**Resultado e conclusão.**
- O logit tem o menor AIC/BIC entre as cinco ligações (AIC ≈ 68,96 mil; probit ≈ 68,97 mil, só 12,5 acima; as demais ≥ 119,4 mil).
- Deviance/gl ≈ 3,06 e φ de Pearson ≈ 3,07: sobredispersão menor que sem o 1º turno, mas ainda clara. Os EPs do binomial puro são pequenos demais e seus testes não valem.
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
- Quase-binomial: φ ≈ 3,07, EPs ≈ 1,75 vezes maiores; coeficientes idênticos aos do binomial.
- Beta-binomial: θ ≈ 7,6 mil, ρ ≈ 0,00013. A correlação é pequena; para o n mediano (7.201), 1 + (n−1)ρ ≈ 1,9, e cresce com n.
- Os resíduos quantílicos têm desvio 1,66 no binomial e ≈ 0,98 no beta-binomial.
- Coeficientes do beta-binomial diferem dos do binomial, porque a variância extra tira o domínio dos municípios grandes. Exemplo (bloco com `BLOCOS_PADRAO`, como nas linhas de base): `bloco_outro` cai de 0,012 (quase-binomial) para 0,005, e `pct_vab_agro` de −0,0066 para −0,0036.

**Conclusão.** O beta-binomial é o único tratamento que dá verossimilhança utilizável e descreve a dispersão. O quase-binomial mantém os pesos por votos e serve para ponto e EP corrigido. Gera `coef_binomial_quasi_bb.csv` e as figuras `mlg_*`.

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
| Seleção de variáveis por BIC | `bb_selecao` | Backward e both dão o mesmo conjunto; forward acrescenta `pct_analfabeto`; funde Nordeste na referência Sudeste |
| L1 | `l1_logit` | C por CV de 5 grupos, regra de 1 EP |

**Pressupostos.**
- O offset pressupõe persistência pura do logit de 2022 (swing uniforme).
- A seleção mantém o lag, o bloco do prefeito e a região sempre, e só testa as contínuas. A fusão de níveis da região testa a igualdade de dois coeficientes por razão de verossimilhanças.
- O L1 usa duas linhas por município (pesos p e 1−p, cada município com peso 1), X padronizado, e não pondera por votos.

**Resultado e conclusões.**
- **Offset rejeitado.** LRT do offset contra livre: 2ΔlogL ≈ 10.803 (gl = 1); o AIC piora ≈ 10.801. Com o 1º turno no modelo, o nível do PT vem de `lag1t` (coeficiente padronizado 0,757, contra 0,077 de `lag`); fixar `lag` em 1 desloca esse peso e destrói o ajuste. O coeficiente livre de `lag` em escala bruta é 0,089 no beta-binomial e 0,075 no quase-binomial (Wald contra 1 rejeita nos dois).
- Spline do lag (LRT ≈ 81, 3 gl) e região × lag (LRT ≈ 142, 4 gl) melhoram a verossimilhança de modo significativo, mas pouco o erro ponderado. Interações positivas no Nordeste (0,023) e no Norte (0,022); Centro-Oeste 0,001 e Sul 0,004 (p = 0,094).
- UF fixa melhora muito: ΔAIC ≈ −833 contra a região no beta-binomial. No quase-binomial o erro ponderado cai de 0,39 para 0,35 p.p.
- GLMM: dá o mesmo erro ponderado que o efeito fixo (0,35 p.p.) e aceita UF nova (efeito 0). σ_UF = 0,019 no logit (φ = 2,64); os efeitos por UF vão de −0,034 a 0,040.
- `BinomialBayesMixedGLM` do statsmodels exige resposta 0/1 e recusa dados agrupados. Por isso o GLMM é o PQL próprio de `especificacoes.py`.
- **Fusão na média.** Região: a regra de 8.1 funde Nordeste na referência Sudeste (p = 0,085) e o `bb_selecao` mantém a fusão (`regiao_fundir` = Nordeste + Sudeste no `selecao_variaveis.json`): três dummies de região (Centro-Oeste, Norte, Sul). Bloco: a regra de 8.1 funde centro e sem prefeito na direita e mantém só a esquerda separada. A fusão vale só para a linhagem `bb_selecao`; a Parte 6 (comparação de famílias) mantém `BLOCOS_PADRAO` (`bloco_outro` = centro + sem prefeito), porque compara famílias numa especificação fixa. A dispersão usa as 5 regiões.
- **Seleção de variáveis.** 22 contínuas candidatas além da defasagem, `lag1t` e `outros_1t` incluídas. Backward e both por BIC dão o mesmo conjunto: 17 contínuas além de `lag` (`lag1t`, `nota_dep`, `pct_dep_esq`, `comparecimento`, `delta_comp`, `log_pop`, `log_dens`, `pct_mulheres`, `pct_60_mais`, `pct_superior`, `cobertura_pbf`, `pib_pc`, `pct_vab_agro`, `pct_evangelicos`, `pct_pretos_pardos`, `pct_urbana`, `pct_fund_inc`), 18 com a defasagem. **`lag1t` entra e `outros_1t` não** (o BIC não paga o termo). Saem `pct_16_24`, `pct_analfabeto`, `pct_sem_religiao` e `pct_indigenas`, além de `outros_1t`. BIC 62.571,4 (AIC 62.412,4; o modelo cheio tem BIC 25,3 acima) e MAE por UF no LOSO 0,213 p.p. Modelo `bb_selecao` (24 parâmetros, nome anterior: backward). O conjunto antigo de 11 contínuas, sem `lag1t`, fica 10.275 pontos de BIC acima: o 1º turno do ciclo é o ganho dominante.
- **Outros critérios, para comparar.** Forward por BIC: 18 contínuas (acrescenta `pct_analfabeto`; BIC 62.575,3; LOSO 0,216). AIC (backward, forward e both): 21 variáveis (acrescentam `outros_1t`, `pct_16_24`, `pct_analfabeto` e `pct_sem_religiao`; BIC 62.588,1; LOSO 0,227). Hosmer-Lemeshow (Paula, seção 4.6.9; P_E = 0,20 e P_S = 0,25): o primeiro estágio termina em 21 variáveis; como no passo 4 do livro (p. 360), antes das interações cada coeficiente passa por um teste de Wald e sai uma variável por vez, a de maior p acima de 0,05, porque o limiar frouxo do LRT deixa entrar variáveis sem efeito claro; só `pct_sem_religiao` (p = 0,084) sai, restam 20 (BIC 62.582,5; LOSO 0,220). Todos os subconjuntos (triagem pelo BIC quase-binomial e reajuste beta-binomial dos 200 melhores): melhor BIC 62.571,7 com 16 variáveis (LOSO 0,207), **que não achou o ótimo do stepwise** (62.571,4, 0,3 ponto abaixo). Fraqueza da triagem; não muda a escolha.
- [deduced] O conjunto do BIC tem o menor BIC e um LOSO entre os menores (0,213, contra 0,207 dos subconjuntos, 0,216 do forward, 0,227 do AIC e 0,229 do cheio); as diferenças ficam dentro de 1 EP (0,035; [05](05-validacao-cruzada.md)), então vale a parcimônia.
- L1: com a regra de 1 EP sobre a perda de CV de 5 grupos, só `lag1t` fica com coeficiente diferente de zero (`coef_l1_logit.csv`); o encolhimento é excessivo e a versão esparsa ajusta mal (erro ponderado ≈ 1,07 p.p.).

### 8.1 Categóricas e binária: regra de fusão de níveis
Roteiro de Paula (2013, seções 2.10.2, 4.2 a 4.4 e 4.6.5). A regra foi fixada antes de ver os resultados, para a fusão não depender do que se queria achar. As tabelas por nível que propõem os testes estão em [02](02-aed.md), seção 3.1.
- **Regra.** Cada fator parte dos níveis brutos (região: referência Sudeste; bloco: referência direita). A cada passo os candidatos são todos os pares de níveis fora da referência e cada nível fora da referência fundido na referência. Funde-se o de maior p do LRT se p > 0,05, reajusta-se e repete-se até nenhum candidato ter p > 0,05.
- **Rótulo da referência nas tabelas de passos.** O rótulo da referência acumula os níveis já fundidos nela (direita, depois direita+centro, depois direita+centro+sem_prefeito), para cada passo mostrar o que a referência contém; só o rótulo muda, não os números nem as decisões.
- **Binária.** `prefeito_suplementar_2024` entra só se o LRT de adicioná-la, depois das fusões, der p ≤ 0,05.
- **Onde testar.** Todos os LRT são no beta-binomial completo (22 contínuas, além da defasagem), porque o ψ bruto e o LRT univariado confundem o fator com as demais covariáveis. Os EP de Wald ignoram a correlação espacial; a significância sob essa correlação é checada com EP de Conley ([08](08-dependencia-espacial.md)).
- **Resultado nos dados completos.**
  - Região: no passo 0, o maior p é Nordeste contra a referência Sudeste (p = 0,085, Δ = 2,97), que a regra funde; nenhum outro par passa de 0,05 (o mais alto é Centro-Oeste × Norte, p = 0,010, no passo 0 e 0,039 no passo 1). A fusão vale também no `bb_selecao` (ver Parte 8, fusão na média). Passos em `results/tables/fusao_regiao_passos.csv`. Com a fusão, o fator tem 3 gl (Δ = 342,7; p = 5,7e-74) e OR contra a referência: Centro-Oeste 0,986, Norte 0,980, Sul 0,968.
  - Bloco (n: direita 4.770, esquerda 735, centro 49, sem prefeito 16): centro e sem prefeito são fundidos na direita (nenhum se distingue da referência) e só a esquerda fica separada. Esquerda contra a direita: Δ = 12,4 (1 gl; p = 4,2e-4; no passo 0, Δ = 12,0 e p = 5,4e-4, `fusao_bloco_passos.csv`); OR 0,995 (IC95% 0,992 a 0,998). Com 49 e 16 municípios o teste tem pouco poder. Passos (Tabela 8.12): centro com sem prefeito p = 0,592 (fundidos entre si no passo 0); depois centro+sem prefeito com direita p = 0,232 (fundidos na direita).
  - Suplementar (27 municípios): fica fora do modelo; LRT de adicioná-la: OR 0,998 (IC95% 0,985 a 1,010), p = 0,727.
  - Consequência: `bloco_outro` deixa de existir na linhagem `bb_selecao` (bloco = só `bloco_esquerda`). A Parte 6 mantém `BLOCOS_PADRAO` (centro + sem prefeito = `outro`) de propósito: ela compara famílias numa especificação fixa, e a fusão é um passo de seleção do `bb_selecao`. Decisão registrada também em [09](09-modelo-final.md).
- **Seleção aninhada no LOSO.** As fusões são refeitas em cada dobra, para o erro de validação não depender de uma fusão escolhida com a região de teste; contagens por dobra em [05](05-validacao-cruzada.md).

## Parte 9 - Comparação

**Função.** Reúne todos os candidatos em `comparacao_modelos.csv`.

**Pressupostos.** Colunas `aic`/`bic` só existem para verossimilhanças binomiais sobre (y, n): binomial, beta-binomial e L1 (aproximação sem a penalização). Quase-binomial e GLMM têm `qaic` (só o quase-binomial) ou `NaN`. Regressão beta, OLS e WLS têm outra resposta, então ficam em `NaN`.

**Resultado (dentro da amostra; modelo: AIC / BIC / φ de Pearson / erro ponderado em p.p.).**
- `bb_uf`: 61.571 / 61.923 / n.a. / 0,50
- `bb_reglag`: 62.270 / 62.502 / n.a. / 0,47
- `bb_spline`: 62.329 / 62.554 / n.a. / 0,48
- `bb_logit`: 62.404 / 62.610 / n.a. / 0,50
- `bb_selecao`: 62.412 / 62.571 / n.a. / 0,50
- `bb_offset`: 73.205 / 73.404 / n.a. / 1,35
- `bin_logit`: 68.957 / 69.156 / 3,07 / 0,39
- `quasi_*` (reglag, uf): QAIC ≈ 23,3 mil e 25,2 mil / n.a. / 2,93 e 2,66 / 0,38 e 0,35

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
