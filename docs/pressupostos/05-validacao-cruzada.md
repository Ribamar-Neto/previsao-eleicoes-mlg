# 05 - Validação cruzada, escolha do modelo e sensibilidade regional (Parte 11)

Parte 11 do `notebooks/02-MLG.ipynb`. As estimativas regionais e seus intervalos não estão aqui: saem só do modelo final, na Parte 16 ([09](09-modelo-final.md)).

## 1. Por que validar fora da amostra

- **Decisão.** O modelo e as estimativas regionais saem de previsão fora da amostra.
- **Razão.** Um binomial com ligação canônica e dummies de região reproduz os totais regionais *exatamente* dentro da amostra (equações de escore; Partes 5 a 9).
  Esse acerto é por construção e não mede capacidade de previsão. As métricas dentro da amostra (MAE ponderado de 0,35 a 1,35 p.p., sem o `l1_logit`) também não decidem entre famílias:
  o beta-binomial dá menos peso às grandes cidades que o quasi-binomial, então os dois otimizam critérios diferentes.
- **Conclusão permitida.** Só a previsão de dados retidos compara famílias diferentes na mesma régua.

## 2. Esquemas de validação

| Esquema | Parâmetros | Pressuposto | Para que serve |
|---|---|---|---|
| 5-fold aleatório | 5 partes, semente 0 | municípios intercambiáveis | erro de município com vizinhos no treino (otimista) |
| Leave-one-UF-out (LOSO) | 27 dobras, uma UF por vez | UF retida é "inédita" | erro de prever um estado novo |

- **Razão do LOSO.** A autocorrelação espacial é forte (I de Moran do share 2026 = 0,896): os municípios não são independentes. No 5-fold, os vizinhos do município de teste estão no treino
  e o erro sai menor que o de uma previsão real. O LOSO remove esses vizinhos.
- **Modelos com efeito fixo de UF** (`quasi_uf`, `bb_uf`) entram só no 5-fold: a UF retida não tem efeito próprio no treino, e prevê-la exigiria inventar um valor.
  O `glmm_uf` (efeito aleatório de UF) entra nos dois: para UF inédita o efeito aleatório vale zero.
- **Padronização.** Média e desvio das covariáveis são recalculados em cada treino (a padronização é ajustada só na dobra de treino), sem vazamento do teste.
- **Duplicata.** `quasi_logit` produz o mesmo π̂ de `bin_logit` (mesmo estimador de β; o quasi só altera a variância). Foi avaliado uma vez. Restam 20 candidatos distintos (18 em LOSO).
- **Cache.** Previsões fora da amostra ficam em `results/cv/*.parquet`.

## 3. Métricas

Calculadas sobre as previsões fora da amostra empilhadas (cada município é previsto uma vez). **`mae_uf_pp` e `mae_reg_pp` ponderam as previsões por n̂**, o mesmo critério da Parte 11 e das Partes 13 a 17 do `02-MLG`; `mae_pond_pp`, `rmse_pond_pp` e `erro_nacional_pp` ponderam pelo n real:

- `mae_pond_pp`, `rmse_pond_pp`: erro do share municipal em p.p., ponderado por votos válidos. `mae_pp`: sem ponderar. `deviance`: deviance binomial. `erro_nacional_pp`: viés do total nacional.
- `mae_uf_pp`: média, sobre as 27 UFs, do |share previsto da UF − observado|, com share ponderado por votos. `mae_reg_pp`: o mesmo para as 5 regiões.
- **Erro-padrão (EP).** LOSO: desvio dos 27 erros por UF / √27. 5-fold: desvio do MAE ponderado dos 5 folds / √5.
- **Pressuposto do EP.** Trata as UFs (ou folds) como repetições independentes. UFs vizinhas se parecem, então o EP é um pouco otimista; serve como régua de ruído, não como teste formal.

## 4. Regra de decisão (fixada antes dos resultados)

1. **Primária**: `mae_uf_pp` em LOSO. Razão: imita prever um estado inédito, e o alvo são shares regionais, somas de UFs.
2. **1 erro-padrão**: modelos até (melhor + EP) empatam; vence o de menos parâmetros (`k` do ajuste a todos os dados; beta-binomial conta `log_theta`, quasi conta φ).
3. **Secundária**: MAE ponderado no 5-fold, para desempate com `k` igual.
4. **Relatório**: modelo escolhido e vice (o melhor por erro médio, se for outro).

## 5. Resultados (tabelas `results/tables/cv_loso.csv`, `cv_kfold.csv`, `cv_uf_loso.csv`)

O LOSO deste notebook é livre de vazamento: o ajuste de cada dobra não vê o 2º turno da UF retida. Todos os candidatos usam as covariáveis do 1º turno do mesmo ciclo (`lag1t`, `outros_1t`), conhecidas antes do 2º turno ([03](03-modelos-candidatos.md)). O conjunto de candidatas tem 22 covariáveis contínuas além da defasagem (a Parte 8 seleciona 17 delas).

LOSO, melhores por erro médio da UF (p.p.), EP, `k`, MAE ponderado municipal no 5-fold:

| Modelo | LOSO UF-MAE | EP | k | 5-fold MAE pond. |
|---|---|---|---|---|
| **bb_selecao** | **0,213** | 0,035 | **24** | 0,509 |
| bin_logit (= quasi_logit) | 0,242 | 0,041 | 30 | 0,414 |
| bb_logit | 0,243 | 0,043 | 31 | 0,508 |
| wls_elogit | 0,244 | 0,041 | 31 | 0,413 |
| ols_elogit | 0,248 | 0,043 | 31 | 0,542 |
| betareg | 0,257 | 0,043 | 31 | 0,548 |
| bb_spline | 0,259 | 0,044 | 34 | 0,489 |
| quasi_spline | 0,262 | 0,043 | 34 | 0,404 |
| bin_probit | 0,268 | 0,040 | 30 | 0,405 |
| quasi_reglag | 0,277 | 0,046 | 35 | 0,401 |
| bb_reglag | 0,302 | 0,052 | 35 | 0,472 |
| glmm_uf | 0,395 | 0,053 | 27 | 0,394 |

- **Escolhido: `bb_selecao`** (beta-binomial, logit; 17 covariáveis contínuas além da defasagem, selecionadas por BIC, com `lag1t`, Nordeste e Sudeste fundidos (3 dummies de região: Centro-Oeste, Norte, Sul) e o bloco de esquerda; 24 parâmetros, ver [03](03-modelos-candidatos.md), Parte 8). Também é o de menor erro médio. **Vice: `bin_logit`** (segundo menor erro médio).
- Limite de 1 EP, com o EP do melhor modelo da regra (o de menor erro, `bb_selecao`, 0,035): 0,213 + 0,035 = 0,248 p.p. (é a `faixa_empate_original` de `final_decisao.json`). **Quatro modelos** ficam dentro dele (`bb_selecao`, `bin_logit`, `bb_logit`, `wls_elogit`). A diferença entre eles é menor que o ruído: a escolha segue a parcimônia e **não** indica superioridade preditiva.
- Ficam claramente fora: `l1_logit` e `bin_cloglog` (0,92 p.p. cada), `bb_offset` e `quasi_offset` (1,28 e 1,24; fixar o coeficiente da defasagem em 1 piora), `bin_loglog` (0,61), `bin_cauchit` (0,44), `glmm_uf` (0,395) e, por pouco, `bb_reglag` (0,302) e `quasi_reglag` (0,277).
- **Limite de aceitação das extensões (L).** Nas Partes 11 e 13 a 17 do `02-MLG`, uma extensão do modelo escolhido só é aceita se o MAE por UF no LOSO não passa de L = MAE por UF **aninhado** de `bb_selecao` + EP do melhor modelo LOSO = 0,2497 + 0,0407 = **0,2904 p.p.** (0,2541 com o 0,2134 sem aninhar), com o EP do melhor modelo LOSO excluído o aninhado (`bin_logit`). É distinto do limite de 1 EP da escolha entre modelos (0,248 no registro; 0,283 na seleção aninhada, seção 8). A tolerância é a do 1 EP da escolha original. Ver a seção 8.
- **Ressalva sobre o alvo.** Nas métricas fora da regra, o binomial vence onde o alvo é a soma das UFs: erro médio do share regional 0,052 p.p. (`bin_logit`) contra 0,111 p.p. (`bb_selecao`); deviance LOSO 24,0 mil contra 26,4 mil. O beta-binomial dá menos peso às grandes cidades, que dominam o share regional. Quem priorizar o share regional deve preferir `bin_probit` ou `bin_logit`. A regra fixada manda `bb_selecao`.
- **5-fold.** `glmm_uf` (0,077 de MAE por UF) e `quasi_uf` (0,090) lideram porque têm efeito de UF, que só ajuda em UFs já vistas (o `bb_uf` fica em 0,196); em LOSO o `glmm_uf` vai a 0,395. Isso confirma que o efeito de UF não generaliza para UF inédita.
- **Calibração** (`results/figures/cv_calibracao.png`; saída da célula de código da calibração no LOSO, 20 faixas de π̂ de 0,23 a 0,85, ponderadas por votos): maior distância entre share previsto e observado de 0,73 p.p. no `bb_selecao` (π̂ ≈ 0,565), 0,39 p.p. no `bin_logit` e 1,50 p.p. no `bb_offset` (abaixo da diagonal em 15 das 20 faixas, π̂ ≈ 0,57). O gráfico só traz esses três modelos; o `l1_logit` não está nele.
- Figuras: `cv_loso_mae_uf.png`.

## 6. Sensibilidade regional: por que o erro regional LOSO não escolhe modelo

Tabela: `results/tables/sensibilidade_regional.csv`.

**Decisão.** A regra de escolha (menor erro LOSO por UF, depois 1 erro-padrão em favor de menos parâmetros) é mantida: `bb_selecao`. As estimativas regionais publicadas vêm só do modelo final, na Parte 16.

**Mecanismo [deduzido, não verificado por simulação].** Num binomial de ligação canônica com dummies de região, a equação de escore de cada dummy impõe Σ_u w_u δ_u = 0, com δ_u o resíduo de share da UF u dentro da amostra e w_u seu peso em votos na região.
Ao reter a UF u, o intercepto regional se move em Δ = −w_u δ_u / (1 − w_u), e o erro retido da UF fica ≈ δ_u / (1 − w_u). O erro regional LOSO é então Σ_u w_u δ_u / (1 − w_u).
- Pesos iguais: vale (1/(1 − w)) Σ_u w δ_u = 0 exatamente.
- Pesos desiguais: fica próximo de zero, não exatamente. A aproximação ignora a mudança nos demais coeficientes.
- Probit: satisfaz as equações ponderadas por votos só aproximadamente.
- Beta-binomial: pesa cada município por n/(1 + (n−1)ρ), logo o escore não é ponderado por votos e o cancelamento não ocorre.

**Consequência.** Um erro regional LOSO pequeno num binomial canônico mostra esse mecanismo, não um modelo melhor. A coluna regional de `cv_loso.csv` favorece esses modelos por construção; **a métrica justa de seleção é o erro por UF.**

Share de Lula (%), estimativa LOSO pontual; erro = estimativa − real (p.p.):

| Região | UFs | w_u máx. | Real | Probit | Logit | `bb_selecao` |
|---|---|---|---|---|---|---|
| Centro-Oeste | 4 | 0,42 | 39,99 | 40,16 (+0,17) | 40,09 (+0,10) | 40,18 (+0,19) |
| Nordeste | 9 | 0,25 | 69,37 | 69,33 (−0,04) | 69,33 (−0,04) | 69,48 (+0,11) |
| Norte | 7 | 0,48 | 50,01 | 49,89 (−0,12) | 49,91 (−0,09) | 50,03 (+0,02) |
| Sudeste | 4 | 0,51 | 47,24 | 47,23 (−0,01) | 47,23 (−0,01) | 47,45 (+0,21) |
| Sul | 3 | 0,38 | 37,42 | 37,40 (−0,02) | 37,41 (−0,02) | 37,46 (+0,03) |
| Brasil | 27 | 0,21 | 51,78 | 51,59 (−0,18) | 51,59 (−0,19) | 51,75 (−0,03) |

| Métrica (p.p.) | Probit | Logit | `bb_selecao` |
|---|---|---|---|
| MAE por UF (LOSO) | 0,268 | 0,242 | 0,213 |
| MAE regional (LOSO) | 0,071 | 0,052 | 0,111 |

- O cancelamento não é exato: o erro do logit vai de −0,19 p.p. (Brasil) a +0,10 (Centro-Oeste, w_u máx. 0,42), não zero.
- A diferença entre os três modelos vai de 0,05 p.p. (Sul) a 0,22 p.p. (Sudeste) por região e é de 0,16 p.p. no Brasil. Os três erram no máximo 0,21 p.p. em qualquer região.
- **A escolha do modelo importa pouco para a resposta regional**: muda o share em até 0,22 p.p. (a largura do IC95% regional do modelo final, 0,44 a 0,80 p.p. com Conley e σ_UF, é maior que essa diferença só em parte das regiões). Importa só onde a região é próxima de 50%: o Norte vai de 49,89% (probit) a 50,03% (`bb_selecao`).
- Em erro por UF, `bb_selecao` (0,213) é o melhor e também o de menos parâmetros entre os empatados (24 contra 30 do binomial); o custo está no erro regional (0,111 contra 0,052), pela razão acima.

## 7. O que a Parte 11 permite concluir

- A previsão de um estado inédito erra cerca de 0,25 p.p. no share da UF, com qualquer das famílias empatadas, e cerca de 0,05 a 0,11 p.p. no share regional.
- O erro fora da amostra (0,49 a 0,54 p.p. municipal ponderado em `bin_logit` e `bb_selecao`) é próximo do de dentro da amostra (0,39 e 0,50); o que o ajuste dentro da amostra esconde é o erro de nível de UF, de cerca de 0,25 p.p.
- Não permite concluir que o beta-binomial prevê melhor que o binomial.

## 8. Seleção aninhada do `bb_selecao`

A escolha das 17 contínuas e as fusões usaram o 2º turno de 2026 de todas as UFs, então o 0,213 acima é otimista. **Regra (fixada antes de rodar).** Em cada uma das 27 dobras LOSO, só com as 26 UFs de treino: testa-se a fusão de regiões e a de blocos de prefeito por razão de verossimilhanças (funde se p > 0,05); o passo a passo por BIC (backward, forward e both) fica com o conjunto de menor BIC; ajusta-se e prevê-se a UF retida com n̂. O MAE por UF dessas previsões substitui o do `bb_selecao` na regra de 1 EP e em L (`results/cv/loso_bb_selecao_aninhada.parquet`, `results/tables/cv_loso_aninhado.csv`). Os outros modelos não selecionam variáveis e ficam como estão.

- **Resultado.** MAE por UF aninhado 0,2497 p.p. (EP 0,045) contra 0,2134 (EP 0,035): +0,036 p.p., cerca de 0,81 EP aninhado. Limite de 1 EP: 0,242 + 0,041 = 0,283 (`cv_loso_aninhado.csv`); `bb_selecao` segue dentro dele (folga de 0,033 p.p.) e é o de menor k (24): **a escolha não muda**. Na regra aninhada, 10 modelos ficam dentro do limite (4 no registro).
- **Estabilidade por dobra** (Tabela 11.3 do `02-MLG`). Contínuas selecionadas: 15 a 19 por dobra, mediana 17; só 8 das 27 dobras repetem exatamente o conjunto do registro. 14 variáveis entram em todas as dobras; entram só em algumas `log_dens` (25 dobras), `nota_dep` (22), `log_pop` (11), `outros_1t` (10), `pct_analfabeto` (2) e `pct_sem_religiao` (2). O bloco de esquerda é mantido nas 27. Maiores erros por UF: BA 0,70, RJ 0,64 e ES 0,61 p.p.
- **Fusão de região.** Há fusão em 22 das 27 dobras: Centro-Oeste+Norte com Nordeste+Sudeste em 11, só Nordeste+Sudeste em 7, quatro outras variantes em 4 e nenhuma em 5. O registro funde Nordeste+Sudeste, que a seleção refeita também encontra em 18 das 27 dobras. A fusão na região de referência é mantida dentro de cada dobra (antes era descartada, erro corrigido).
- **Ressalva.** As extensões (`bb_disp`, `bb_slx`) seguem com a seleção dos dados completos; contra um L aninhado o limite fica mais folgado do que numa comparação em pé de igualdade. Nenhuma decisão aceita está perto do limite (a mais próxima, `bb_slx`, 0,194 contra 0,290).
- **Ordem de execução.** O 02 grava o LOSO do `bb_selecao` (`loso_bb_selecao.parquet`) e o 03 o lê, sem reajustar; o 03 grava `cv_loso_aninhado.csv` e `sensibilidade_regional.csv` e o 05 os lê; o 05 grava `final_decisao.json` e o Passo 6 do 02 o lê. Por isso a ordem é 02, 03, 05 e de novo 02 ([01](01-visao-geral.md), seção Organização dos notebooks).

## Referências

- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning*, 2ª ed. Springer (regra de 1 erro-padrão, cap. 7).
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8).
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B* 36(2).
- Gelman, A.; Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models*, 2ª ed. Chapman & Hall.
- Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics* 31(4) (beta-binomial).
