# 05 - Validação cruzada, escolha do modelo e sensibilidade regional (Parte 11)

Parte 11 do `notebooks/02-MLG.ipynb`. As estimativas regionais e seus intervalos não estão aqui: saem só do modelo final, na Parte 16 ([09](09-modelo-final.md)).

## 1. Por que validar fora da amostra

- **Decisão.** O modelo e as estimativas regionais saem de previsão fora da amostra.
- **Razão.** Um binomial com ligação canônica e dummies de região reproduz os totais regionais *exatamente* dentro da amostra (equações de escore; Partes 5 a 9).
  Esse acerto é por construção e não mede capacidade de previsão. As métricas dentro da amostra (MAE ponderado de 0,88 a 1,40 p.p., sem o `l1_logit`) também não decidem entre famílias:
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

O LOSO deste notebook é livre de vazamento: o ajuste de cada dobra não vê o 2º turno da UF retida. O conjunto de candidatas tem 20 covariáveis contínuas (a Parte 8 seleciona 12 delas, [03](03-modelos-candidatos.md)).

LOSO, melhores por erro médio da UF (p.p.), EP, `k`, MAE ponderado municipal no 5-fold:

| Modelo | LOSO UF-MAE | EP | k | 5-fold MAE pond. |
|---|---|---|---|---|
| **bb_selecao** | **0,994** | 0,154 | **20** | 1,192 |
| bin_logit (= quasi_logit) | 1,009 | 0,142 | 28 | 1,122 |
| wls_elogit | 1,010 | 0,142 | 29 | 1,122 |
| quasi_spline | 1,015 | 0,144 | 32 | 1,126 |
| bin_probit | 1,016 | 0,139 | 28 | 1,131 |
| ols_elogit | 1,046 | 0,159 | 29 | 1,207 |
| betareg | 1,052 | 0,162 | 29 | 1,207 |
| bb_logit | 1,060 | 0,160 | 29 | 1,208 |
| quasi_reglag | 1,065 | 0,141 | 33 | 1,117 |

- **Escolhido: `bb_selecao`** (beta-binomial, logit; 12 covariáveis contínuas além da defasagem, selecionadas por BIC, com Nordeste e Norte fundidos na média; 20 parâmetros, ver [03](03-modelos-candidatos.md), Parte 8). Também é o de menor erro médio. **Vice: `bin_logit`** (segundo menor erro médio).
- Limite de 1 EP, com o EP do melhor modelo da regra (o de menor erro, `bb_selecao`, 0,154): 0,994 + 0,154 = 1,148 p.p. (é a `faixa_empate_original` de `final_decisao.json`). **Treze modelos** ficam dentro dele. A diferença entre eles é menor que o ruído: a escolha segue a parcimônia e **não** indica superioridade preditiva.
- Ficam claramente fora: `l1_logit` (2,04 p.p.; encolhimento excessivo), `bb_offset` e `quasi_offset` (1,31-1,34; fixar o coeficiente da defasagem em 1 piora), `bin_cloglog` (1,29), `bb_reglag` (1,21).
- **Limite de aceitação das extensões.** Nas Partes 11 e 13 a 17 do `02-MLG`, uma extensão do modelo escolhido só é aceita se o MAE por UF no LOSO não passa de L = MAE por UF **aninhado** de `bb_selecao` + EP do melhor modelo LOSO = 1,0669 + 0,1419 = **1,2088 p.p.** (1,1358 com o 0,994 sem aninhar), com o EP do melhor modelo LOSO excluído o aninhado (`bin_logit`). A tolerância é a do 1 EP da escolha original. Ver a seção 8.
- **Ressalva sobre o alvo.** Nas métricas fora da regra, o binomial vence onde o alvo é a soma das UFs: erro médio do share regional 0,09 p.p. (`bin_logit`) contra 0,42 p.p. (`bb_selecao`); deviance LOSO 162,7 mil contra 167,7 mil. O beta-binomial dá menos peso às grandes cidades, que dominam o share regional. Quem priorizar o share regional deve preferir `bin_probit` ou `bin_logit`. A regra fixada manda `bb_selecao`.
- **5-fold.** `glmm_uf` (0,246 de MAE por UF) e `quasi_uf` (0,261) lideram porque têm efeito de UF, que só ajuda em UFs já vistas (o `bb_uf` também, 0,504); em LOSO o `glmm_uf` fica em 1,051. Isso confirma que o efeito de UF não generaliza para UF inédita.
- **Calibração** (`results/figures/cv_calibracao.png`): em 20 faixas de π̂, o maior desvio entre π̂ médio e share observado é 0,63 p.p. (`bb_selecao`) e 0,90 p.p. (`bin_probit`); `l1_logit`, usado como contraste, chega a 3,3 p.p.
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
| Centro-Oeste | 4 | 0,42 | 39,99 | 40,02 (+0,03) | 40,01 (+0,02) | 40,31 (+0,32) |
| Nordeste | 9 | 0,25 | 69,37 | 69,47 (+0,10) | 69,47 (+0,10) | 69,57 (+0,20) |
| Norte | 7 | 0,48 | 50,01 | 50,04 (+0,04) | 50,07 (+0,07) | 49,53 (−0,48) |
| Sudeste | 4 | 0,51 | 47,24 | 47,46 (+0,22) | 47,47 (+0,23) | 47,87 (+0,63) |
| Sul | 3 | 0,38 | 37,42 | 37,38 (−0,04) | 37,39 (−0,04) | 36,94 (−0,48) |
| Brasil | 27 | 0,21 | 51,78 | 51,73 (−0,05) | 51,73 (−0,04) | 51,84 (+0,06) |

| Métrica (p.p.) | Probit | Logit | `bb_selecao` |
|---|---|---|---|
| MAE por UF (LOSO) | 1,016 | 1,009 | 0,994 |
| MAE regional (LOSO) | 0,086 | 0,091 | 0,421 |

- O cancelamento não é exato onde um estado domina a região (SP no Sudeste, w_u = 0,51): ali o erro do logit é +0,23 p.p., não zero.
- A diferença entre os três modelos vai de 0,10 p.p. (Nordeste) a 0,55 p.p. (Norte) por região e é de 0,11 p.p. no Brasil. Os três erram menos de 0,64 p.p. em qualquer região.
- **A escolha do modelo importa pouco para a resposta regional**: muda o share em até 0,6 p.p., menos que a largura do IC95% regional (1,8 a 3,3 p.p. no modelo final). Importa só onde a região é próxima de 50%: o Norte vai de 49,53% (`bb_selecao`) a 50,07% (logit).
- Em erro por UF, `bb_selecao` (0,994) é o melhor e também o de menos parâmetros entre os empatados (20 contra 28 do binomial); o custo está no erro regional (0,421 contra 0,09), pela razão acima.

## 7. O que a Parte 11 permite concluir

- A previsão de um estado inédito erra cerca de 1 p.p. no share da UF, com qualquer das famílias empatadas, e cerca de 0,1 a 0,4 p.p. no share regional.
- O erro fora da amostra (cerca de 1,3 a 1,4 p.p. municipal ponderado) é próximo do de dentro da amostra (0,9-1,4); o que o ajuste dentro da amostra esconde é o erro de nível de UF, de cerca de 1 p.p.
- Não permite concluir que o beta-binomial prevê melhor que o binomial.

## 8. Seleção aninhada do `bb_selecao`

A escolha das 12 contínuas e a fusão Nordeste + Norte usaram o 2º turno de 2026 de todas as UFs, então o 0,994 acima é otimista. **Regra (fixada antes de rodar).** Em cada uma das 27 dobras LOSO, só com as 26 UFs de treino: testa-se a fusão de regiões e a de blocos de prefeito por razão de verossimilhanças (funde se p > 0,05); o passo a passo por BIC (backward, forward e both) fica com o conjunto de menor BIC; ajusta-se e prevê-se a UF retida com n̂. O MAE por UF dessas previsões substitui o do `bb_selecao` na regra de 1 EP e em L (`results/cv/loso_bb_selecao_aninhada.parquet`, `results/tables/cv_loso_aninhado.csv`). Os outros modelos não selecionam variáveis e ficam como estão.

- **Resultado.** MAE por UF aninhado 1,0669 p.p. (EP 0,163) contra 0,994 (EP 0,154): +0,073 p.p., cerca de metade de um EP. Limite de 1 EP: 1,009 + 0,142 = 1,151 (`cv_loso_aninhado.csv`); `bb_selecao` segue dentro dele (folga de 0,084 p.p.) e é o de menor k (20): **a escolha não muda**.
- **Estabilidade por dobra.** 11 a 14 contínuas (mediana 12); 13 das 27 dobras repetem o conjunto do registro. O conjunto de menor BIC vem do backward em 26 dobras e do forward em 1. Nordeste + Norte fundidos em 18 dobras e, em 2, também com o Sudeste (referência); Norte + Sudeste em 4, fusões próprias em 2 e nenhuma em 1. Blocos: centro + sem prefeito fundidos (esquerda separada) em 25 dobras e esquerda + sem prefeito em 2. Maiores erros: AP -3,40, DF -2,59, GO +2,35.
- **Fusão na referência.** A fusão de região na região de referência (Sudeste) agora é mantida dentro de cada dobra; antes era descartada (erro corrigido), e por isso os números acima mudaram.
- **Ressalva.** As extensões (`bb_disp`, `bb_slx`) seguem com a seleção dos dados completos; contra um L aninhado o limite fica mais folgado do que numa comparação em pé de igualdade. Nenhuma decisão aceita está perto do limite (a mais próxima, `bb_slx`, 1,060 contra 1,2088).
- **Ordem de execução.** O 02 grava o LOSO do `bb_selecao` (`loso_bb_selecao.parquet`) e o 03 o lê, sem reajustar; o 03 grava `cv_loso_aninhado.csv` e `sensibilidade_regional.csv` e o 05 os lê; o 05 grava `final_decisao.json` e o Passo 6 do 02 o lê. Por isso a ordem é 02, 03, 05 e de novo 02 ([01](01-visao-geral.md), seção Organização dos notebooks).

## Referências

- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning*, 2ª ed. Springer (regra de 1 erro-padrão, cap. 7).
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8).
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B* 36(2).
- Gelman, A.; Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models*, 2ª ed. Chapman & Hall.
- Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics* 31(4) (beta-binomial).
