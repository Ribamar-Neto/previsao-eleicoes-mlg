# 05 - Validação cruzada, escolha do modelo e sensibilidade regional (Parte 11)

Notebook: `notebooks/03-validacao.ipynb`. As estimativas regionais e seus intervalos não estão aqui: saem só do modelo final, na Parte 16 ([09](09-modelo-final.md)).

## 1. Por que validar fora da amostra

- **Decisão.** O modelo e as estimativas regionais saem de previsão fora da amostra.
- **Razão.** Um binomial com ligação canônica e dummies de região reproduz os totais regionais *exatamente* dentro da amostra (equações de escore; notebook 02).
  Esse acerto é por construção e não mede capacidade de previsão. As métricas dentro da amostra (MAE ponderado de 0,92 a 1,37 p.p.) também não decidem entre famílias:
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

Calculadas sobre as previsões fora da amostra empilhadas (cada município é previsto uma vez). **`mae_uf_pp` e `mae_reg_pp` ponderam as previsões por n̂**, o mesmo critério dos notebooks 03 e 05; `mae_pond_pp`, `rmse_pond_pp` e `erro_nacional_pp` ponderam pelo n real:

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

O LOSO deste notebook é livre de vazamento: o ajuste de cada dobra não vê o 2º turno da UF retida. Antes da correção, `bin_probit` dava 0,930 e o modelo então escolhido, 1,000 (EP 0,130); os números abaixo substituem esses.

LOSO, melhores por erro médio da UF (p.p.), EP, `k`, MAE ponderado municipal no 5-fold:

| Modelo | LOSO UF-MAE | EP | k | 5-fold MAE pond. |
|---|---|---|---|---|
| bin_probit | 0,998 | 0,135 | 25 | 1,160 |
| wls_elogit | 0,999 | 0,136 | 26 | 1,151 |
| bin_logit (= quasi_logit) | 1,000 | 0,136 | 25 | 1,152 |
| quasi_spline | 1,012 | 0,136 | 29 | 1,157 |
| ols_elogit | 1,030 | 0,147 | 26 | 1,155 |
| betareg | 1,034 | 0,151 | 26 | 1,154 |
| **bb_selecao** | **1,035** | 0,156 | **19** | 1,150 |
| bb_logit | 1,036 | 0,152 | 26 | 1,159 |
| quasi_reglag | 1,050 | 0,137 | 30 | 1,148 |

- **Escolhido: `bb_selecao`** (beta-binomial, logit; 11 covariáveis contínuas além da defasagem, selecionadas por BIC, com Nordeste e Norte fundidos na média; 19 parâmetros, ver [03](03-modelos-candidatos.md), Parte 8). **Vice: `bin_probit`** (melhor erro médio).
- Limite de 1 EP: 0,998 + 0,135 = 1,133 p.p. **Treze modelos** ficam dentro dele. A diferença entre eles é menor que o ruído: a escolha segue a parcimônia e **não** indica superioridade preditiva.
- Ficam claramente fora: `l1_logit` (2,04 p.p.; encolhimento excessivo), `bb_offset` e `quasi_offset` (1,30-1,33; fixar o coeficiente da defasagem em 1 piora), `bin_cloglog` (1,20), `bb_reglag` (1,17).
- **Limite de aceitação das extensões.** Nos notebooks 03 e 05, uma extensão do modelo escolhido só é aceita se o MAE por UF no LOSO não passa de L = MAE por UF **aninhado** de `bb_selecao` + EP do melhor modelo LOSO = 1,1168 + 0,1347 = **1,2515 p.p.** (1,1695 com o 1,0348 sem aninhar; 1,1308 no LOSO que vazava o 2º turno). A tolerância é a do 1 EP da escolha original. Ver a seção 8.
- **Ressalva sobre o alvo.** Nas métricas fora da regra, o binomial vence onde o alvo é a soma das UFs: erro médio do share regional 0,10 p.p. (`bin_logit`) contra 0,39 p.p. (`bb_selecao`); deviance LOSO 160,5 mil contra 161,3 mil. O beta-binomial dá menos peso às grandes cidades, que dominam o share regional. Quem priorizar o share regional deve preferir `bin_probit` ou `bin_logit`. A regra fixada manda `bb_selecao`.
- **5-fold.** `glmm_uf` (1,026) e `quasi_uf` (1,034) lideram porque têm efeito de UF, que só ajuda em UFs já vistas; em LOSO o `glmm_uf` fica em 1,067. Isso confirma que o efeito de UF não generaliza para UF inédita.
- **Calibração** (`results/figures/cv_calibracao.html`): em 20 faixas de π̂, o maior desvio entre π̂ médio e share observado é 0,63 p.p. (`bb_selecao`) e 0,90 p.p. (`bin_probit`); `l1_logit`, usado como contraste, chega a 3,3 p.p.
- Figuras: `cv_loso_mae_uf.html`, `cv_kfold_mae.html`.

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
| Centro-Oeste | 4 | 0,42 | 39,99 | 40,04 (+0,05) | 40,03 (+0,03) | 40,38 (+0,39) |
| Nordeste | 9 | 0,25 | 69,37 | 69,42 (+0,05) | 69,42 (+0,06) | 69,36 (−0,01) |
| Norte | 7 | 0,48 | 50,01 | 50,14 (+0,13) | 50,16 (+0,15) | 49,44 (−0,56) |
| Sudeste | 4 | 0,51 | 47,24 | 47,47 (+0,23) | 47,48 (+0,24) | 47,65 (+0,40) |
| Sul | 3 | 0,38 | 37,42 | 37,39 (−0,03) | 37,40 (−0,03) | 36,86 (−0,56) |
| Brasil | 27 | 0,21 | 51,78 | 51,73 (−0,05) | 51,73 (−0,04) | 51,67 (−0,11) |

| Métrica (p.p.) | Probit | Logit | `bb_selecao` |
|---|---|---|---|
| MAE por UF (LOSO) | 0,998 | 1,000 | 1,035 |
| MAE regional (LOSO) | 0,097 | 0,101 | 0,387 |

- O cancelamento não é exato onde um estado domina a região (SP no Sudeste, w_u = 0,51): ali o erro do logit é +0,24 p.p., não zero.
- A diferença entre os três modelos vai de 0,07 p.p. (Nordeste) a 0,70 p.p. (Norte) por região e é de 0,06 p.p. no Brasil. Os três erram menos de 0,57 p.p. em qualquer região.
- **A escolha do modelo importa pouco para a resposta regional**: muda o share em até 0,7 p.p., menos que a largura do IC95% regional (cerca de 2,8 p.p. no modelo final). Importa só onde a região é próxima de 50%: o Norte vai de 49,44% (`bb_selecao`) a 50,16% (logit).
- O custo de `bb_selecao` em erro por UF (1,035 contra 0,998 do probit) está dentro de 1 erro-padrão, razão pela qual a regra o escolhe por ter menos parâmetros (19 contra 25).

## 7. O que a Parte 11 permite concluir

- A previsão de um estado inédito erra cerca de 1 p.p. no share da UF, com qualquer das famílias empatadas, e cerca de 0,1 a 0,4 p.p. no share regional.
- O erro fora da amostra (cerca de 1,2 a 1,3 p.p. municipal ponderado) é próximo do de dentro da amostra (0,9-1,4); o que o ajuste dentro da amostra esconde é o erro de nível de UF, de cerca de 1 p.p.
- Não permite concluir que o beta-binomial prevê melhor que o binomial.

## 8. Seleção aninhada do `bb_selecao`

A escolha das 11 contínuas e a fusão Nordeste + Norte usaram o 2º turno de 2026 de todas as UFs, então o 1,0348 acima é otimista. **Regra (fixada antes de rodar).** Em cada uma das 27 dobras LOSO, só com as 26 UFs de treino: testa-se a fusão de regiões e a de blocos de prefeito por razão de verossimilhanças (funde se p > 0,05); o passo a passo por BIC (backward, forward e both) fica com o conjunto de menor BIC; ajusta-se e prevê-se a UF retida com n̂. O MAE por UF dessas previsões substitui o do `bb_selecao` na regra de 1 EP e em L (`results/cv/loso_bb_selecao_aninhada.parquet`, `results/tables/cv_loso_aninhado.csv`). Os outros modelos não selecionam variáveis e ficam como estão.

- **Resultado.** MAE por UF aninhado 1,1168 p.p. (EP 0,162) contra 1,0348 (EP 0,156): +0,082 p.p., cerca de metade de um EP (0,51). Limite de 1 EP: 0,998 + 0,135 = 1,133; `bb_selecao` segue dentro dele (folga de 0,016 p.p.) e é o de menor k (19): **a escolha não muda**.
- **Estabilidade por dobra.** 10 a 13 contínuas (mediana 11); só 9 das 27 dobras repetem o conjunto do registro. Nordeste + Norte fundidos em 17 dobras e, em 7, também com o Sudeste (referência); GO, MA e BA têm fusões próprias ou nenhuma (1 dobra cada). Blocos: esquerda e centro + sem prefeito fundidos em 16 dobras, só esquerda em 10 e esquerda e sem prefeito em 1. Maiores erros: AP -3,17, DF -2,59, GO +2,45.
- **Fusão na referência.** A fusão de região na região de referência (Sudeste) agora é mantida dentro de cada dobra; antes era descartada (erro corrigido), e por isso os números acima mudaram.
- **Ressalva.** As extensões (`bb_disp`, `bb_slx`) seguem com a seleção dos dados completos; contra um L aninhado o limite fica mais folgado do que numa comparação em pé de igualdade. Nenhuma decisão está perto do limite (o mais próximo, `bb_slx`, 1,078 contra 1,2515).
- **Ordem de execução.** O notebook 03 grava `cv_loso_aninhado.csv` e o 05 o lê; o 05 grava `final_decisao.json` e o Passo 6 do 02 o lê. Por isso a ordem é 02, 03, 05 e de novo 02.

## Referências

- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning*, 2ª ed. Springer (regra de 1 erro-padrão, cap. 7).
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8).
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B* 36(2).
- Gelman, A.; Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models*, 2ª ed. Chapman & Hall.
- Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics* 31(4) (beta-binomial).
