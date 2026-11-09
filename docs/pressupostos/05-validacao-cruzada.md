# 05 - Validação cruzada e escolha do modelo (Parte 11)

Notebook: `notebooks/03-validacao.ipynb`, células 0 a 10 (contagem a partir de 0). Código: `src/modelos.py` (`folds_kfold`, `folds_por_grupo`, `metricas`, `agregar`)
e `src/especificacoes.py` (`prever_fold`, `parametros_ajuste`).

## 1. Por que validar fora da amostra (células 0-1)

- **Decisão.** O modelo e as estimativas regionais saem de previsão fora da amostra.
- **Razão.** Um binomial com ligação canônica e dummies de região reproduz os totais regionais *exatamente* dentro da amostra (equações de escore; notebook 02, célula 3).
  Esse acerto é por construção e não mede capacidade de previsão. As métricas dentro da amostra (MAE ponderado de 0,92 a 1,37 p.p.) também não decidem entre famílias:
  o beta-binomial dá menos peso às grandes cidades que o quasi-binomial, então os dois otimizam critérios diferentes.
- **Conclusão permitida.** Só a previsão de dados retidos compara famílias diferentes na mesma régua.

## 2. Esquemas de validação (células 1, 4)

| Esquema | Função | Parâmetros | Pressuposto | Para que serve |
|---|---|---|---|---|
| 5-fold aleatório | `folds_kfold(n, k=5, seed=0)` | 5 partes, semente 0 | municípios intercambiáveis | erro de município com vizinhos no treino (otimista) |
| Leave-one-UF-out (LOSO) | `folds_por_grupo(uf)` | 27 dobras, uma UF por vez | UF retida é "inédita" | erro de prever um estado novo |

- **Razão do LOSO.** A autocorrelação espacial é forte (I de Moran do share 2026 = 0,896): os municípios não são independentes. No 5-fold, os vizinhos do município de teste estão no treino
  e o erro sai menor que o de uma previsão real. O LOSO remove esses vizinhos.
- **Modelos com efeito fixo de UF** (`quasi_uf`, `bb_uf`) entram só no 5-fold: a UF retida não tem efeito próprio no treino, e prevê-la exigiria inventar um valor.
  O `glmm_uf` (efeito aleatório de UF) entra nos dois: para UF inédita o efeito aleatório vale zero.
- **Padronização.** Média e desvio das covariáveis são recalculados em cada treino (`Candidato.ajustar` ajusta o `Padronizador` ao treino), sem vazamento do teste.
- **Duplicata.** `quasi_logit` produz o mesmo π̂ de `bin_logit` (mesmo estimador de β; o quasi só altera a variância). Foi avaliado uma vez. Restam 20 candidatos distintos (18 em LOSO).
- **Cache.** Previsões fora da amostra ficam em `results/cv/*.parquet`; `RECALCULAR = True` (topo da célula 2) refaz tudo. Execução completa: cerca de 5 min com 14 processos; com cache, cerca de 40 s.

## 3. Métricas (células 5-6)

Calculadas sobre as previsões fora da amostra empilhadas (cada município é previsto uma vez):

- `mae_pond_pp`, `rmse_pond_pp`: erro do share municipal em p.p., ponderado por votos válidos. `mae_pp`: sem ponderar. `deviance`: deviance binomial. `erro_nacional_pp`: viés do total nacional.
- `mae_uf_pp`: média, sobre as 27 UFs, do |share previsto da UF − observado|, com share ponderado por votos. `mae_reg_pp`: o mesmo para as 5 regiões.
- **Erro-padrão (EP).** LOSO: desvio dos 27 erros por UF / √27. 5-fold: desvio do MAE ponderado dos 5 folds / √5.
- **Pressuposto do EP.** Trata as UFs (ou folds) como repetições independentes. UFs vizinhas se parecem, então o EP é um pouco otimista; serve como régua de ruído, não como teste formal.

## 4. Regra de decisão (célula 1, fixada antes dos resultados)

1. **Primária**: `mae_uf_pp` em LOSO. Razão: imita prever um estado inédito, e o alvo são shares regionais, somas de UFs.
2. **1 erro-padrão**: modelos até (melhor + EP) empatam; vence o de menos parâmetros (`k` do ajuste a todos os dados; beta-binomial conta `log_theta`, quasi conta φ).
3. **Secundária**: MAE ponderado no 5-fold, para desempate com `k` igual.
4. **Relatório**: modelo escolhido e vice (o melhor por erro médio, se for outro).

## 5. Resultados (células 6, 8, 9; tabelas `results/tables/cv_loso.csv`, `cv_kfold.csv`, `cv_uf_loso.csv`)

LOSO, melhores por erro médio da UF (p.p.), EP, `k`, MAE ponderado municipal no 5-fold:

| Modelo | LOSO UF-MAE | EP | k | 5-fold MAE pond. |
|---|---|---|---|---|
| bin_probit | 0,930 | 0,130 | 25 | 1,151 |
| wls_elogit | 0,939 | 0,130 | 26 | 1,145 |
| bin_logit (= quasi_logit) | 0,943 | 0,130 | 25 | 1,145 |
| quasi_spline | 0,955 | 0,130 | 29 | 1,153 |
| ols_elogit | 0,969 | 0,138 | 26 | 1,167 |
| betareg | 0,975 | 0,140 | 26 | 1,166 |
| bb_logit | 0,985 | 0,139 | 26 | 1,170 |
| quasi_reglag | 0,996 | 0,131 | 30 | 1,145 |
| **bb_backward** | **1,000** | 0,142 | **20** | 1,159 |

- **Escolhido: `bb_backward`** (beta-binomial, logit, 11 covariáveis além da defasagem, selecionadas por BIC; 20 parâmetros). **Vice: `bin_probit`** (melhor erro médio).
- Doze modelos ficam a menos de 1 EP (1,061 p.p.) do melhor. A diferença entre eles é menor que o ruído: a escolha segue a parcimônia e **não** indica superioridade preditiva.
- Ficam claramente fora: `l1_logit` (2,0 p.p.; encolhimento excessivo), `bb_offset` e `quasi_offset` (1,25-1,27; fixar o coeficiente da defasagem em 1 piora), `bin_cloglog`, `bin_loglog`, `bb_reglag`.
- **Ressalva sobre o alvo.** Nas métricas fora da regra, o binomial vence onde o alvo é a soma das UFs: erro médio do share regional 0,08 p.p. (`bin_logit`) contra 0,36 p.p. (`bb_backward`); deviance LOSO 156 mil contra 163 mil.
  O beta-binomial dá menos peso às grandes cidades, que dominam o share regional. Quem priorizar o share regional deve preferir `bin_probit` ou `bin_logit`. A regra fixada manda `bb_backward`.
- **5-fold.** `glmm_uf` (1,033) e `quasi_uf` (1,043) lideram porque têm efeito de UF, que só ajuda em UFs já vistas; em LOSO o `glmm_uf` fica em 1,044. Isso confirma que o efeito de UF não generaliza para UF inédita.
- **Calibração** (célula 9, `results/figures/cv_calibracao.html`): em 20 faixas de π̂, o maior desvio entre π̂ médio e share observado é 0,73 p.p. (`bb_backward`) e 0,87 p.p. (`bin_probit`); `l1_logit`, usado como contraste, chega a 3,3 p.p.
- Figuras: `cv_loso_mae_uf.html`, `cv_kfold_mae.html`.

## 6. O que a Parte 11 permite concluir

- A previsão de um estado inédito erra cerca de 1 p.p. no share da UF, com qualquer das famílias empatadas, e cerca de 0,1 a 0,4 p.p. no share regional.
- O erro fora da amostra (cerca de 1,2 p.p. municipal ponderado) é próximo do de dentro da amostra (0,9-1,4); o que o ajuste dentro da amostra esconde é o erro de nível de UF, de cerca de 1 p.p.
- Não permite concluir que o beta-binomial prevê melhor que o binomial.

## Referências

- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning*, 2ª ed. Springer (regra de 1 erro-padrão, cap. 7).
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8).
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B* 36(2).
- McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models*, 2ª ed. Chapman & Hall.
- Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics* 31(4) (beta-binomial).
