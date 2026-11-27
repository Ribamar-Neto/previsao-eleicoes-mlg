# Diagnóstico dos modelos binomiais (Parte 10 do `02-MLG`)

Parte 10 do `02-MLG`: Passos 1 a 5 e conclusão antiga nas seções 1 a 7; Passo 6, dos modelos escolhidos, na seção 8. Ordem de execução: o Passo 6 lê `results/tables/final_decisao.json`, escrito pela Parte 15 do mesmo notebook; o Passo 6 roda depois da Parte 15 (Run-All único, valores em memória); sem o json o Passo 6 só analisa o `bb_selecao` e usa o corte de Conley de 200 km, sem aviso. Tabelas em `results/tables/diag_*.csv`, figuras em `results/figures/diag_*.png`, funções gráficas em `src/graficos.py`.

**Objetivo.** Verificar os pressupostos de `bin_logit`, `quasi_reglag`, `bb_logit`, `bb_reglag`, `bb_spline` e `glmm_uf` antes da validação cruzada (Parte 11). Este documento não escolhe modelo.

**Pressupostos testados.**
1. Distribuição da resposta (binomial ou beta-binomial com θ constante).
2. Nenhum município domina o ajuste.
3. Preditor linear e variância bem especificados.
4. Independência entre municípios.

## 1. Resíduos quantílicos aleatorizados

- **Função:** `G.residuos_quantilicos_rapido(y, n, pi, theta, rng)`, equivalente a `modelos.residuos_quantilicos`. Sorteia u ~ U(F(y-1), F(y)) e devolve Φ⁻¹(u).
- **Parâmetros:** `theta=None` para binomial; `theta` do ajuste para beta-binomial; `rng=0` fixo (resultado reprodutível).
- **Pressuposto:** F é a distribuição ajustada. Se o modelo está certo, os resíduos são N(0,1) exatos, mesmo com resposta discreta (Dunn e Smyth, 1996). Resíduos de Pearson e de deviance não têm essa propriedade.
- **Decisão:** a FDA beta-binomial da scipy leva ~9 s por chamada para 5.570 municípios; no envelope seriam 50 chamadas. Usamos quadratura sobre a beta, com número de pontos crescente em n (a binomial vira um degrau estreito em p quando n é grande). Erro máximo da FDA ~2e-4; 99% dos resíduos diferem menos de 0,04 da versão exata, e as diferenças grandes ficam em |r| > 4. A versão rápida está só em `graficos_diagnostico.py` porque `src/` não foi alterado.
- **Para quase-binomial e GLMM** não há verossimilhança completa; usamos a binomial com o π̂ do modelo. O desvio padrão acima de 1 mede a sobredispersão que esses modelos absorvem só no erro-padrão (quase) ou no efeito de UF (GLMM).

| modelo | dp | assimetria | curtose em excesso | % \|r\|>2 | % \|r\|>3 |
|---|---|---|---|---|---|
| esperado N(0,1) | 1 | 0 | 0 | 4,55 | 0,27 |
| bin_logit | 1,66 | 0,04 | 1,52 | 19,0 | 7,0 |
| quasi_reglag | 1,63 | -0,06 | 1,63 | 17,6 | 6,9 |
| bb_logit | 0,98 | -0,18 | 2,24 | 4,4 | 0,68 |
| bb_reglag | 0,98 | -0,20 | 2,33 | 4,3 | 0,72 |
| bb_spline | 0,98 | -0,17 | 2,25 | 4,3 | 0,72 |
| glmm_uf | 1,53 | 0,10 | 1,58 | 16,3 | 6,1 |

**Conclusão.** Binomial, quase-binomial e GLMM não descrevem a dispersão (dp 1,53 a 1,66, menos que sem o 1º turno, mas bem acima de 1). Os beta-binomiais acertam a escala média, mas têm cauda pesada e assimetria negativa. A média do resíduo é ~0 em todos (GLMM: -0,06).

## 2. QQ-plot

- **Leitura:** inclinação > 1 é variância subestimada; curvatura nas caudas é cauda mais pesada que o modelo.
- **Conclusão:** coerente com a tabela acima: inclinação ~1,6 nos modelos sem θ; beta-binomial sobre a diagonal no centro, com caudas abertas.

## 3. Half-normal com envelope simulado

- **Método:** bootstrap paramétrico. Simula y sob o modelo ajustado (beta-binomial: p ~ Beta, depois binomial), reajusta o modelo inteiro, ordena |r|, e toma os quantis 2,5% e 97,5% ponto a ponto (Atkinson, 1985; Moral et al., 2017). O reajuste leva em conta que os parâmetros foram estimados.
- **Parâmetros:** `bin_logit` B = 99 (reajuste de 0,04 s); `bb_reglag` B = 49, porque cada reajuste beta-binomial leva ~4 s (envelope em ~6 min). Com B = 49 a banda é mais ruidosa (extremos = 2º menor e 2º maior valor).
- **Resultado:**

| modelo | B | % fora | % acima | % abaixo |
|---|---|---|---|---|
| bin_logit | 99 | 99,4 | 99,4 | 0 |
| bb_reglag | 49 | 95,0 | 2,37 | 92,7 |

- **Conclusão.** `bin_logit` está fora da banda em toda a extensão (razão máxima 2,10 entre observado e limite superior). Em `bb_reglag` o corpo está abaixo da banda (resíduos pequenos demais) e as caudas acima (2,37%). Esse padrão é o de uma mistura de dispersões: [deduced] θ não é constante. O dp do resíduo por região apoia isso (`diag_dp_regiao.csv`, `bb_reglag`: Sul 0,75; Sudeste 0,89; Centro-Oeste 0,92; Norte 0,94; Nordeste 1,20). Pontos fora de banda ponto a ponto não são p-valores; mesmo com o modelo certo, alguns por cento saem.

## 4. Alavanca e distância de Cook

- **Fórmulas:** h = diag(W^½ X (X'WX)⁻¹ X' W^½) com W = nπ̂(1-π̂); D = r_P² h / (φ̂ p (1-h)²), r_P resíduo de Pearson, φ̂ = 2,93 (`comparacao_modelos.csv`, `quasi_reglag`), p = número de colunas do modelo (Pregibon, 1981; Cook, 1977; Williams, 1987). Feito à mão porque `get_influence()` não trata a resposta de duas colunas com φ de Pearson própria.
- **Pressuposto:** o ajuste não depende de poucas observações. Referências 2p/N e 4/N sinalizam muitos municípios (488 com h > 2p/N e 373 com D > 4/N no `quasi_reglag`); num N desse tamanho servem só de triagem. A prova é o reajuste.
- **Cortes no gráfico de Cook:** linhas em 4/N (triagem: acima dela, em vermelho, vale olhar o município) e em D = 1 (regra clássica de Cook: muito influente). Eixo em log, porque Brasília e São Paulo (D > 1) comprimem os demais em zero na escala linear.
- **Resultado.** Top 10 por Cook: São Paulo (D = 2,70; h = 0,58), Salvador (0,90; h = 0,26), Campos dos Goytacazes (0,20), Juiz de Fora (0,15), Porto Alegre (0,15), Goiânia (0,14), Brasília (0,13; h = 0,69), Belo Horizonte (0,11), São Luís (0,08), Natal (0,08). Todos são capitais ou grandes cidades, porque W cresce com n; só São Paulo passa de D = 1 (Salvador chega perto).
- **Reajuste sem esses 10** (`diag_sem_influentes.csv`; a padronização é refeita no subconjunto, efeito desprezível):

| modelo | maior \|mudança\| | termos com \|mudança\| > 0,5 EP | trocas de sinal |
|---|---|---|---|
| quasi_reglag | 2,92 EP (`log_pop`) | 25 de 34 | 2 |
| bb_reglag | 0,29 EP (`log_dens`) | 0 de 35 | 0 |

- **Conclusão.** O quase-binomial depende das grandes cidades; o beta-binomial não, porque θ finito limita o peso de n grande. Isso favorece o beta-binomial para inferência sobre coeficientes; não diz nada sobre previsão.

## 5. Resíduos contra ajustado, `lag` e região

- **Funções:** `G.residuos_contra` (média do resíduo em 20 faixas de mesmo tamanho, em vermelho) e `G.caixa_regiao`.
- **Pressuposto:** média 0 e dispersão 1 em qualquer faixa de η̂, de `lag` e em cada região.
- **Resultado:** Spearman do resíduo com η̂ e com `lag` entre 0,029 e 0,090 em todos os modelos; correlação de |r| com η̂ entre 0,124 e 0,153 (a dispersão cresce levemente com a proporção ajustada). Médias regionais dos beta-binomiais ≤ 0,05 em módulo; em `bin_logit` chegam a -0,14 (Sudeste) e no GLMM a -0,14 (Centro-Oeste).
- **Conclusão.** Não há tendência na média: o preditor linear (incluindo `bb_spline` e `bb_reglag`) não deixa forma funcional óbvia. O problema está na variância, não na média, e `bb_spline`/`bb_reglag` não melhoram os resíduos frente a `bb_logit`. A heterogeneidade regional de dispersão (seção 3) aparece de novo no box plot.

## 6. I de Moran dos resíduos

- **Função:** `esda.Moran(r, w, permutations=999)`. Pesos: rainha, padronizados por linha, ilhas ligadas ao vizinho mais próximo (mesma receita da `01-AED`; 0 ilhas restantes, 5,79 vizinhos em média). Mapa de `bb_reglag` em PNG, geometria simplificada a 0,05°.
- **Pressuposto:** resíduos independentes. Se o preditor absorve a estrutura espacial, I ≈ 0.
- **Resultado** (`diag_moran.csv`; p_sim = 0,001 em todos, o mínimo com 999 permutações):

| modelo | I | z_sim |
|---|---|---|
| bin_logit | 0,319 | 39,5 |
| quasi_reglag | 0,332 | 40,8 |
| bb_logit | 0,292 | 37,1 |
| bb_reglag | 0,306 | 38,7 |
| bb_spline | 0,299 | 38,0 |
| glmm_uf | 0,239 | 29,3 |
| bin_logit, resíduo de Pearson | 0,298 | 37,4 |

- **Conclusão.** Há forte autocorrelação espacial residual em todos. A EDA tinha I = 0,90 na proporção observada e 0,54 no *swing*; os preditores explicam a maior parte, mas não tudo. A informação efetiva é menor que N: **os erros-padrão e intervalos de todos os modelos continuam pequenos demais**, mesmo no beta-binomial, e valores-p de coeficientes não devem ser lidos literalmente. A estimativa pontual de π não fica necessariamente enviesada. A UF aleatória é o que mais reduz I (0,239), pouco. A checagem honesta de desempenho é a validação deixando uma UF de fora (Parte 11); ela deixa vizinhos de fronteira no treino, então ainda é levemente otimista.

## 7. O que se pode concluir

| modelo | distribuição | preditor | influentes | independência |
|---|---|---|---|---|
| bin_logit | falha | ok | não avaliado | falha |
| quasi_reglag | falha na forma | ok | falha (25 de 34 termos > 0,5 EP) | falha |
| bb_logit, bb_spline | escala ok; formato e θ constante falham | ok | ok (irmão `bb_reglag`) | falha |
| bb_reglag | escala ok; formato e θ constante falham | ok | ok | falha |
| glmm_uf | falha | ok | não avaliado | falha (menor I) |

**Decisões registradas.**
- Nenhum modelo é escolhido aqui: os diagnósticos medem pressupostos, e a escolha é por validação cruzada (Parte 11).
- O beta-binomial é o único com resíduos de escala correta e coeficientes estáveis; o que falta nele é dispersão por região e dependência espacial.
- Remédios candidatos, não implementados: θ dependente da região; termo espacial; erros-padrão robustos com agrupamento por UF.
- Limitação: o envelope de `bb_reglag` usa B = 49 por custo; os resíduos são aleatorizados (uma realização, `rng=0`), então as contagens por faixa variam na segunda casa decimal entre sementes.

*A tabela acima cobre só os modelos antigos; `bb_selecao` e `bb_slx` estão na seção 8.*

## 8. Pressupostos dos modelos escolhidos (Passo 6)

`bb_selecao` (menor BIC pela regra de 1 EP; 24 parâmetros) e `bb_slx` (modelo final, θᵢ por município; 46 parâmetros) passam pela bateria completa. Tabelas `results/tables/diag_final_*.csv`; figuras `results/figures/diag_final_*.png`.

**Referências do método:** ligação (Pregibon, 1980; Hinkley, 1985); GVIF (Fox e Monette, 1992); número de condição (Belsley, Kuh e Welsch, 1980).

**Decisões.** As 7 regras abaixo foram fixadas antes dos resultados. Os testes são diagnóstico: nenhum modelo muda, e os remédios vão para [12-pendencias.md](12-pendencias.md). Inferência dos testes com covariância de Conley a 200 km (corte de [08](08-dependencia-espacial.md)), porque os resíduos são espacialmente dependentes (8.5).

**Os φ e θ daqui não são os de 3,07 e 7,6 mil.** Aqui, φ de Pearson é o do π̂ de cada beta-binomial lido como binomial, dividido por N − p: 4,08 (`bb_selecao`, `comparacao_modelos.csv`; o do `bb_slx` não é reproduzível das saídas atuais e fica fora); θ é a mediana dos θᵢ: 7,53 mil para `bb_selecao` (θ constante) e 12,6 mil para `bb_slx` (a mediana cai no Sudeste, `final_ep.csv`). Os 3,07 e 7,6 mil de [01](01-visao-geral.md) e [03](03-modelos-candidatos.md) são de outros modelos: φ do `bin_logit` e θ constante do `bb_logit` (7,63 mil). Não há conflito; só não são comparáveis.

| # | Regra fixada | Veredito `bb_selecao` / `bb_slx` |
|---|---|---|
| 1 | Ligação: η̂² padronizado como covariável extra, Wald com Conley; sinaliza se p < 0,05; efeito = P95 e máx. de \|Δπ̂\| | sinaliza nos dois (p Conley 1e-4 e 5e-3) |
| 2 | Linearidade, uma covariável por vez (resíduo parcial em 20 faixas e spline natural de 4 gl): material se p < 0,05 **e** P95 de \|Δπ̂\| ≥ 0,5 p.p. | nenhuma falha |
| 3 | Distribuição: resíduos quantílicos (rng = 0), QQ, envelope B = 49 | escala ok; formato falha |
| 4 | Variância: DP por região e por quintil de n; Spearman de \|r\| com η̂ e log n | `bb_slx` ok; `bb_selecao` heterogêneo |
| 5 | Independência: I de Moran, 999 permutações | falha nos dois |
| 6 | Colinearidade da matriz de média do `bb_slx`: VIF > 10 (fatores: GVIF^(1/(2gl)) > √10) e número de condição | 11 de 38 fatores acima de 10 |
| 7 | Influência: Cook e DFBETA de um passo; reajuste sem os 10 maiores, mudança em EP de Conley | só troca de sinal reprova; nenhuma (0 / 0); maior mudança 1,37 / 1,05 EP |

### 8.1 Ligação

Coeficiente de η̂²: 0,0041 (p Conley 1,3e-4) / 0,0029 (p Conley 4,6e-3). P95 de \|Δπ̂\| = 0,13 / 0,09 p.p.; máximo 0,25 / 0,19 p.p.; ganho de log-verossimilhança 23,5 / 11,5. **Sinalizada nos dois**, pela regra. O efeito é pequeno (abaixo de 0,5 p.p.); remédio possível, não aplicado: ligação paramétrica (pendência).

### 8.2 Linearidade

18 covariáveis contínuas por modelo (`lag`, `lag1t` e as 16 demais selecionadas). Com p Conley < 0,05: 14 (`bb_selecao`) e 11 (`bb_slx`); `nao_linear` em nenhuma. Material pela regra: nenhuma; o maior P95 é 0,21 p.p. (`cobertura_pbf`, `bb_selecao`) e 0,18 (`cobertura_pbf`, `bb_slx`), bem abaixo do corte de 0,5. O p-valor sozinho não decide: com N = 5.570 detecta curvaturas sem importância prática, por isso o critério de tamanho.

### 8.3 Distribuição

| modelo | dp | assimetria | curtose em excesso | % \|r\|>2 | % \|r\|>3 |
|---|---|---|---|---|---|
| `bb_reglag` (referência, seção 1) | 0,98 | -0,20 | 2,33 | 4,29 | 0,72 |
| `bb_selecao` | 0,98 | -0,17 | 2,27 | 4,31 | 0,70 |
| `bb_slx` | 0,97 | -0,08 | 2,25 | 3,82 | 0,47 |

Envelope (B = 49), % de pontos fora / acima / abaixo da banda:

| modelo | fora | acima | abaixo |
|---|---|---|---|
| `bb_reglag` | 95,0 | 2,37 | 92,7 |
| `bb_selecao` | 94,6 | 2,44 | 92,2 |
| `bb_slx` | 72,8 | 1,06 | 71,7 |

A dispersão por região não muda a curtose (2,27 para 2,25) e reduz a assimetria (-0,17 para -0,08); o padrão "corpo abaixo da banda, caudas acima" persiste. O envelope não valida a distribuição (pendência). A escala está certa (dp ≈ 1).

### 8.4 Variância

`bb_selecao`: DP por região de 0,75 (Sul) a 1,19 (Nordeste); por quintil de n, Q5 = 0,92. Spearman de \|r\| com η̂ = 0,146 e com log n = 0,045. `bb_slx`: DP por região de 0,92 a 1,02; Q5 = 0,97; Spearman 0,012 e 0,030. Conclusão: a dispersão por região ([07](07-dispersao-regional.md)) resolve a heterogeneidade; `log n` não entra (p = 0,278).

### 8.5 Independência espacial

I de Moran = 0,298 (`bb_selecao`) e 0,270 (`bb_slx`), z simulado 38,0 e 34,1, p simulado 0,001 (o mínimo com 999 permutações). Falha nos dois; o SLX não remove a autocorrelação ([08](08-dependencia-espacial.md)). A correlação residual é tratada na inferência pelo erro-padrão de Conley, não aqui.

### 8.6 Colinearidade do `bb_slx`

Número de condição de Belsley 82,4 (41 colunas). Onze de 38 fatores acima de 10: `w_lag1t` 251,7; `w_lag` 228,7; `lag1t` 114,4; `lag` 110,5; `w_pct_pretos_pardos` 23,7; `pct_pretos_pardos` 18,4; `w_cobertura_pbf` 17,6; `w_pct_fund_inc` 14,3; `w_log_dens` 11,7; `pct_fund_inc` 10,2; `log_dens` 10,0. Região: GVIF 15,0 com 3 gl (Nordeste fundido no Sudeste), ajustado 1,57 (limite 3,16); bloco 1,10 ajustado. Razão: `lag` e `lag1t` medem quase o mesmo nível do PT, e cada covariável compartilha informação com a própria defasagem espacial (VIF médio 34,0 nas `w_*` contra 16,9 nas próprias). **Efeito:** a leitura de coeficientes individuais desses termos é frágil; previsão e Wald conjunto não são afetados, e o ganho do SLX foi medido por validação cruzada, não por coeficientes.

### 8.7 Influência

Os 10 maiores Cook de um passo são em geral municípios pequenos: `bb_selecao` lidera Belo Monte/AL (0,069), Santo Antônio de Lisboa/PI (0,057), Caldeirão Grande/BA (0,035); `bb_slx`, Belo Monte/AL (0,067), Santo Antônio de Lisboa/PI (0,043), Jutaí/AM (0,027). Reajuste sem os 10: maior mudança 1,37 EP no `bb_selecao` (`log_theta`) e 1,05 no `bb_slx` (`disp_reg_Nordeste`); 1 de 24 e 6 de 46 termos acima de 0,5 EP; 0 trocas de sinal. **Regra 7:** só a troca de sinal de um coeficiente reprova; uma mudança de 1,4 EP sem troca de sinal não reprova. Contraste com o quase-binomial (seção 4: 25 de 34 acima de 0,5 EP, 2 trocas de sinal).

### 8.8 Conclusão

Passa: escala da variância, preditor linear (nenhuma covariável material), sem troca de sinal frente a influentes (a mudança de até 1,37 EP em `log_theta` fica registrada). Falha: ligação (efeito pequeno, sinalizada nos dois), formato da distribuição (cauda pesada, envelope), independência espacial. Fragilidade de leitura: coeficientes de `lag` e `lag1t` e do `bb_slx` por colinearidade. Pendências em [12](12-pendencias.md).

## Referências

- Atkinson, A. C. (1985). *Plots, Transformations and Regression*. Oxford University Press.
- Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15-18.
- Belsley, D. A.; Kuh, E.; Welsch, R. E. (1980). *Regression Diagnostics*. Wiley.
- Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
- Fox, J.; Monette, G. (1992). Generalized collinearity diagnostics. *Journal of the American Statistical Association*, 87(417), 178-183.
- Hinkley, D. V. (1985). Transformation diagnostics for linear models. *Biometrika*, 72(3), 487-496.
- Moral, R. A.; Hinde, J.; Demétrio, C. G. B. (2017). Half-normal plots and overdispersed models in R: the hnp package. *Journal of Statistical Software*, 81(10), 1-23.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
- Pregibon, D. (1980). Goodness of link tests for generalized linear models. *Applied Statistics*, 29(1), 15-24.
- Pregibon, D. (1981). Logistic regression diagnostics. *The Annals of Statistics*, 9(4), 705-724.
- Williams, D. A. (1987). Generalized linear model diagnostics using the deviance and single case deletions. *Applied Statistics (JRSS C)*, 36(2), 181-191.
