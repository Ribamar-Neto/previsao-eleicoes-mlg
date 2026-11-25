# Diagnóstico dos modelos binomiais (Parte 10 do `02-MLG`)

Parte 10 do `02-MLG`: Passos 1 a 5 e conclusão antiga nas seções 1 a 7; Passo 6, dos modelos escolhidos, na seção 8. Ordem de execução: o Passo 6 lê `results/tables/final_decisao.json`, escrito pela Parte 15 do mesmo notebook; o Passo 6 roda depois da Parte 15 (Run-All único, valores em memória); sem o json o Passo 6 só analisa o `bb_selecao` e usa o corte de Conley de 200 km, sem aviso. Tabelas em `results/tables/diag_*.csv`, figuras em `results/figures/diag_*.png`, funções gráficas em `notebooks/graficos_diagnostico.py`.

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
| bin_logit | 3,42 | -0,03 | -0,60 | 54,4 | 38,7 |
| quasi_reglag | 3,42 | -0,04 | -0,60 | 54,8 | 38,9 |
| bb_logit | 1,03 | -0,51 | 3,78 | 4,9 | 0,97 |
| bb_reglag | 1,03 | -0,49 | 3,67 | 4,8 | 0,97 |
| bb_spline | 1,03 | -0,52 | 3,72 | 4,9 | 0,95 |
| glmm_uf | 3,20 | 0,04 | -0,39 | 50,7 | 34,4 |

**Conclusão.** Binomial, quase-binomial e GLMM não descrevem a dispersão (dp 3,20 a 3,42). Os beta-binomiais acertam a escala média, mas têm cauda pesada e assimetria negativa. A média do resíduo é ~0 em todos (GLMM: -0,11).

## 2. QQ-plot

- **Leitura:** inclinação > 1 é variância subestimada; curvatura nas caudas é cauda mais pesada que o modelo.
- **Conclusão:** coerente com a tabela acima: inclinação ~3,4 nos modelos sem θ; beta-binomial sobre a diagonal no centro, com caudas abertas.

## 3. Half-normal com envelope simulado

- **Método:** bootstrap paramétrico. Simula y sob o modelo ajustado (beta-binomial: p ~ Beta, depois binomial), reajusta o modelo inteiro, ordena |r|, e toma os quantis 2,5% e 97,5% ponto a ponto (Atkinson, 1985; Moral et al., 2017). O reajuste leva em conta que os parâmetros foram estimados.
- **Parâmetros:** `bin_logit` B = 99 (reajuste de 0,04 s); `bb_reglag` B = 49, porque cada reajuste beta-binomial leva ~4 s (envelope em ~6 min). Com B = 49 a banda é mais ruidosa (extremos = 2º menor e 2º maior valor).
- **Resultado:**

| modelo | B | % fora | % acima | % abaixo |
|---|---|---|---|---|
| bin_logit | 99 | 100 | 100 | 0 |
| bb_reglag | 49 | 92,4 | 4,38 | 88,0 |

- **Conclusão.** `bin_logit` está fora da banda em toda a extensão (razão máxima 4,25 entre observado e limite superior). Em `bb_reglag` o corpo está abaixo da banda (resíduos pequenos demais) e as caudas acima (4,38%). Esse padrão é o de uma mistura de dispersões: [deduced] θ não é constante. O dp do resíduo por região apoia isso (`diag_dp_regiao.csv`: Sudeste 0,81; Sul 0,90; Centro-Oeste 0,92; Nordeste 1,21; Norte 1,39). Pontos fora de banda ponto a ponto não são p-valores; mesmo com o modelo certo, alguns por cento saem.

## 4. Alavanca e distância de Cook

- **Fórmulas:** h = diag(W^½ X (X'WX)⁻¹ X' W^½) com W = nπ̂(1-π̂); D = r_P² h / (φ̂ p (1-h)²), r_P resíduo de Pearson, φ̂ = 21,56, p = 29 (Pregibon, 1981; Cook, 1977; Williams, 1987). Feito à mão porque `get_influence()` não trata a resposta de duas colunas com φ de Pearson própria.
- **Pressuposto:** o ajuste não depende de poucas observações. Referências 2p/N e 4/N sinalizam 483 e 419 municípios; num N desse tamanho servem só de triagem. A prova é o reajuste.
- **Resultado.** Top 10 por Cook: Brasília (D = 6,15; h = 0,67), São Paulo (2,95; h = 0,57), Manaus (0,92), Fortaleza (0,49), Recife (0,39), Goiânia (0,25), Feira de Santana (0,10), Belém (0,09), Campos dos Goytacazes (0,07), Macapá (0,06). Todos são capitais ou grandes cidades, porque W cresce com n.
- **Reajuste sem esses 10** (`diag_sem_influentes.csv`; a padronização é refeita no subconjunto, efeito desprezível):

| modelo | maior \|mudança\| | termos com \|mudança\| > 0,5 EP | trocas de sinal |
|---|---|---|---|
| quasi_reglag | 9,51 EP (`bloco_outro`) | 25 de 32 | 2 |
| bb_reglag | 0,18 EP (`bloco_outro`) | 0 de 33 | 0 |

- **Conclusão.** O quase-binomial depende das grandes cidades; o beta-binomial não, porque θ finito limita o peso de n grande. Isso favorece o beta-binomial para inferência sobre coeficientes; não diz nada sobre previsão.

## 5. Resíduos contra ajustado, `lag` e região

- **Funções:** `G.residuos_contra` (média do resíduo em 20 faixas de mesmo tamanho, em vermelho) e `G.caixa_regiao`.
- **Pressuposto:** média 0 e dispersão 1 em qualquer faixa de η̂, de `lag` e em cada região.
- **Resultado:** Spearman do resíduo com η̂ e com `lag` entre 0,010 e 0,047 em todos os modelos; correlação de |r| com η̂ entre 0,092 e 0,114 (a dispersão cresce levemente com a proporção ajustada). Médias regionais dos beta-binomiais ≤ 0,01 em módulo; em `bin_logit` chegam a -0,45 (Norte) e no GLMM a -0,58 (Sul).
- **Conclusão.** Não há tendência na média: o preditor linear (incluindo `bb_spline` e `bb_reglag`) não deixa forma funcional óbvia. O problema está na variância, não na média, e `bb_spline`/`bb_reglag` não melhoram os resíduos frente a `bb_logit`. A heterogeneidade regional de dispersão (seção 3) aparece de novo no box plot.

## 6. I de Moran dos resíduos

- **Função:** `esda.Moran(r, w, permutations=999)`. Pesos: rainha, padronizados por linha, ilhas ligadas ao vizinho mais próximo (mesma receita da `01-AED`; 0 ilhas restantes, 5,79 vizinhos em média). Mapa de `bb_reglag` em PNG, geometria simplificada a 0,05°.
- **Pressuposto:** resíduos independentes. Se o preditor absorve a estrutura espacial, I ≈ 0.
- **Resultado** (`diag_moran.csv`; p_sim = 0,001 em todos, o mínimo com 999 permutações):

| modelo | I | z_sim |
|---|---|---|
| bin_logit | 0,367 | 46,7 |
| quasi_reglag | 0,371 | 47,2 |
| bb_logit | 0,337 | 42,6 |
| bb_reglag | 0,339 | 42,5 |
| bb_spline | 0,335 | 42,5 |
| glmm_uf | 0,295 | 38,2 |
| bin_logit, resíduo de Pearson | 0,357 | 45,6 |

- **Conclusão.** Há forte autocorrelação espacial residual em todos. A EDA tinha I = 0,90 na proporção observada e 0,54 no *swing*; os preditores explicam a maior parte, mas não tudo. A informação efetiva é menor que N: **os erros-padrão e intervalos de todos os modelos continuam pequenos demais**, mesmo no beta-binomial, e valores-p de coeficientes não devem ser lidos literalmente. A estimativa pontual de π não fica necessariamente enviesada. A UF aleatória é o que mais reduz I (0,295), pouco. A checagem honesta de desempenho é a validação deixando uma UF de fora (Parte 11); ela deixa vizinhos de fronteira no treino, então ainda é levemente otimista.

## 7. O que se pode concluir

| modelo | distribuição | preditor | influentes | independência |
|---|---|---|---|---|
| bin_logit | falha | ok | não avaliado | falha |
| quasi_reglag | falha na forma | ok | falha | falha |
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

`bb_selecao` (menor BIC pela regra de 1 EP; 19 parâmetros) e `bb_slx` (modelo final, θᵢ por município; 37 parâmetros) passam pela bateria completa. Tabelas `results/tables/diag_final_*.csv`; figuras `results/figures/diag_final_*.png`.

**Referências do método:** ligação (Pregibon, 1980; Hinkley, 1985); GVIF (Fox e Monette, 1992); número de condição (Belsley, Kuh e Welsch, 1980).

**Decisões.** As 7 regras abaixo foram fixadas antes dos resultados. Os testes são diagnóstico: nenhum modelo muda, e os remédios vão para [12-pendencias.md](12-pendencias.md). Inferência dos testes com covariância de Conley a 200 km (corte de [08](08-dependencia-espacial.md)), porque os resíduos são espacialmente dependentes (8.5).

**Os φ e θ daqui não são os de 21,24 e 651.** Aqui, φ de Pearson é o do π̂ de cada beta-binomial lido como binomial, dividido por N − p: 24,2 (`bb_selecao`, `comparacao_modelos.csv`; o do `bb_slx` não foi recalculado); θ é a mediana dos θᵢ: 648 para `bb_selecao` (θ constante; o do `bb_slx` não foi recalculado). Os 21,24 e 651 de [01](01-visao-geral.md) e [03](03-modelos-candidatos.md) são de outros modelos: φ do `bin_logit` e θ constante do `bb_logit`. Não há conflito; só não são comparáveis.

| # | Regra fixada | Veredito `bb_selecao` / `bb_slx` |
|---|---|---|
| 1 | Ligação: η̂² padronizado como covariável extra, Wald com Conley; sinaliza se p < 0,05; efeito = P95 e máx. de \|Δπ̂\| | `bb_selecao` não sinaliza (p = 0,095); `bb_slx` sinaliza |
| 2 | Linearidade, uma covariável por vez (resíduo parcial em 20 faixas e spline natural de 4 gl): material se p < 0,05 **e** P95 de \|Δπ̂\| ≥ 0,5 p.p. | só `pct_pretos_pardos` falha |
| 3 | Distribuição: resíduos quantílicos (rng = 0), QQ, envelope B = 49 | escala ok; formato falha |
| 4 | Variância: DP por região e por quintil de n; Spearman de \|r\| com η̂ e log n | `bb_slx` ok; `bb_selecao` heterogêneo |
| 5 | Independência: I de Moran, 999 permutações | falha nos dois |
| 6 | Colinearidade da matriz de média do `bb_slx`: VIF > 10 (fatores: GVIF^(1/(2gl)) > √10) e número de condição | 7 termos acima de 10 |
| 7 | Influência: Cook e DFBETA de um passo; reajuste sem os 10 maiores, mudança em EP de Conley | ok |

### 8.1 Ligação

Coeficiente de η̂²: -0,0054 (p Conley 0,095; p do modelo < 0,001) / -0,0076 (p Conley 0,007). P95 de \|Δπ̂\| = 0,17 / 0,25 p.p.; máximo 0,33 / 0,41 p.p.; ganho de log-verossimilhança 7,0 / 13,9. **Não sinalizada em `bb_selecao`; sinalizada em `bb_slx`**, pela regra. O efeito é pequeno (abaixo de 0,5 p.p.), mas sistemático no `bb_slx`; remédio possível, não aplicado: ligação paramétrica (pendência).

### 8.2 Linearidade

13 covariáveis contínuas por modelo (a defasagem e as 12 selecionadas). Com p Conley < 0,05: 5 (`bb_selecao`) e 9 (`bb_slx`); `nao_linear` só em `pct_pretos_pardos`, quase todas de efeito pequeno. Material pela regra: só `pct_pretos_pardos` (P95 0,64 / 0,63 p.p.; máximo 1,34 / 1,52 p.p.). O maior P95 entre as demais é 0,45 p.p. (`cobertura_pbf` e `pct_sem_religiao` em `bb_selecao`; `pct_sem_religiao` chega a 0,49 em `bb_slx`), abaixo do corte de 0,5. O p-valor sozinho não decide: com N = 5.570 detecta curvaturas sem importância prática, por isso o critério de tamanho.

### 8.3 Distribuição

| modelo | dp | assimetria | curtose em excesso | % \|r\|>2 | % \|r\|>3 |
|---|---|---|---|---|---|
| `bb_reglag` (referência, seção 1) | 1,03 | -0,49 | 3,67 | 4,79 | 0,97 |
| `bb_selecao` | 1,03 | -0,53 | 3,77 | 4,87 | 0,99 |
| `bb_slx` | 1,02 | -0,43 | 2,24 | 4,96 | 0,86 |

Envelope (B = 49), % de pontos fora / acima / abaixo da banda:

| modelo | fora | acima | abaixo |
|---|---|---|---|
| `bb_reglag` | 92,4 | 4,38 | 88,0 |
| `bb_selecao` | 92,7 | 4,90 | 87,8 |
| `bb_slx` | 74,6 | 5,58 | 69,0 |

A dispersão por região baixa a curtose (3,77 para 2,24), mas o padrão "corpo abaixo da banda, caudas acima" persiste. O envelope não valida a distribuição (pendência). A escala está certa (dp ≈ 1).

### 8.4 Variância

`bb_selecao`: DP por região de 0,82 (Sudeste) a 1,39 (Norte); por quintil de n, Q5 = 0,88. Spearman de \|r\| com η̂ = 0,106 e com log n = -0,081. `bb_slx`: DP por região de 0,99 a 1,07; Q5 = 0,99; Spearman 0,002 e -0,007. Conclusão: a dispersão por região e log n ([07](07-dispersao-regional.md)) resolve a heterogeneidade; a mesma correção explica por que a curtose cai.

### 8.5 Independência espacial

I de Moran = 0,334 (`bb_selecao`) e 0,348 (`bb_slx`), z simulado 42,3 e 43,8, p simulado 0,001 (o mínimo com 999 permutações). Falha nos dois; o SLX não remove a autocorrelação ([08](08-dependencia-espacial.md)). A correlação residual é tratada na inferência pelo erro-padrão de Conley, não aqui.

### 8.6 Colinearidade do `bb_slx`

Número de condição de Belsley 21,3 (31 colunas). Sete de 28 fatores acima de 10: `w_pct_pretos_pardos` 28,5; `pct_pretos_pardos` 24,0; `w_cobertura_pbf` 17,4; `w_lag` 16,5; `w_pct_16_24` 13,7; `lag` 11,7; `w_log_dens` 10,1. Região: GVIF 31,6 com 3 gl, ajustado 1,78 (limite 3,16); bloco 1,07 ajustado. Razão: cada covariável compartilha informação com a própria defasagem espacial (VIF médio 8,9 nas `w_*` contra 6,5 nas próprias). **Efeito:** a leitura de coeficientes individuais desses termos é frágil; previsão e Wald conjunto não são afetados, e o ganho do SLX foi medido por validação cruzada, não por coeficientes.

### 8.7 Influência

Os 10 maiores Cook de um passo são municípios pequenos, não capitais: `bb_selecao` lidera Jutaí/AM (0,025), São João das Missões/MG, Uiramutã/RR; `bb_slx`, São João das Missões/MG (0,040), Jutaí/AM (0,027), Brasília/DF (0,020). Reajuste sem os 10: maior mudança 0,69 EP no `bb_selecao` (`log_theta`) e 0,67 no `bb_slx` (`disp_reg_Norte`); 1 de 19 e 3 de 37 termos acima de 0,5 EP; 0 trocas de sinal. Contraste com o quase-binomial (seção 4: 25 de 32 acima de 0,5 EP).

### 8.8 Conclusão

Passa: escala da variância, preditor linear (exceto `pct_pretos_pardos`), estabilidade frente a influentes. Falha: ligação (efeito pequeno), formato da distribuição (cauda pesada, envelope), independência espacial. Fragilidade de leitura: coeficientes do `bb_slx` por colinearidade. Pendências em [12](12-pendencias.md).

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
