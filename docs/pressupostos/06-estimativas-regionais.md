# 06 - Estimativas regionais e intervalos (Parte 12)

Notebook: `notebooks/03-validacao.ipynb`, células 11 a 15 (contagem a partir de 0). Código: `src/modelos.py` (`simular_votos_grupo`, `BetaBinomial`).
Tabelas: `results/tables/estimativas_regionais.csv`, `estimativas_uf.csv`. Figuras: `results/figures/cv_estimativas_regionais.html`, `cv_estimativas_uf.html`.

## 1. Estimativa pontual (célula 12)

- **Decisão.** Cada UF é prevista pelo modelo escolhido (`bb_backward`) ajustado *sem ela* (LOSO). A estimativa regional é Σ n_i·π̂_i / Σ n_i sobre os municípios da região; o share de Flávio é 100 menos o de Lula.
- **Razão.** O ajuste com todos os dados reproduziria os totais regionais por construção; só o LOSO deixa a estimativa ser testada contra o resultado real.
- **Alvo "real".** Totais recalculados sobre os 5.570 municípios modelados. Boa Esperança do Norte (MT, sem voto de 2022, PT 945 e Flávio 2.353) fica fora; os CSVs oficiais usados já coincidem exatamente com as linhas modeladas
  (diferença 0 votos). Somá-la mudaria o Centro-Oeste em −0,004 p.p. e o Brasil em −0,0006 p.p. (célula 14).
- **Pressuposto.** Os votos válidos n_i de cada município são tratados como conhecidos; o modelo prevê só a divisão entre os dois candidatos.

## 2. Intervalos de previsão (células 12-13)

Função `simular_votos_grupo(beta_hat, cov_beta, X, n, grupos, link, theta, sigma, B, rng, offset)`: devolve uma matriz B × G de votos de Lula simulados por grupo.

1. β ~ N(β̂, Σ̂), com β̂ e Σ̂ do ajuste sem a UF (incerteza de estimação).
2. Choque de UF δ ~ N(0, σ²) somado ao preditor logit de todos os municípios da UF (a UF inteira erra junto).
3. π = g⁻¹(Xβ + δ); Y ~ BetaBin(n, πθ, (1−π)θ) com θ do ajuste (θ ≈ 630, ρ ≈ 0,0016). Sem θ, Y ~ Bin(n, π). θ é tratado como fixo.
4. B = 1.000 sorteios por UF, somados por região e para o Brasil; dividem-se pelos votos válidos e tomam-se os quantis 2,5/50/97,5%.

**Escolha de σ: RMS dos resíduos LOSO das UFs no logit (σ = 0,0536)**, não o σ do `glmm_uf` (0,0532).
- Razão: o σ do GLMM mede a variação entre UFs depois de dar a cada UF seu efeito, com todos os dados. Ao prever um estado inédito, o erro inclui também a extrapolação do efeito regional, que só o resíduo LOSO mede.
- Custo: σ é estimado nas mesmas 27 UFs em que a cobertura é conferida, então a cobertura por UF é otimista; e a variância de β entra duas vezes (no sorteio e no resíduo LOSO), o que alarga o intervalo.
- Na prática os dois σ são quase iguais, e a escolha não muda a conclusão.
- **Pressupostos.** (a) δ independente entre UFs; (b) sorteios de β independentes entre as 27 reestimações, embora elas usem quase os mesmos dados; (c) σ constante entre UFs.
  Com Moran de 0,90, (a) e (b) tendem a estreitar o intervalo regional, porque erros de UFs vizinhas andam juntos.

## 3. Resultados (células 13-14)

Share de Lula, % dos votos válidos (previsto fora da amostra, IC95%, real):

| Região | Previsto | IC95% | Real | Erro (p.p.) |
|---|---|---|---|---|
| Centro-Oeste | 40,32 | 38,61 - 41,97 | 39,99 | +0,33 |
| Nordeste | 69,42 | 68,50 - 70,27 | 69,37 | +0,05 |
| Norte | 49,50 | 48,10 - 51,03 | 50,01 | −0,51 |
| Sudeste | 47,71 | 46,11 - 49,47 | 47,24 | +0,47 |
| Sul | 37,00 | 35,51 - 38,46 | 37,42 | −0,43 |
| Brasil | 51,91 | 51,14 - 52,63 | 51,78 | +0,13 |

O share de Flávio é 100 menos cada valor (e o IC, espelhado). Erro absoluto médio: 0,36 p.p. nas regiões e 1,00 p.p. nas 27 UFs.

**Cobertura empírica do IC95%:**

| σ do choque de UF | UFs dentro | Regiões dentro | Largura média UF / região (p.p.) |
|---|---|---|---|
| RMS do LOSO (0,054) | 27/27 | 5/5 | 5,3 / 2,9 |
| GLMM (0,053) | 27/27 | 5/5 | 5,3 / 2,8 |
| 0 (só β e amostragem) | 20/27 (74%) | 4/5 | 2,4 / 1,2 |

## 4. O que se pode e o que não se pode concluir

- O modelo acerta o nível nacional e regional com erro de no máximo 0,51 p.p. em todas as regiões, em previsão fora da amostra. A ordem das regiões é reproduzida; Nordeste (69%) e Sul (37%) ficam separados dos demais com folga.
- **Cobertura.** Os 100% não provam calibração: com 5 regiões, 5/5 acertos é compatível com cobertura real a partir de ~55%; σ foi estimado nas mesmas UFs; os intervalos são largos (±2,6 p.p. por UF).
  Sem o choque de UF a cobertura cai para 74%, então **o choque de UF é necessário**: a incerteza de β e a amostragem beta-binomial subestimam muito o erro de prever um estado inédito.
- O intervalo do Norte (48,1-51,0%) inclui o empate de 50%: o modelo não separa o Norte de um resultado equilibrado.
- Um conjunto de teste melhor seria outra eleição (backtest, notebook 04); aqui há uma só eleição, então a cobertura é conferida nas próprias UFs do ano.

## 5. Sensibilidade ao modelo: por que o erro regional LOSO não escolhe modelo (células 16-17)

Tabela: `results/tables/sensibilidade_regional.csv`.

**Decisão.** A regra de escolha fixada antes dos resultados (menor erro LOSO por UF, depois 1 erro-padrão em favor de menos parâmetros) é mantida: `bb_backward`.

**Mecanismo [deduzido, não verificado por simulação].** Num binomial de ligação canônica com dummies de região, a equação de escore de cada dummy impõe Σ_u w_u δ_u = 0, com δ_u o resíduo de share da UF u dentro da amostra e w_u seu peso em votos na região.
Ao reter a UF u, o intercepto regional se move em Δ = −w_u δ_u / (1 − w_u), e o erro retido da UF fica ≈ δ_u / (1 − w_u). O erro regional LOSO é então Σ_u w_u δ_u / (1 − w_u).
- Pesos iguais: vale (1/(1 − w)) Σ_u w δ_u = 0 exatamente.
- Pesos desiguais: fica próximo de zero, não exatamente. A aproximação ignora a mudança nos demais coeficientes.
- Probit: satisfaz as equações ponderadas por votos só aproximadamente.
- Beta-binomial: pesa cada município por n/(1 + (n−1)ρ), logo o escore não é ponderado por votos e o cancelamento não ocorre.

**Consequência.** Um erro regional LOSO pequeno num binomial canônico mostra esse mecanismo, não um modelo melhor. A coluna regional de `cv_loso.csv` favorece esses modelos por construção; **a métrica justa de seleção é o erro por UF.**

Share de Lula (%), estimativa LOSO pontual; erro = estimativa − real (p.p.):

| Região | UFs | w_u máx. | Real | Probit | Logit | `bb_backward` |
|---|---|---|---|---|---|---|
| Centro-Oeste | 4 | 0,42 | 39,99 | 39,995 (+0,00) | 39,984 (−0,01) | 40,32 (+0,33) |
| Nordeste | 9 | 0,26 | 69,37 | 69,45 (+0,08) | 69,46 (+0,09) | 69,42 (+0,05) |
| Norte | 7 | 0,49 | 50,01 | 50,15 (+0,15) | 50,17 (+0,17) | 49,50 (−0,51) |
| Sudeste | 4 | 0,52 | 47,24 | 47,39 (+0,14) | 47,39 (+0,15) | 47,71 (+0,47) |
| Sul | 3 | 0,38 | 37,42 | 37,42 (0,00) | 37,43 (0,00) | 37,00 (−0,43) |
| Brasil | 27 | 0,21 | 51,78 | 51,87 (+0,09) | 51,88 (+0,10) | 51,91 (+0,13) |

| Métrica (p.p.) | Probit | Logit | `bb_backward` |
|---|---|---|---|
| MAE por UF (LOSO) | 0,930 | 0,943 | 1,000 |
| MAE regional (LOSO) | 0,075 | 0,083 | 0,357 |

- O cancelamento não é exato onde um estado domina a região (SP no Sudeste, w_u = 0,52): ali o erro do logit é +0,15 p.p., não zero.
- A diferença entre os três modelos é de 0,04 p.p. (Nordeste) a 0,68 p.p. (Norte) por região e de 0,03 p.p. no Brasil. Os três erram menos de 0,52 p.p. em qualquer região.
- **A escolha do modelo importa pouco para a resposta regional**: muda o share em até 0,7 p.p., menos que a largura do IC95% regional (2,9 p.p. em média). Importa só onde a região é próxima de 50%: o Norte vai de 49,50% (`bb_backward`) a 50,17% (logit), e a conclusão "Norte equilibrado" vale nos três.
- O custo de `bb_backward` em erro por UF (1,000 contra 0,930 do probit) está dentro de 1 erro-padrão, razão pela qual a regra o escolhe por ter menos parâmetros (20 contra 25).

## Referências

- Gelman, A.; Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3ª ed. Chapman & Hall/CRC (distribuição preditiva posterior).
- Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics* 31(4).
- Efron, B.; Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8).
