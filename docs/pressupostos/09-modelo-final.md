# 09 - Modelo final: regra de informação, estimativas regionais e cobertura (Parte 16 do `02-MLG`)

Notebook `notebooks/02-MLG.ipynb`. Tabelas em `results/tables/final_estimativas_regionais.csv`, `final_estimativas_uf.csv`, `final_cobertura.csv` e `final_comparacao_regional.csv`; figura `results/figures/final_regional.png`. Decisões anteriores: dispersão em [07](07-dispersao-regional.md), filtro espacial, SLX e Conley em [08](08-dependencia-espacial.md), extensões rejeitadas em [10](10-extensoes-rejeitadas.md).

**Modelo final = `bb_slx`**: beta-binomial com os termos de média de `bb_selecao` (18 contínuas, 3 dummies de região (Centro-Oeste, Norte, Sul; Nordeste e Sudeste fundidos) e o bloco de esquerda), dispersão pelas 5 regiões (`bb_disp`, [07](07-dispersao-regional.md): passo A, dummies de região, Wald de Conley ×1 (qui-quadrado) p = 5,59e-13, mantido; passo B, log n, p = 0,278, descartado) mais SLX, isto é, W·x das 18 variáveis dos vizinhos ([08](08-dependencia-espacial.md)). Tem **46 parâmetros** (41 de β e 5 de γ, `final_ep.csv`). MAE por UF: 0,1397 p.p. no 5-fold e 0,1940 no LOSO (`final_cv.csv`), abaixo do limite de aceitação L = 0,2904 (0,2541 sem aninhar; [05](05-validacao-cruzada.md), seção 5) e do de `bb_selecao`, 0,2134 (0,2497 aninhado). Nas decisões das extensões, MAE por UF no LOSO: `bb_disp` 0,2176, `bb_esf` 0,4802 (rejeitado; k = 25, I de Moran 0,2830 → 0,2180), `bb_final` 0,3404 e `bb_slx` 0,1940 (final; W_12vars, I de Moran 0,2830 → 0,2697). Os EPs são os de Conley a 150 km.

## 1. Regra de informação: o que cada estimativa pode saber

Cada número do trabalho responde a uma pergunta e só pode usar a informação dessa pergunta. A pergunta desta Parte é a da **UF inédita**: dadas 26 UFs com o 2º turno apurado, quanto dá a 27ª?

- **Decisão.** O modelo usa só covariáveis anteriores ao 2º turno: o comparecimento do 1º turno de 2026, a variação do comparecimento do 2º turno de 2022 (com o SLX sobre a mesma coluna anterior ao turno) e, **desde a mudança que trouxe o 1º turno do mesmo ano**, o resultado do 1º turno de 2026 por município (`lag1t`, logit de Lula/(Lula+Bolsonaro), e `outros_1t`, parcela dos votos válidos dos demais candidatos; denominador zero vale logit 0 e `outros_1t` 0). O 1º turno é conhecido antes do 2º, e para uma UF inédita ele existe: a regra de informação o admite, e o 1º turno do mesmo ano é entrada legítima, não vazamento. A seleção por BIC ficou com `lag1t` e descartou `outros_1t`. No LOSO, a UF retida entra com **n̂ no lugar do n real**: n̂ = eleitorado 2026 × (n / aptos do 2º turno de 2022). As UFs de treino mantêm o n real, porque já foram apuradas.
- **Razão.** O comparecimento do próprio 2º turno e o n real (votos Lula + Flávio de 2026) não existem para uma UF inédita nem durante a apuração. Um LOSO que os usa responde a outra pergunta (imputar com informação contemporânea) e dá um erro menor do que o de uma previsão real. Valor histórico, anterior à mudança e não reproduzido nesta revisão: o vazamento era pequeno e de sinal misto (cerca de 0,05 p.p. no MAE por UF, na escala do modelo anterior). A regra remove o vazamento por princípio: a pergunta só pode usar a informação que existe nela.
- **Viabilidade de n̂.** A soma de n̂ fica em 0,969 da soma do n real, com erro percentual médio municipal de 3,81 % (mediana 2,88 %; recalculado com `n_estimado` (`src/especificacoes.py`) sobre `preparar` (`src/backtest.py`) em `data/processed/municipios.parquet`).
- **Consequência.** Como o LOSO já pesa a UF retida por n̂ e o modelo só usa covariáveis pré-turno, a estimativa desta Parte é a do cenário "UF inédita"; não há segunda versão "sem vazamento" a reportar. As outras perguntas ficam em outras Partes: explicação com todos os dados (Partes 5 a 10, 13 a 17), previsão ex-ante ([06](06-backtest.md)) e apuração parcial na ordem real das seções ([11](11-apuracao-parcial.md)).

### Todos os dados ou uma amostra?

Os municípios são um censo. Uma amostra aleatória de municípios seria um teste mais fraco que o LOSO, por dois fatos de dependência espacial: vizinhos votam parecido (I de Moran da proporção observada 0,90) e o que o modelo não explica também é espacial (I dos resíduos 0,27 dentro da amostra e 0,30 no LOSO, [10](10-extensoes-rejeitadas.md)). Numa amostra aleatória cada município retido teria vizinhos de treino com resíduos parecidos, e o erro sairia otimista: o mesmo efeito do 5-fold (MAE por UF 0,140 contra 0,194 no LOSO). O LOSO retira a UF inteira e só deixa os vizinhos da divisa, por isso é o teste honesto desta pergunta.

## 2. Receita das estimativas

Para cada UF deixada de fora, ajusta-se nas 26 restantes; o bloco de β da covariância de Conley (150 km) é calculado nesse ajuste de treino; θᵢ é previsto por município; B = 1000 sorteios no erro regional e B = 10 000 na cobertura por UF, semente 7. Pressupostos: δ de UF independente entre UFs, sorteios de β independentes entre as 27 reestimações (embora usem quase os mesmos dados) e σ constante. Com I de Moran alto, os dois primeiros tendem a estreitar o intervalo regional.

1. β ~ N(β̂, Σ̂) do ajuste sem a UF (incerteza de estimação, com Conley).
2. Choque de UF δ = σ̂·T_k, T de Student com k graus de liberdade, somado ao preditor logit de todos os municípios da UF (a UF inteira erra junto). k = número de UFs que estimam σ̂: 27 no 02; 26 no σ_{-u} aninhado; 27−|r| no σ_{-r} de uma região. **Razão:** σ̂ é a RMS de k resíduos de UF de média zero, então o resíduo de uma UF nova é σ̂·T_k; tratar σ̂ como conhecido estreita o intervalo. A mesma regra vale no 03 ([11](11-apuracao-parcial.md)).
3. Y ~ BetaBin(n̂, πθ, (1−π)θ), com θ previsto por município.
4. Somam-se os votos por região e Brasil e tomam-se os quantis.

- **σ_UF = 0,0114**: RMS dos resíduos LOSO das UFs no logit (antes da covariável do 1º turno do mesmo ano: 0,0570). Razão da escolha: o σ de um GLMM mede a variação entre UFs com todos os dados; ao prever uma UF inédita o erro inclui também a extrapolação do efeito regional, que só o resíduo LOSO mede.
- **σ_UF aninhado (conferência).** Para pontuar a UF u, o intervalo usa σ_{-u}: RMS dos resíduos LOSO das outras 26 UFs (0,0102 a 0,0116; `results/tables/final_sigma_aninhado.csv`). Para a região r, σ_{-r}: RMS das UFs fora de r (Centro-Oeste 0,0121, Nordeste 0,0107, Norte 0,0128, Sudeste 0,0090, Sul 0,0118). O Brasil não tem versão aninhada (usaria todas as UFs). Razão: a cobertura não pode depender de σ ajustado na UF pontuada.
- **Alvo "real".** Totais recalculados sobre os 5.570 municípios modelados (Boa Esperança do Norte, MT, fica fora; efeito −0,004 p.p. no Centro-Oeste).

## 3. Estimativas regionais

Lula %, Conley + σ_UF (choque t), intervalo de 95% (Brasil: 90%). Erro = estimativa − real.

| Região | Real | Estimado | IC | erro (p.p.) |
|---|---|---|---|---|
| Centro-Oeste | 39,99 | 40,13 | [39,77; 40,47] | +0,14 |
| Nordeste | 69,37 | 69,55 | [69,33; 69,77] | +0,19 |
| Norte | 50,01 | 49,91 | [49,57; 50,26] | −0,10 |
| Sudeste | 47,24 | 47,50 | [47,11; 47,91] | +0,26 |
| Sul | 37,42 | 37,47 | [37,15; 37,80] | +0,05 |
| Brasil | 51,78 | 51,78 | [51,63; 51,94] (90%) | +0,00 |

- Erro regional absoluto médio 0,145 p.p. (as 5 regiões); por UF, 0,194 p.p. (LOSO). São só 5 regiões, e o erro regional LOSO não escolhe modelos ([05](05-validacao-cruzada.md), seção 6): é descrição, não evidência de superioridade.
- O intervalo do Norte ainda inclui o empate de 50% ([49,57; 50,26]): o modelo não separa o Norte de um resultado equilibrado. Nordeste e Sul ficam separados dos demais com folga.

## 4. Cobertura

Choque de UF t (regra da seção 2); entre parênteses, o mesmo com choque normal (`final_cobertura.csv`, B = 10 000).

| Covariância + σ | UF 95% | UF 90% | Região 95% | Região 90% |
|---|---|---|---|---|
| Conley + σ_UF | 92,6% (25/27) [normal: 25/27] | 88,9% | 5/5 | 100% [80%] |
| Conley + σ_UF aninhado | 92,6% (25/27) [25/27] | 88,9% | 5/5 | 80% [80%] |
| Conley + σ = 0 | 77,8% (21/27) | 74,1% | 3/5 | 60% |
| Modelo + σ_UF | 92,6% (25/27) [25/27] | 88,9% | 5/5 | 100% [80%] |
| Modelo + σ_UF aninhado | 92,6% (25/27) [24/27] | 88,9% | 5/5 | 80% [60%] |
| Modelo + σ = 0 | 70,4% (19/27) | 70,4% | 3/5 | 60% |

Largura média do IC 95% com Conley + σ_UF: 1,296 p.p. por UF e 0,657 por região com o choque t (normal: 1,261 e 0,639; com σ = 0: 0,750 e 0,335). A cobertura a 90% por UF é 0,889 em todas as linhas com σ_UF.

- **O choque de UF leva a cobertura perto do nominal.** Com σ = 0, 77,8% das UFs ficam dentro do IC95 (21 de 27; 70,4% com o EP do modelo); com σ_UF, 25 de 27, com t e com normal. As 2 UFs fora são **BA** (erro +0,57 p.p., IC95 de 1,03 p.p. de largura) e **SP** (+0,69 p.p., largura 1,34), em `final_estimativas_uf.csv`.
- **Teste binomial.** Se a cobertura verdadeira fosse 95%, P(X ≤ 25 | n = 27) = 0,394 (0,151 para os 24/27 da rodada anterior) e E[X] = 25,65. Logo 25/27 **não** é evidência de cobertura insuficiente; 27 UFs são poucas para estimar cobertura.
- **Regra do σ_UF fica como está.** Alternativas rejeitadas: bootstrap dos resíduos das outras UFs, choque regional, ou um σ escolhido para trazer as UFs para dentro. Cada uma seria ajustada nos mesmos 27 pontos que ela pontua. B = 10 000 na cobertura por UF mantém o ruído de Monte Carlo (cerca de 0,02 a 0,03 p.p. no quantil de 2,5% com B = 1000) abaixo do que decidiria uma UF limítrofe; as linhas com choque normal ficam como comparação.
- **Conley ajuda pouco.** Com σ = 0, passa de 70,4% para 77,8% das UFs, pois a covariância de β fica maior.
- **σ_UF aninhado não muda a cobertura por UF** (25/27 a 95%, 88,9% a 90%); a cobertura regional a 90% cai de 5/5 para 4/5. Logo a cobertura não depende de σ ter sido ajustado na UF pontuada.
- **Otimismo restante.** Com 5 regiões, 5/5 acertos é compatível com cobertura real a partir de ~55%. A cobertura vale como descrição, não como validação.

## 5. Decisão

Os resultados do projeto vêm do modelo final `bb_slx` com EPs de Conley. `bb_selecao` segue como o modelo escolhido pela regra de seleção do [doc 01](01-visao-geral.md); o modelo final é o refinamento dele (dispersão por região, mais SLX das 18 variáveis). Com o 1º turno do mesmo ano entre as covariáveis, nenhuma aceitação ou rejeição das extensões mudou (dispersão e SLX aceitos; ESF, krigagem e CAR rejeitados); mudaram a dispersão (`regiao` no lugar de `regiao_logn`), o k do ESF (25 no lugar de 50) e o conjunto de variáveis (18 contínuas no lugar de 13, com Nordeste e Sudeste fundidos; [01](01-visao-geral.md)).

## Referências

- Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3ª ed. Chapman & Hall/CRC (distribuição preditiva posterior).
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
