# 09 - Modelo final: regra de informação, estimativas regionais e cobertura (Parte 16 do `02-MLG`)

Notebook `notebooks/02-MLG.ipynb`. Tabelas em `results/tables/final_estimativas_regionais.csv`, `final_estimativas_uf.csv`, `final_cobertura.csv` e `final_comparacao_regional.csv`; figura `results/figures/final_regional.png`. Decisões anteriores: dispersão em [07](07-dispersao-regional.md), filtro espacial, SLX e Conley em [08](08-dependencia-espacial.md), extensões rejeitadas em [10](10-extensoes-rejeitadas.md).

**Modelo final = `bb_slx`**: beta-binomial com os termos de média de `bb_selecao` (com a fusão de 8.1: Nordeste e Norte numa só dummy na média e só o bloco de esquerda separado; a fusão vale para toda a linhagem `bb_selecao`, e só a comparação de famílias da Parte 6 mantém os grupos padrão), dispersão pelas 5 regiões separadas e log n (`bb_disp`, [07](07-dispersao-regional.md)) mais SLX, isto é, W·x das 13 variáveis dos vizinhos ([08](08-dependencia-espacial.md)). Tem **37 parâmetros** (31 de β e 6 de γ). MAE por UF: 0,846 p.p. no 5-fold e 1,073 no LOSO (limite de aceitação 1,2088; o valor fica acima do de `bb_selecao`, 1,005, dentro da tolerância). Os EPs são os de Conley a 200 km.

## 1. Regra de informação: o que cada estimativa pode saber

Cada número do trabalho responde a uma pergunta e só pode usar a informação dessa pergunta. A pergunta desta Parte é a da **UF inédita**: dadas 26 UFs com o 2º turno apurado, quanto dá a 27ª?

- **Decisão.** O modelo usa só covariáveis anteriores ao 2º turno (por exemplo, o comparecimento do 1º turno de 2026 e a variação do comparecimento do 2º turno de 2022, com o SLX sobre a mesma coluna anterior ao turno). No LOSO, a UF retida entra com **n̂ no lugar do n real**: n̂ = eleitorado 2026 × (n / aptos do 2º turno de 2022). As UFs de treino mantêm o n real, porque já foram apuradas.
- **Razão.** O comparecimento do próprio 2º turno e o n real (votos Lula + Flávio de 2026) não existem para uma UF inédita nem durante a apuração. Um LOSO que os usa responde a outra pergunta (imputar com informação contemporânea) e dá um erro menor do que o de uma previsão real. Valor histórico, anterior à mudança e não reproduzido nesta revisão: o vazamento era pequeno e de sinal misto (cerca de 0,05 p.p. no MAE por UF). A regra remove o vazamento por princípio: a pergunta só pode usar a informação que existe nela.
- **Viabilidade de n̂.** A soma de n̂ fica em 0,969 da soma do n real, com erro percentual médio municipal de 3,81 % (mediana 2,88 %; conferido nos dados atuais).
- **Consequência.** Como o LOSO já pesa a UF retida por n̂ e o modelo só usa covariáveis pré-turno, a estimativa desta Parte é a do cenário "UF inédita"; não há segunda versão "sem vazamento" a reportar. As outras perguntas ficam em outras Partes: explicação com todos os dados (Partes 5 a 10, 13 a 17), previsão ex-ante ([06](06-backtest.md)) e apuração parcial na ordem real das seções ([11](11-apuracao-parcial.md)).

### Todos os dados ou uma amostra?

Os municípios são um censo. Uma amostra aleatória de municípios seria um teste mais fraco que o LOSO, por dois fatos de dependência espacial: vizinhos votam parecido (I de Moran da proporção observada 0,90) e o que o modelo não explica também é espacial (I dos resíduos 0,35 dentro da amostra e 0,44 no LOSO, [10](10-extensoes-rejeitadas.md)). Numa amostra aleatória cada município retido teria vizinhos de treino com resíduos parecidos, e o erro sairia otimista: o mesmo efeito do 5-fold (MAE por UF 0,846 contra 1,073 no LOSO). O LOSO retira a UF inteira e só deixa os vizinhos da divisa, por isso é o teste honesto desta pergunta.

## 2. Receita das estimativas

Para cada UF deixada de fora, ajusta-se nas 26 restantes; o bloco de β da covariância de Conley (200 km) é calculado nesse ajuste de treino; θᵢ é previsto por município; B = 1000 sorteios, semente 7. Pressupostos: δ de UF independente entre UFs, sorteios de β independentes entre as 27 reestimações (embora usem quase os mesmos dados) e σ constante. Com I de Moran alto, os dois primeiros tendem a estreitar o intervalo regional.

1. β ~ N(β̂, Σ̂) do ajuste sem a UF (incerteza de estimação, com Conley).
2. Choque de UF δ ~ N(0, σ²_UF) somado ao preditor logit de todos os municípios da UF (a UF inteira erra junto).
3. Y ~ BetaBin(n̂, πθ, (1−π)θ), com θ previsto por município.
4. Somam-se os votos por região e Brasil e tomam-se os quantis.

- **σ_UF = 0,0570**: RMS dos resíduos LOSO das UFs no logit (antes: 0,0558 no commit anterior e 0,0564 antes das três covariáveis novas). Razão da escolha: o σ de um GLMM mede a variação entre UFs com todos os dados; ao prever uma UF inédita o erro inclui também a extrapolação do efeito regional, que só o resíduo LOSO mede.
- **Ressalva.** σ_UF é estimado nas mesmas UFs em que a cobertura é conferida; a cobertura com σ_UF é **otimista**. Pendência: σ_UF por validação aninhada ([12](12-pendencias.md)).
- **Alvo "real".** Totais recalculados sobre os 5.570 municípios modelados (Boa Esperança do Norte, MT, fica fora; efeito −0,004 p.p. no Centro-Oeste).

## 3. Estimativas regionais

Lula %, Conley + σ_UF, intervalo de 95% (Brasil: 90%). Erro = estimativa − real.

| Região | Real | Estimado | IC | erro (p.p.) |
|---|---|---|---|---|
| Centro-Oeste | 39,99 | 39,96 | [38,44; 41,42] | −0,03 |
| Nordeste | 69,37 | 69,58 | [68,56; 70,49] | +0,21 |
| Norte | 50,01 | 49,28 | [47,75; 50,78] | −0,72 |
| Sudeste | 47,24 | 47,94 | [46,26; 49,64] | +0,69 |
| Sul | 37,42 | 36,92 | [35,38; 38,39] | −0,50 |
| Brasil | 51,78 | 51,82 | [51,15; 52,50] (90%) | +0,04 |

- Erro regional absoluto médio 0,431 p.p. (as 5 regiões); por UF, 1,073 p.p. (LOSO). São só 5 regiões, e o erro regional LOSO não escolhe modelos ([05](05-validacao-cruzada.md), seção 6): é descrição, não evidência de superioridade.
- O intervalo do Norte inclui o empate de 50%: o modelo não separa o Norte de um resultado equilibrado. Nordeste e Sul ficam separados dos demais com folga.

## 4. Cobertura

| Covariância + σ | UF 95% | UF 90% | Região 95% | Região 90% |
|---|---|---|---|---|
| Conley + σ_UF | 100% (27/27) | 92,6% | 5/5 | 5/5 |
| Conley + σ = 0 | 48,1% (13/27) | 40,7% | 2/5 | 40% |
| Modelo + σ_UF | 100% (27/27) | 92,6% | 5/5 | 5/5 |
| Modelo + σ = 0 | 33,3% (9/27) | 29,6% | 2/5 | 20% |

Largura média do IC 95% com Conley + σ_UF: 5,42 p.p. por UF e 2,87 por região (com σ = 0: 2,01 e 0,82).

- **O choque de UF é o que sustenta a cobertura.** Com σ = 0 só 48% das UFs ficam dentro do IC95 (13 de 27; no commit anterior, 52%), ou 33% com o EP do modelo, porque o erro por UF é um choque comum a todos os municípios da UF que o modelo não vê; com σ_UF a cobertura é de 100%. A cobertura IC90 com σ_UF caiu de 96,3% para 92,6%.
- **Conley ajuda.** Com σ = 0, passa de 33% para 48% das UFs, pois a covariância de β fica maior.
- **Otimismo.** Os 100% não provam calibração: com 5 regiões, 5/5 acertos é compatível com cobertura real a partir de ~55%, e σ_UF entra ajustado nas mesmas 27 UFs. A cobertura vale como descrição, não como validação.

## 5. Decisão

Os resultados do projeto vêm do modelo final `bb_slx` com EPs de Conley. `bb_selecao` segue como o modelo escolhido pela regra de seleção do [doc 01](01-visao-geral.md); o modelo final é o refinamento dele (dispersão por região e log n, mais SLX das 13 variáveis). Nenhuma aceitação ou rejeição das extensões mudou com as três covariáveis novas (fund. incompleto, sem religião, indígenas); o conjunto de variáveis perde três das 12 anteriores e ganha quatro ([01](01-visao-geral.md)).

## Referências

- Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3ª ed. Chapman & Hall/CRC (distribuição preditiva posterior).
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
