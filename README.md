# Previsão das Eleições 2026 por MLG — 2º turno presidencial (Lula × Flávio Bolsonaro)

## 📌 Sobre o projeto

Este repositório contém o desenvolvimento de um trabalho acadêmico sobre **Modelos Lineares Generalizados (MLG) da família binomial**, aplicados à votação por município no **segundo turno presidencial de 2026** (Lula × Flávio Bolsonaro), com agregação das previsões às **5 regiões** do Brasil.

O objetivo é comparar especificações de MLG com um critério de escolha fixado **antes** dos resultados e estimar a participação regional de cada candidato com o modelo escolhido.

> **Fontes principais:** TSE, IBGE, Portal da Transparência (CGU) e Bolognesi, Ribeiro e Codato (2023). Detalhes na seção [Sobre o Dataset](#-sobre-o-dataset).

A base final tem **5.570 municípios** (um censo, não uma amostra) e 160 colunas. Notebooks e documentação em português.

> **Para começar:** a seção [O trabalho passo a passo](#-o-trabalho-passo-a-passo) conta, em linguagem simples, o que fizemos em cada etapa e o que concluímos.

> **Ressalva (falácia ecológica).** O modelo descreve **municípios**, não eleitores: os coeficientes são associações entre municípios e não permitem inferir o comportamento individual (Robinson, 1950). Veja todas as [limitações](#limitações).

---

## 🎯 Objetivos

### Objetivo geral

Estimar a participação regional de Lula e de Flávio Bolsonaro no 2º turno de 2026 com um MLG binomial escolhido por regra fixada antes dos resultados ([`docs/pressupostos/01-visao-geral.md`](docs/pressupostos/01-visao-geral.md)).

### Objetivos específicos

* Coletar e consolidar as fontes de dados (TSE, IBGE, CGU) em uma base por município;
* Realizar uma análise exploratória da votação e das covariáveis;
* Comparar famílias e ligações (binomial, quase-binomial, beta-binomial, GLMM, regressão beta);
* Diagnosticar superdispersão, resíduos, influência e autocorrelação espacial;
* Escolher o modelo por validação cruzada leave-one-state-out (LOSO) e regra do 1 erro-padrão;
* Estimar as participações regionais apenas com previsões fora da amostra, com intervalos de incerteza;
* Testar a previsão em dados não vistos na apuração parcial (2022 e 2026, ordem real de apuração);
* Modelar a dispersão por região e a dependência espacial (SLX, filtro de autovetores, EPs de Conley);
* Comparar as estimativas finais com o resultado real.

---

## 📊 Sobre o Dataset

Fontes (regras de merge e limitações em [`docs/dados/consolidacao.md`](docs/dados/consolidacao.md)):

* **TSE:** votação presidencial 2018, 2022 e 2026 por município; perfil do eleitorado 2022 e 2026; votos em deputado federal e partido do prefeito (ideologia).
* **IBGE:** Censo 2022, PIB dos Municípios, tabela de equivalência IBGE × TSE e malha municipal.
* **Portal da Transparência (CGU):** famílias do Auxílio Brasil (set/2022) e Bolsa Família (set/2026), porque a API do MDS está indisponível.
* **Ideologia dos partidos:** classificação de Bolognesi, Ribeiro e Codato (2023).


| Variável | Papel | Unidade | Mín | Mediana | Máx | Caxingó (PI) | Fonte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `y` (votos de Lula no 2º turno de 2026) | Resposta | contagem | 197 | 3.851 | 3.864.354 | 3.800 | TSE |
| `n` (votos em Lula + Flávio, só Brasil) | Total (Lula + Flávio) | contagem | 848 | 7.201 | 6.962.277 | 4.628 | TSE |
| `lag`: Logit da participação de Lula no 2º turno de 2022 | Defasagem | logit | -2,090 | 0,145 | 2,727 | 1,757 | TSE |
| `lag1t`: Logit de Lula/(Lula+Bolsonaro) no 1º turno de 2026 (conhecido antes do 2º turno) | Defasagem do mesmo ciclo | logit | -2,467 | -0,060 | 2,334 | 1,413 | TSE |
| `outros_1t`: Fração dos votos válidos do 1º turno de 2026 que foi para os demais candidatos (candidata, sai na seleção) | Defasagem do mesmo ciclo | fração | 0,012 | 0,057 | 0,332 | 0,043 | TSE |
| `nota_dep`: Nota ideológica média dos votos de deputado federal (0 esquerda a 10 direita) | Ideologia | escala 0-10 | 2,898 | 6,710 | 8,074 | 5,315 | TSE + Bolognesi et al. (2023) |
| `pct_dep_esq`: Votos de deputado federal em partidos de esquerda | Ideologia | fração | 0,005 | 0,111 | 0,802 | 0,453 | TSE + Bolognesi et al. (2023) |
| `comparecimento`: Comparecimento no 1º turno de 2026 | Mobilização | fração | 0,599 | 0,794 | 0,927 | 0,862 | TSE |
| `delta_comp`: Variação do comparecimento no 2º turno, de 2018 para 2022 | Mobilização | fração | -0,143 | 0,016 | 0,156 | -0,003 | TSE |
| `log_pop`: Log da população | Território | log(hab) | 6,725 | 9,312 | 16,254 | 8,612 | IBGE (Censo 2022) |
| `log_dens`: Log da densidade demográfica | Território | log(hab/km²) | -1,872 | 3,191 | 9,541 | 2,416 | IBGE (Censo 2022) |
| `pct_mulheres`: Mulheres no eleitorado | Demografia | fração | 0,434 | 0,509 | 0,557 | 0,490 | TSE (perfil) |
| `pct_60_mais`: Eleitorado de 60 anos ou mais | Demografia | fração | 0,091 | 0,244 | 0,467 | 0,179 | TSE (perfil) |
| `pct_16_24`: Eleitorado de 16 a 24 anos (candidata, sai na seleção) | Demografia | fração | 0,058 | 0,123 | 0,304 | 0,173 | TSE (perfil) |
| `pct_fund_inc`: Eleitorado com até o fundamental incompleto | Escolaridade | fração | 0,085 | 0,422 | 0,676 | 0,616 | TSE (perfil) |
| `pct_superior`: Eleitorado com ensino superior completo | Escolaridade | fração | 0,015 | 0,066 | 0,366 | 0,035 | TSE (perfil) |
| `pct_analfabeto`: Eleitorado analfabeto (candidata, sai na seleção) | Escolaridade | fração | 0,001 | 0,046 | 0,276 | 0,208 | TSE (perfil) |
| `cobertura_pbf`: Famílias do Bolsa Família sobre a população (pode passar de 1) | Renda e proteção social | razão | 0,003 | 0,267 | 1,464 | 0,731 | CGU + IBGE |
| `pct_vab_agro`: Valor adicionado bruto da agropecuária no total | Economia | fração | -0,419 | 0,190 | 0,935 | 0,127 | IBGE (PIB dos Municípios) |
| `pib_pc`: PIB per capita de 2023 (o modelo usa o log) | Economia | R$ por habitante | 7.701 | 29.895 | 722.442 | 12.485 | IBGE (PIB dos Municípios) |
| `pct_evangelicos`: Evangélicos na população | Religião | fração | 0,013 | 0,217 | 0,887 | 0,082 | IBGE (Censo 2022) |
| `pct_sem_religiao`: Pessoas sem religião (candidata, sai na seleção) | Religião | fração | 0,000 | 0,042 | 0,378 | 0,041 | IBGE (Censo 2022) |
| `pct_indigenas`: Indígenas na população (candidata, sai na seleção) | Raça/cor | fração | 0,000 | 0,001 | 0,945 | 0,000 | IBGE (Censo 2022) |
| `pct_pretos_pardos`: Pretos e pardos na população | Raça/cor | fração | 0,025 | 0,608 | 0,972 | 0,808 | IBGE (Censo 2022) |
| `pct_urbana`: População em situação urbana | Território | fração | 0,073 | 0,708 | 1,000 | 0,184 | IBGE (Censo 2022) |
| `bloco_pref`: ideologia do partido do prefeito eleito em 2024 | Ideologia local | categórica | direita (4770), esquerda (735), centro (49), sem prefeito (16) | | | direita | TSE + Bolognesi et al. (2023) |
| `regiao`: região do IBGE | Região | categórica | Nordeste (1794), Sudeste (1668), Sul (1191), Centro-Oeste (467), Norte (450) | | | Nordeste | IBGE |


Todos os percentuais são frações em [0, 1]; `pct_vab_agro` e `cobertura_pbf` fogem da faixa (valores da fonte). Nenhuma covariável do 2º turno de 2026 entra no modelo (ela é desconhecida enquanto a apuração não chega). As covariáveis do **1º turno de 2026** (`lag1t`, `outros_1t` e o comparecimento) entram porque são entradas conhecidas antes do 2º turno, e o 2º turno só começa depois da apuração do 1º. As candidatas `outros_1t`, `pct_16_24`, `pct_analfabeto`, `pct_sem_religiao` e `pct_indigenas` entram nos modelos candidatos e saem na seleção por BIC; `bloco_pref` (`bloco_prefeito_2024`) **permanece** no modelo ([`docs/pressupostos/03-modelos-candidatos.md`](docs/pressupostos/03-modelos-candidatos.md)).

---

## 🎯 Variável resposta

A resposta é o número de votos de Lula no município, dado o total de votos nos dois candidatos:

* $Y_i$ = votos de Lula no município $i$;
* $n_i$ = votos de Lula + votos de Flávio (só Brasil; o exterior, UF `ZZ`, fica fora).

$$
Y_i \sim \mathrm{Bin}(n_i, \pi_i), \qquad g(\pi_i) = x_i^\top \beta, \qquad \pi_{\text{Flávio}} = 1 - \pi_{\text{Lula}}
$$

Participação regional (soma sobre os municípios da região $r$):

$$
\hat p_r = \frac{\sum_i \hat\pi_i\, n_i}{\sum_i n_i}
$$

onde:

* $g$ é a função de ligação (logit no modelo final);
* $x_i$ são as covariáveis do município;
* participações são frações nos dados e pontos percentuais (p.p.) nos relatórios.

**Exemplo.** Dois municípios na mesma região: $n = 10.000$ com $\hat\pi = 0{,}60$ e $n = 30.000$ com $\hat\pi = 0{,}40$. Então $\hat p = (6.000 + 12.000)/40.000 = 45{,}0\%$ para Lula e $55{,}0\%$ para Flávio.

**Casos de borda.**

* Boa Esperança do Norte (MT) não está nas 5.570 linhas (instalado depois do Censo 2022); seus votos (PT 945, Bolsonaro 2.353) são reportados à parte e mudam o Centro-Oeste em −0,004 p.p.;
* o voto no exterior (UF `ZZ`) é descartado;
* brancos e nulos ficam fora de $n_i$;
* com ligação canônica e dummies de região, o total regional **dentro da amostra** é exato por construção; por isso as estimativas regionais vêm só de previsões fora da amostra (LOSO).

---

## 🔎 Variáveis explicativas

O modelo final usa a defasagem (`lag`) e 17 variáveis contínuas escolhidas por seleção passo a passo por BIC (`bb_selecao`), entre elas o voto do 1º turno do mesmo ano (`lag1t`, conhecido antes do 2º turno), mais as mesmas 18 defasadas espacialmente (SLX), o bloco do prefeito (só esquerda; centro e sem prefeito fundidos à direita) e dummies de região (três, Centro-Oeste, Norte e Sul contra o Sudeste, que absorve o Nordeste pela fusão), num total de 46 parâmetros (41 da média e 5 da dispersão).

Além dessas 18, o modelo usa:

* `W·x` de cada uma das 18 (W = vizinhança Queen, padronizada por linha);
* `bloco_pref` (esquerda; referência: direita, que absorve centro e sem prefeito);
* região (Centro-Oeste, Nordeste, Norte, Sul; referência: Sudeste).

As 18 variáveis contínuas, em linguagem simples. Exemplo: Recife (PE). Faixa: mínimo e máximo nos 5.570 municípios. Todo `pct_*` é fração em [0, 1] (0,15 = 15%). Definições completas em [`docs/dados/dicionario.md`](docs/dados/dicionario.md#variáveis-do-modelo-final-exemplo-e-faixa).

| Nome | Descrição | Unidade | Exemplo | Faixa |
| --- | --- | --- | --- | --- |
| `lag` | Voto de Lula no 2º turno de 2022 (logit da participação) | logit | 0,254 | -2,090 a 2,727 |
| `lag1t` | Voto de Lula contra Bolsonaro no 1º turno de 2026 (logit); entrada conhecida antes do 2º turno | logit | 0,365 | -2,467 a 2,334 |
| `nota_dep` | Nota ideológica média dos votos de deputado federal (0 esquerda a 10 direita) | escala 0-10 | 5,600 | 2,898 a 8,074 |
| `pct_dep_esq` | Votos de deputado federal em partidos de esquerda | fração | 0,153 | 0,005 a 0,802 |
| `comparecimento` | Quem votou, entre os aptos, no 1º turno de 2026 | fração | 0,841 | 0,599 a 0,927 |
| `delta_comp` | Variação do comparecimento no 2º turno, de 2018 para 2022 | fração | 0,001 | -0,143 a 0,156 |
| `log_pop` | População, em log | log(hab) | 14,214 | 6,725 a 16,254 |
| `log_dens` | Densidade demográfica, em log | log(hab/km²) | 8,823 | -1,872 a 9,541 |
| `pct_mulheres` | Mulheres no eleitorado | fração | 0,554 | 0,434 a 0,557 |
| `pct_60_mais` | Eleitores de 60 anos ou mais | fração | 0,257 | 0,091 a 0,467 |
| `pct_superior` | Eleitores com ensino superior completo | fração | 0,180 | 0,015 a 0,366 |
| `cobertura_pbf` | Famílias do Bolsa Família por domicílio (pode passar de 1) | razão | 0,251 | 0,003 a 1,464 |
| `pib_pc` | PIB per capita de 2023 (o modelo usa o log; exemplo e faixa em R$ por habitante) | R$ | 44.563 | 7.701 a 722.442 |
| `pct_vab_agro` | Agropecuária no valor adicionado bruto (1 município tem VAB agro negativo) | fração | 0,001 | -0,419 a 0,935 |
| `pct_evangelicos` | Evangélicos na população | fração | 0,278 | 0,013 a 0,887 |
| `pct_pretos_pardos` | Pretos e pardos na população | fração | 0,608 | 0,025 a 0,972 |
| `pct_urbana` | População em situação urbana | fração | 1,000 | 0,073 a 1,000 |
| `pct_fund_inc` | Eleitores com até o fundamental incompleto | fração | 0,207 | 0,085 a 0,676 |

Seleção: o passo a passo por BIC (backward e both) chega ao mesmo conjunto, com BIC 62.571,43 (o forward para num conjunto de 18 variáveis, BIC 62.575,35); a busca em todos os subconjuntos não achou esse ótimo (melhor BIC 62.571,70). As covariáveis contínuas são padronizadas (escore z) só na dobra de treino, sem vazamento na validação cruzada. Dispersão: $\log\theta = Z\gamma$, com dummies de região.

---

## 🧭 O trabalho passo a passo

Esta seção conta o trabalho em ordem, em linguagem simples: o que fizemos em cada etapa, o que
concluímos e por que isso importa. Os detalhes técnicos estão nas seções seguintes e em
[`docs/pressupostos/`](docs/pressupostos/README.md).

**1. O ponto de partida.** Juntamos TSE, IBGE e CGU numa tabela com os 5.570 municípios e 160
colunas. A pergunta: de cada 100 votos dados a Lula ou a Flávio num município, quantos foram
para Lula? Como temos todos os municípios, não há amostra: o desafio é prever bem o que não foi
usado no ajuste. Começamos com **26 covariáveis numéricas**, mais o bloco partidário do prefeito
e a região.

**2. Olhando os dados (Partes 1 a 4).** O voto de 2022 explica quase tudo: o logit de 2026
contra o de 2022 tem inclinação 0,96 e R² 0,976. A inclinação fica abaixo de 1, então há
regressão à média: municípios muito lulistas ou muito bolsonaristas ficam um pouco menos
extremos. Por isso estimamos o peso de 2022 em vez de fixá-lo em 1. Entre 2018 e 2022, quando o
candidato do PT mudou, a inclinação foi 0,80. A mudança por região foi pequena (no máximo 0,06
em log-odds). Municípios vizinhos se parecem muito (I de Moran 0,90 na proporção e 0,54 na
mudança), e isso volta em quase todas as etapas ([02](docs/pressupostos/02-aed.md)).

**3. Tirando as variáveis repetidas: 26 → 22.** Pelo VIF, algumas covariáveis mediam a mesma
coisa: na lista cheia o VIF chegou a 124. Saíram quatro:

* `log_eleitorado_2026`: correlação 0,99 com `log_populacao_2022`;
* `pct_catolicos_2022`: católicos, evangélicos e sem religião somam quase 100%, então a terceira
  sai das outras duas;
* `taxa_alfabetizacao_15mais_2022`: correlação −0,90 com `pct_analfabeto_2026`, que é mais recente;
* `nota_media_dep_2026`: mede o mesmo que `pct_dep_esq_2026`, que é mais fácil de interpretar.

Com a lista reduzida, o VIF máximo ainda é 78,3 (`lag1t`; `lag` tem 71,6): os dois logits medem quase o mesmo voto e ficam juntos por decisão.

**4. Da lista recomendada ao modelo: 22 → 23.** A lista que entra nos modelos é a recomendada
pela AED (22 numéricas, `lag` incluído) mais `nota_dep`, que volta como candidata. A razão dessa
volta não está registrada; ela está nas [pendências](docs/pressupostos/12-pendencias.md). As três
covariáveis `pct_ate_fund_incompleto_2026`, `pct_sem_religiao_2022` e `pct_indigenas_2022` agora
entram como candidatas, e a variação do comparecimento (`delta_comp`) também. Vale aqui uma regra:
nada do 2º turno de 2026 entra como covariável, porque é desconhecido antes da apuração. As covariáveis do 1º turno de 2026 (`lag1t`, `outros_1t`) entram: são entradas conhecidas antes do 2º turno. Por isso o
comparecimento é o do 1º turno de 2026, e a variação de comparecimento é a do 2º turno de 2018
para 2022.

**5. Que modelo usar (Partes 5 a 7).** Entre as cinco ligações da binomial, o logit venceu. A
variação entre municípios é maior do que a binomial supõe: φ ≈ 3,07, ou seja, os
erros-padrão ficam 1,75 vez maiores do que a binomial diz. A beta-binomial absorve essa
variação extra: o desvio-padrão dos resíduos quantílicos cai de 1,66 para 0,98, quase o ideal de
1. Testamos também quase-binomial, GLMM por UF e regressão beta. Fixar o voto de 2022 como offset
foi rejeitado: o teste da razão de verossimilhanças contra a defasagem livre dá 10.803 com 1 gl
([03](docs/pressupostos/03-modelos-candidatos.md)).

**6. Juntando categorias (Parte 8.1).** Com testes de razão de verossimilhanças e uma regra
fixada antes de ver os resultados:

* **regiões:** no teste de fusão da Parte 8.1 só Nordeste e Sudeste não se distinguem (p = 0,085), e a fusão vale: o Nordeste entra na referência Sudeste na média (três dummies de região); na dispersão as cinco regiões ficam separadas;
* **prefeito de centro e município sem prefeito** não se distinguem da direita, e a regra os
  funde nela; só o bloco de esquerda fica separado (p = 6e-4). A fusão vale para a seleção
  de variáveis (`bb_selecao`) e para o modelo final ([03](docs/pressupostos/03-modelos-candidatos.md));
* **prefeito eleito em eleição suplementar** não acrescenta nada (p = 0,495) e ficou de fora.

**7. Escolhendo as variáveis: 23 → 18.** A seleção passo a passo por BIC (para trás e nos dois sentidos) chegou ao
mesmo conjunto: o voto de 2022 e mais 17 covariáveis, entre elas `lag1t`. Saíram `outros_1t`, `pct_16_24`,
`pct_analfabeto`, `pct_sem_religiao` e `pct_indigenas` (o passo para frente guarda `pct_analfabeto` e tira `log_pop`). Outros critérios ficaram com mais variáveis
(além do `lag`): AIC com 20; Hosmer-Lemeshow com 21 no primeiro estágio e 20 depois do teste de Wald do passo 4, que
tirou só `pct_sem_religiao` (p = 0,191). Como todos erram quase o mesmo na validação (de 0,213 a 0,242 p.p.),
ficamos com o do BIC (`bb_selecao`).

**Funil das variáveis contínuas** (o `lag` conta em todas as linhas):

| Etapa | Contínuas | Fonte |
| --- | --- | --- |
| Originais na base | 26 | `results/tables/vif_covariaveis.csv` (28 linhas, menos os 2 blocos) |
| Após o VIF | 22 | mesma tabela, coluna `VIF_reduzida` |
| Candidatas dos modelos | 23 | `CONTINUAS` em `src/especificacoes.py` |
| Selecionadas por BIC | 18 | `results/tables/selecao_variaveis.json` |

**8. Conferindo o modelo (Parte 10).** Na beta-binomial, as cidades grandes deixam de dominar o
ajuste: sem os 10 municípios mais influentes, os coeficientes mudam no máximo 0,29 erro-padrão,
contra 2,9 na quase-binomial. Mas os resíduos de vizinhos continuam parecidos (I de Moran
≈ 0,30), e isso não some ([04](docs/pressupostos/04-diagnostico.md)).

**9. Validação e escolha (Parte 11).** O teste principal tira uma UF inteira do ajuste, prevê
essa UF e repete isso para as 27 (LOSO). O erro típico é de cerca de 0,2 ponto percentual por UF.
O menor erro foi o de `bb_selecao` (0,213 p.p., EP 0,035), e quatro dos 18 modelos ficaram a menos
de 1 erro-padrão dele (limite de 0,248; o segundo é `bin_logit`, 0,242). Pela regra fixada antes,
escolhemos o mais simples entre eles, que é o próprio `bb_selecao`, com 24 parâmetros (0,250 quando a seleção de variáveis é refeita dentro de cada dobra, a
medida sem otimismo). A escolha vem da simplicidade, não de uma vantagem demonstrada ([05](docs/pressupostos/05-validacao-cruzada.md)).

**10. Refinando (Partes 13 a 15).**

* **Dispersão:** varia por região. O Nordeste é o mais disperso (θ ≈ 4.647) e o Sul o mais homogêneo
  (θ ≈ 28.930), e o teste confirma com folga (Wald robusto, p = 5,6e-13, 4 gl); o tamanho do município
  (`log n`) foi testado e não entra (p = 0,278);
* **filtro de autovetores:** rejeitado. Melhora o ajuste, mas erra 0,480 p.p. numa UF nova (limite 0,290), porque
  o padrão espacial de uma UF não vale para outra;
* **SLX** (a média das covariáveis dos vizinhos): aceito, com erro de 0,194 p.p., dentro do limite
  de 0,290. Mas a correlação residual caiu pouco (Moran de 0,283 para 0,270);
* **erros-padrão de Conley** a 150 km: ficam em mediana 1,3 vez maiores, e 3 dos 46 termos deixam
  de ser significativos ([07](docs/pressupostos/07-dispersao-regional.md), [08](docs/pressupostos/08-dependencia-espacial.md)).

**11. O resultado (Parte 16).** O modelo final (`bb_slx`, 46 parâmetros) previu 51,79% para Lula
no Brasil, contra 51,78% reais. O erro médio por região foi de 0,15 p.p.: o maior foi no Sudeste
(+0,26) e o menor no Sul (+0,05). As 5 regiões ficaram dentro dos intervalos de 95% e 25 das 27 UFs
(92,6%) também; essa cobertura é otimista, porque parte da incerteza foi estimada nas mesmas UFs
([09](docs/pressupostos/09-modelo-final.md)).

**12. O que não funcionou (Parte 17).** Tentamos remover a correlação dos resíduos com krigagem e
com CAR/BYM. Nenhum dos dois passou da regra (queda mínima de 0,05 no I de Moran fora da amostra): a krigagem
baixou 0,013 e o CAR 0,036. Os resíduos de outras UFs não ajudam a prever uma UF nova
([10](docs/pressupostos/10-extensoes-rejeitadas.md)).

**13. Prevendo durante a apuração (Parte 18).** Ajustamos o modelo só com as urnas já apuradas,
na ordem real do TSE, e previmos o restante. O desenho do modelo (covariáveis, fusão de regiões,
dispersão, SLX, corte de Conley e σ_UF) é escolhido de novo em cada corte, só com os municípios já
apurados; assim o teste de 2026 não usa o resultado final de 2026. O voto do 1º turno do mesmo ano
(`lag1t`, `outros_1t`) é medido nas urnas apuradas e no restante, e entra em todos os cortes. O modelo foi útil em 2026 (6 de 6
cortes) e não em 2022 (3 de 6: com 25%, 50% e 67% apurado, o swing do 1º turno acerta o Brasil por menos). Com 25% apurado em 2026, errou 0,09 p.p. por região, contra 0,46 do
swing uniforme e 0,42 do swing do 1º turno. Com 5% apurado, o erro regional (0,16) fica abaixo do swing (1,49) e do swing do 1º turno (0,52) e o erro
do Brasil (0,16 p.p.) fica abaixo do da contagem bruta (0,97). Em 2022, os cortes de 5% e
10% erram 0,08 e 0,13 p.p. no Brasil. O ajuste de 2022 não tem `delta_comp`: o análogo exigiria o 2º turno de 2014, que não
temos, e usar o valor do próprio ano seria vazamento ([11](docs/pressupostos/11-apuracao-parcial.md)).

### O que isso diz sobre 2022 e sobre eleições futuras parecidas

Esta é a nossa leitura dos resultados, não um teste feito:

* **A eleição anterior é o melhor ponto de partida, mas não se repete igual.** Há regressão à
  média, maior quando o candidato muda (inclinação 0,80 de 2018 para 2022) do que quando ele se
  repete (0,96 de 2022 para 2026). Numa disputa com os mesmos grupos, o voto anterior com peso
  estimado é a base mais forte;
* **a binomial pura subestima a incerteza.** A variação extra entre municípios (φ ≈ 21) e a
  diferença de dispersão entre regiões apareceram nos dois anos e devem aparecer de novo;
* **a vizinhança importa e não sai dos resíduos.** Os intervalos precisam de erros-padrão
  espaciais (Conley) e de um choque por UF. Sem isso, só 21 das 27 UFs ficam dentro do intervalo;
* **na apuração, o modelo ajuda desde as primeiras urnas.** Com o voto do 1º turno do mesmo ano como
  entrada, ele vence a contagem bruta, o swing uniforme e o swing do 1º turno nos 6 cortes de 2026, mas só em 3 dos 6 de 2022
  (mínimo exigido: 4 de 6). O Brasil e as 5 regiões ficaram dentro do intervalo de 95% em todos os 12 cortes;
* **para usar o modelo noutra eleição, é preciso reajustá-lo.** Aplicado sem recalibrar de um
  turno para o outro, ele errou o nível nacional em −4,7 p.p. (no backtest, depois removido). As
  covariáveis também precisam ser anteriores ao turno: por isso o ajuste de 2022 na apuração não
  usa a variação de comparecimento;
* **ele fala de municípios, não de eleitores** (veja a falácia ecológica em
  [Limitações](#limitações)).

---

## 🧪 Metodologia

O desenvolvimento seguiu três notebooks. Cada parte tem seu documento em [`docs/pressupostos/`](docs/pressupostos/README.md).

### 1. Análise exploratória (`01-AED`, Partes 1 a 4)

* Dados, resposta e covariáveis; mapas coropléticos;
* A resposta de 2026 contra 2022 tem inclinação 0,96 (OLS), o que justifica estimar o coeficiente da defasagem em vez de fixá-lo em 1;
* Colinearidade (VIF) das covariáveis.

### 2. Modelos candidatos e diagnóstico (`02-MLG`, Partes 5 a 10)

* Binomial com cinco ligações, quase-binomial (φ de Pearson próprio), beta-binomial, GLMM com PQL próprio, regressão beta e linhas de base (OLS e WLS sobre o logit empírico);
* Defasagem livre, offset, spline, defasagem por região e efeito de UF;
* Diagnóstico: resíduos quantílicos, envelope meio-normal, alavanca e distância de Cook, I de Moran; nos modelos escolhidos (`bb_selecao` e `bb_slx`), também teste de ligação, linearidade por covariável, VIF/GVIF da matriz final e Cook de um passo;
* Parte 8: categóricas por nível (bloco do prefeito, regiões fundidas quando o teste não as separa) e seleção de variáveis por BIC.

### 3. Validação cruzada e regra de seleção (`02-MLG`, Parte 11)

Regra fixada antes dos resultados:

1. menor erro absoluto médio da participação por UF no LOSO;
2. regra do 1 erro-padrão: entre os modelos dentro de 1 EP do melhor, o de menos parâmetros;
3. o MAE ponderado por votos do 5-fold é critério secundário.

Sem vazamento: a padronização é ajustada só na dobra de treino. O erro regional LOSO **não** é critério, porque favorece por construção as binomiais canônicas. (Uma versão anterior vazava informação da UF de teste; foi corrigida e as seções afetadas foram removidas.)

### 4. Modelo final (`02-MLG`, Partes 13 a 17)

* **Parte 13, dispersão por região** (`bb_disp`; o log n foi testado e descartado), confirmada por teste de Wald robusto;
* **Parte 14, ESF e SLX:** filtro de autovetores (rejeitado) e SLX (aceito);
* **Parte 15, erros-padrão de Conley** a 150 km;
* **Parte 16, estimativas regionais e por UF** do modelo final e cobertura dos intervalos;
* **Parte 17, extensões rejeitadas:** krigagem LOSO dos resíduos e CAR/BYM.

### 5. Apuração parcial (`03-apuracao-parcial`, Parte 18)

Estimativa do resultado com parte das urnas apuradas (cortes de 5% a 90%), comparada à extrapolação bruta, ao swing e ao swing do 1º turno.

---


## 🛠️ Tecnologias utilizadas

* **Python** (≥ 3.12) e **uv**
* **Jupyter Notebook**
* **Pandas**, **NumPy** e **PyArrow**
* **Plotly** (gráficos e mapas, exportados em PNG com kaleido)
* **SciPy**
* **Statsmodels** (GLM, beta-binomial)
* **Scikit-learn**
* **GeoPandas** e **geobr** (malha municipal)
* **PySAL** (`libpysal`, `esda`)
* **pytest**

---

## 📁 Estrutura do projeto

```text
.
├── src/          # coleta (src/coleta/), especificações dos modelos, modelos próprios, backtest
├── tests/        # testes unitários
├── notebooks/    # 01-AED, 02-MLG, 03-apuracao-parcial
├── data/         # raw (cache por fonte), interim (um parquet por fonte), processed, referencia
├── results/      # cv, figures, tables
├── docs/         # dados (fontes) e pressupostos (decisões e resultados)
└── README.md
```

### `src/`

Os notebooks não definem funções: toda função auxiliar fica em `src/`, e a célula mostra só o passo estatístico.

```text
src/
├── coleta/          # um módulo por fonte e o consolidador
├── especificacoes.py # candidatos; todo modelo é ajustado por `Candidato.ajustar` (reusado pela Parte 11 do `02-MLG`)
├── modelos.py       # modelos próprios (quase-binomial, beta-binomial, GLMM por PQL)
├── selecao.py       # seleção de variáveis (AIC/BIC, Hosmer-Lemeshow, todos os subconjuntos)
├── pressupostos.py  # diagnóstico dos pressupostos de um modelo já ajustado
├── validacao.py     # validação cruzada (5 dobras e LOSO) e erros por UF e região
├── backtest.py      # funções do backtest 2018 -> 2022 -> 2026 (módulo `BT`, usado por validação e apuração)
├── espacial.py      # autovetores de Moran, Conley, SLX, krigagem e CAR
├── apuracao.py      # apuração parcial e simulação das participações
├── dados.py         # carga comum dos notebooks `02-MLG` e `03-apuracao-parcial`
├── graficos.py      # gráficos Plotly
└── saidas.py        # gravação padronizada de tabelas, figuras e JSON
```

### `data/`

Dados baixados ficam em cache em `data/raw/<fonte>/`; cada fonte grava um `data/interim/<fonte>.parquet`; a consolidação grava `data/processed/municipios.parquet` (5.570 linhas).

### `notebooks/`

Rode `01-AED`, `02-MLG` e `03-apuracao-parcial`, nessa ordem. O `02-MLG` roda de uma vez (Run-All): os valores passam em memória entre as Partes, sem leitura de arquivo de outro notebook. O Passo 6 da Parte 10 roda depois da Parte 15, porque diagnostica `bb_selecao` e `bb_slx`. A Parte 12 (backtest) foi removida; os números das Partes não mudaram.

1. `01-AED.ipynb`: análise exploratória (Partes 1 a 4);
2. `02-MLG.ipynb`: modelos candidatos, diagnóstico, validação cruzada, dispersão, dependência espacial, Conley, estimativas e extensões rejeitadas (Partes 5 a 11 e 13 a 17);
3. `03-apuracao-parcial.ipynb`: apuração parcial (Parte 18).

### `results/`

Previsões fora da amostra (`cv/`), gráficos em PNG (`figures/`, exportados com kaleido) e tabelas em CSV (`tables/`).

### `docs/`

* `docs/dados/`: fontes, regras de consolidação e dicionário;
* `docs/pressupostos/`: decisões, pressupostos e resultados de cada etapa; comece pelo [índice](docs/pressupostos/README.md).

### Reprodução

```bash
uv sync
# Coleta: o crosswalk TSE↔IBGE vem primeiro, porque Bolsa Família e a consolidação dependem dele.
uv run --no-sync python -m src.coleta.ibge_municipios
uv run --no-sync python -m src.coleta.tse_presidente
uv run --no-sync python -m src.coleta.tse_ideologia
uv run --no-sync python -m src.coleta.tse_perfil
uv run --no-sync python -m src.coleta.ibge_censo_pib
uv run --no-sync python -m src.coleta.mds_bolsa_familia
uv run --no-sync python -m src.coleta.consolidar   # grava data/processed/municipios.parquet
uv run --no-sync pytest
uv run --no-sync jupyter lab notebooks/
```

O caminho do repositório tem espaço: coloque-o entre aspas.

---

## 📈 Resultados

Os resultados abaixo são do modelo final `bb_slx` (Partes 13 a 17 do `02-MLG`) e da apuração parcial (Parte 18). Detalhes em [`docs/pressupostos/09-modelo-final.md`](docs/pressupostos/09-modelo-final.md). Todos os números vêm de `results/tables/`.

### Modelo ajustado

$$
Y_i \sim \mathrm{BetaBin}(n_i, \pi_i, \theta_i), \quad
\mathrm{logit}(\pi_i) = \beta_0 + \sum_{k=1}^{18} \beta_k z_{ik} + \sum_{k=1}^{18} \delta_k (Wz)_{ik} + \text{bloco}_i + \text{região}_i, \quad
\log \theta_i = Z_i \gamma
$$

* beta-binomial com ligação logit; 46 parâmetros (41 da média e 5 da dispersão);
* $z$ são as covariáveis padronizadas (o `lag` e as 17 selecionadas); $W$ é a vizinhança Queen padronizada por linha, e $(Wz)_{ik}$ é a média da variável $k$ nos vizinhos (SLX);
* $Z_i$ contém as dummies de região (o $\log n_i$ foi testado e descartado);
* os erros-padrão são os de Conley a 150 km, de 1,0 a 2,3 vezes os do modelo (mediana 1,3).

### Coeficientes (escala z)

Cada covariável contínua entra padronizada (média 0, desvio-padrão 1), então $e^\beta$ é a razão de chances do voto em Lula para **+1 desvio-padrão (DP)** da covariável, mantidos os demais termos. A leitura é sobre a **participação municipal** de Lula nos votos válidos, não sobre o comportamento de um eleitor (ver [Limitações](#limitações)). Os 46 termos vêm de `results/tables/final_ep.csv`; os cinco últimos (γ) são da dispersão, e $e^\gamma$ multiplica θ.

| Termo | β (z) | EP Conley | e^β (OR por 1 DP) | Signif. 5% | Leitura |
| --- | --- | --- | --- | --- | --- |
| intercepto | 0,1940 | 0,0018 | 1,214 | sim | logit de Lula no município de referência (Sudeste e Nordeste, prefeito de direita, covariáveis na média): 54,8% |
| defasagem (logit de 2022) | 0,0994 | 0,0060 | 1,105 | sim | com o 1º turno do mesmo ano no modelo, o voto de 2022 pesa pouco; mantido o resto |
| 1º turno de 2026 (logit de Lula/(Lula+Bolsonaro)) | 0,7300 | 0,0063 | 2,075 | sim | o voto do 1º turno domina: +1 DP multiplica a chance de Lula por 2,08 |
| nota ideológica dos deputados | 0,0021 | 0,0011 | 1,002 | não | sem efeito distinguível de zero, com o EP de Conley |
| % de esquerda na Câmara | −0,0044 | 0,0011 | 0,996 | sim | associação negativa com Lula, mantido o resto |
| comparecimento no 1º turno | −0,0078 | 0,0008 | 0,992 | sim | associação negativa com Lula, mantido o resto |
| variação do comparecimento no 2º turno (2018 a 2022) | −0,0018 | 0,0006 | 0,998 | sim | associação negativa com Lula, mantido o resto |
| log da população | 0,0090 | 0,0010 | 1,009 | sim | associação positiva com Lula, mantido o resto |
| log da densidade | −0,0069 | 0,0012 | 0,993 | sim | associação negativa com Lula, mantido o resto |
| % de mulheres | 0,0081 | 0,0011 | 1,008 | sim | associação positiva com Lula, mantido o resto |
| % de 60 anos ou mais | −0,0061 | 0,0010 | 0,994 | sim | associação negativa com Lula, mantido o resto |
| % com ensino superior | −0,0056 | 0,0010 | 0,994 | sim | associação negativa com Lula, mantido o resto |
| cobertura do Bolsa Família | 0,0066 | 0,0015 | 1,007 | sim | associação positiva com Lula, mantido o resto |
| log do PIB per capita | 0,0009 | 0,0009 | 1,001 | não | sem efeito distinguível de zero, com o EP de Conley |
| % do VAB agropecuário | −0,0044 | 0,0009 | 0,996 | sim | associação negativa com Lula, mantido o resto |
| % de evangélicos | −0,0024 | 0,0009 | 0,998 | sim | associação negativa com Lula, mantido o resto |
| % de pretos e pardos | −0,0029 | 0,0026 | 0,997 | não | sem efeito distinguível de zero, com o EP de Conley |
| % de população urbana | 0,0035 | 0,0011 | 1,004 | sim | associação positiva com Lula, mantido o resto |
| % com até fundamental incompleto | −0,0082 | 0,0014 | 0,992 | sim | associação negativa com Lula, mantido o resto |
| W·defasagem (logit de 2022) | −0,0237 | 0,0096 | 0,977 | sim | vizinhança: associação negativa com Lula, dado o próprio município |
| W·1º turno de 2026 (logit de Lula/(Lula+Bolsonaro)) | 0,0243 | 0,0101 | 1,025 | sim | vizinhança: associação positiva com Lula, dado o próprio município |
| W·nota ideológica dos deputados | 0,0011 | 0,0014 | 1,001 | não | sem efeito distinguível de zero, com o EP de Conley |
| W·% de esquerda na Câmara | −0,0017 | 0,0014 | 0,998 | não | sem efeito distinguível de zero, com o EP de Conley |
| W·comparecimento no 1º turno | 0,0033 | 0,0011 | 1,003 | sim | vizinhança: associação positiva com Lula, dado o próprio município |
| W·variação do comparecimento no 2º turno (2018 a 2022) | −0,0030 | 0,0011 | 0,997 | sim | vizinhança: associação negativa com Lula, dado o próprio município |
| W·log da população | −0,0064 | 0,0014 | 0,994 | sim | vizinhança: associação negativa com Lula, dado o próprio município |
| W·log da densidade | 0,0079 | 0,0020 | 1,008 | sim | vizinhança: associação positiva com Lula, dado o próprio município |
| W·% de mulheres | 0,0033 | 0,0015 | 1,003 | sim | vizinhança: associação positiva com Lula, dado o próprio município |
| W·% de 60 anos ou mais | 0,0014 | 0,0019 | 1,001 | não | sem efeito distinguível de zero, com o EP de Conley |
| W·% com ensino superior | −0,0024 | 0,0020 | 0,998 | não | sem efeito distinguível de zero, com o EP de Conley |
| W·cobertura do Bolsa Família | 0,0119 | 0,0028 | 1,012 | sim | vizinhança: associação positiva com Lula, dado o próprio município |
| W·log do PIB per capita | 0,0077 | 0,0019 | 1,008 | sim | vizinhança: associação positiva com Lula, dado o próprio município |
| W·% do VAB agropecuário | −0,0016 | 0,0015 | 0,998 | não | sem efeito distinguível de zero, com o EP de Conley |
| W·% de evangélicos | −0,0052 | 0,0013 | 0,995 | sim | vizinhança: associação negativa com Lula, dado o próprio município |
| W·% de pretos e pardos | −0,0061 | 0,0033 | 0,994 | não | sem efeito distinguível de zero, com o EP de Conley |
| W·% de população urbana | 0,0042 | 0,0016 | 1,004 | sim | vizinhança: associação positiva com Lula, dado o próprio município |
| W·% com até fundamental incompleto | −0,0013 | 0,0021 | 0,999 | não | sem efeito distinguível de zero, com o EP de Conley |
| prefeito de esquerda | −0,0049 | 0,0018 | 0,995 | sim | prefeito de esquerda (contra direita): chance menor |
| Centro-Oeste | −0,0095 | 0,0045 | 0,991 | sim | contra Sudeste e Nordeste: chance menor |
| Norte | −0,0141 | 0,0046 | 0,986 | sim | contra Sudeste e Nordeste: chance menor |
| Sul | −0,0331 | 0,0039 | 0,967 | sim | contra Sudeste e Nordeste: chance menor |
| γ intercepto (Sudeste) | 9,4388 | 0,1169 | 12.567 (fator em θ) | sim | log θ do Sudeste: θ ≈ 12.567 |
| γ Centro-Oeste | 0,3967 | 0,2585 | 1,487 (fator em θ) | não | θ ≈ 18.687; sem diferença distinguível do Sudeste |
| γ Nordeste | −1,0495 | 0,1991 | 0,350 (fator em θ) | sim | θ ≈ 4.400 (região mais dispersa) |
| γ Norte | −0,2132 | 0,2711 | 0,808 (fator em θ) | não | θ ≈ 10.154; sem diferença distinguível do Sudeste |
| γ Sul | 0,7972 | 0,2076 | 2,219 (fator em θ) | sim | θ ≈ 27.890 (a mais homogênea) |

**Ressalva de colinearidade.** No `bb_slx`, onze termos têm VIF > 10: `W·lag1t` (251,8), `W·defasagem` (229,0), `lag1t` (114,4), a defasagem (110,5), `W·pct_pretos_pardos` (23,8), `pct_pretos_pardos` (18,4), `W·cobertura_pbf` (18,1), `W·pct_fund_inc` (14,5), `W·log_dens` (11,8), `pct_fund_inc` (10,2) e `log_dens` (10,0). Os dois logits (2022 e 1º turno de 2026) e as suas vizinhanças medem quase o mesmo voto, e cada covariável compartilha informação com a própria defasagem espacial, então os coeficientes desses termos, lidos um a um, são frágeis; a previsão não é afetada ([diagnóstico](docs/pressupostos/04-diagnostico.md), seção 8.6).

Leitura geral: o voto do 1º turno do mesmo ano (`lag1t`, OR 2,08 por DP) explica quase tudo e o de 2022 pesa pouco depois dele (OR 1,10); as demais covariáveis ajustam o nível em poucos pontos percentuais, e a vizinhança (termos `W·`) tem efeitos menores e menos estáveis, o que é coerente com o SLX ter reduzido pouco o I de Moran. A variação do comparecimento (`delta_comp`) tem efeito pequeno, mas distinguível de zero com o EP de Conley.

### Superdispersão

```text
φ de Pearson (binomial logit) = 3,07
beta-binomial: θ ≈ 7.632, ρ ≈ 0,00013
DP dos resíduos quantílicos: 0,98 (beta-binomial) contra 1,66 (binomial)
```

Com $n_i$ de milhares, a binomial pura subestima a variância por um fator da ordem de 3; por isso os EPs do binomial puro não valem.

### Seleção de variáveis

O passo a passo por BIC (backward e both) dá o mesmo modelo: 17 contínuas além da defasagem, BIC 62.571,43, LOSO 0,213 (0,250 refazendo a seleção em cada dobra). O forward pára em 18 variáveis (BIC 62.575,35) e a busca em todos os subconjuntos ficou em BIC 62.571,70, sem alcançar esse ótimo.

### Seleção por validação cruzada (LOSO, MAE por UF em p.p.)

```text
melhor              bb_selecao = 0,213 (EP 0,035); segundo bin_logit = 0,242 (EP 0,041)
limite de 1 EP      0,248   (4 de 18 modelos dentro)
escolhido           bb_selecao = 0,213 (0,250 aninhado; k = 24, menor número de parâmetros dentro do limite)
limite de aceitação L = 0,2497 (aninhado) + 0,0407 = 0,2904
bb_disp = 0,218 (aceito)   bb_slx = 0,194 (aceito, final)   bb_esf = 0,480 (rejeitado)
```

Leitura: o MAE de 0,2 p.p. diz que, em média, a participação de Lula prevista para uma UF inteira erra cerca de 0,2 ponto percentual. A previsão não usa nenhum município dessa UF no ajuste. O erro é por UF porque a UF inteira sai do treino em cada dobra do LOSO.

O `bb_slx` tem MAE por UF de 0,140 no 5-fold e 0,194 no LOSO.

### Dispersão regional

```text
Wald robusto (Conley, 300 km): dummies de região p = 5,6e-13 -> mantidas; log n p = 0,278 -> descartado
θ do Sudeste ≈ 12.521; Nordeste ≈ 4.647 (o mais disperso); Sul ≈ 28.930 (o mais homogêneo)
```

Leitura: a dispersão varia por região (o Nordeste tem θ menor, isto é, mais variância extra-binomial) e não depende de $n$ depois de separar as regiões.

### Dependência espacial

```text
I de Moran dos resíduos: 0,283 (sem termo espacial) -> 0,270 (SLX)
filtro de autovetores (k = 25): LOSO 0,480 > 0,2904 -> rejeitado
krigagem LOSO dos resíduos (queda de 0,013 no I de Moran) e CAR/BYM (queda de 0,036): rejeitadas
```

O SLX é aceito pela regra de validação, mas quase não reduz o I de Moran: a autocorrelação espacial residual persiste e os EPs de Conley corrigem a inferência, não as estimativas pontuais ([`docs/pressupostos/08-dependencia-espacial.md`](docs/pressupostos/08-dependencia-espacial.md), [`docs/pressupostos/10-extensoes-rejeitadas.md`](docs/pressupostos/10-extensoes-rejeitadas.md)).

### Estimativas regionais contra o real

Participação de Lula (%) na estimativa fora da amostra (LOSO) com Conley e choque de UF $t$ de Student ($\sigma_{UF}$ = 0,0114); intervalo de 95%:

| Região | Estimado | Intervalo 95% | Real | Erro (p.p.) | Dentro |
| --- | --- | --- | --- | --- | --- |
| Centro-Oeste | 40,13 | [39,77; 40,47] | 39,99 | 0,14 | sim |
| Nordeste | 69,55 | [69,33; 69,77] | 69,37 | 0,19 | sim |
| Norte | 49,91 | [49,57; 50,26] | 50,01 | −0,10 | sim |
| Sudeste | 47,50 | [47,11; 47,91] | 47,24 | 0,26 | sim |
| Sul | 37,47 | [37,15; 37,80] | 37,42 | 0,05 | sim |
| Brasil | 51,78 | [51,60; 51,97] | 51,78 | 0,00 | sim |

Erro regional absoluto médio de 0,145 p.p. nas 5 regiões. As estimativas ficam dentro de 0,4 p.p. do real em todas as regiões e o Brasil dentro de 0,01 p.p.; o maior erro é o do Sudeste (+0,26). São só 5 regiões, e o erro regional LOSO não escolhe modelos: é descrição, não evidência de superioridade.

### Cobertura dos intervalos

| Covariância + σ | UF 95% | Região 95% |
| --- | --- | --- |
| Conley + σ_UF | 92,6% (25/27) | 5/5 |
| Conley + σ = 0 | 77,8% (21/27) | 3/5 |

O choque de UF ($t$ de Student, $\sigma_{UF}$ = 0,0114) melhora a cobertura a 95% de 77,8% para 92,6% das UFs e de 3/5 para 5/5 regiões; a 90%, a das UFs vai de 74,1% a 88,9%. Com $\sigma_{UF}$ aninhado (a UF pontuada usa o σ das outras 26; a região, o das UFs fora dela), a cobertura das UFs é a mesma (25/27 a 95%, 88,9% a 90%), as regiões cobrem 5/5 a 95% e 80% a 90%, com largura de 1,30 p.p. por UF e 0,66 por região. O que resta de otimismo é o tamanho da amostra: 5 regiões e 27 UFs são poucas para estimar cobertura real.

### Apuração parcial

Regra: o modelo é "útil" se vence a extrapolação bruta, o swing e o swing do 1º turno em pelo menos 4 de 6 cortes de urnas apuradas (5% a 90%).

* 2022: 3/6 cortes, "modelo não útil" (nos cortes de 25%, 50% e 67% o swing do 1º turno erra menos o Brasil); 2026: 6/6, "modelo útil". O veredito é o mesmo com as duas definições de $n_u$ (votos restantes);
* o Brasil fica dentro do intervalo de 95% nos 12 cortes (2022 e 2026) e as regiões em 60 de 60. O desenho é reescolhido por corte só com os municípios apurados, e o voto do 1º turno do mesmo ano estreita os intervalos: a largura média é 0,39 p.p. no Brasil (0,39 em 2022 e 0,38 em 2026) e 0,55 p.p. nas regiões;
* o ganho já aparece em 5% das urnas: em 2026 com 5% apurado, o erro regional médio é 0,16 p.p. no modelo contra 1,49 no swing, e o erro do Brasil é 0,16 p.p. no modelo contra 0,97 na contagem bruta; com 25% o erro regional é 0,09 contra 0,46. Em 2022, os cortes de 5% e 10% erram 0,08 e 0,13 p.p. no Brasil; a partir de 50% as estimativas quase não mudam ([`docs/pressupostos/11-apuracao-parcial.md`](docs/pressupostos/11-apuracao-parcial.md)).

## Limitações

> [!WARNING]
> **Falácia ecológica (Robinson, 1950).** O modelo descreve **municípios**, não eleitores. "Municípios com mais evangélicos votam menos em Lula" **não implica** que evangélicos votem menos em Lula: a associação agregada pode existir sem que ela valha para o indivíduo (ou com sinal oposto). Os coeficientes acima são associações entre municípios, condicionais às demais variáveis, e não efeitos causais nem comportamento individual.

Demais limitações:

* **Autocorrelação espacial residual** (I de Moran 0,270): Conley corrige a inferência, não as estimativas pontuais;
* **Cobertura com poucas unidades:** sem $\sigma_{UF}$, 21 das 27 UFs ficam dentro do intervalo; com ele (também aninhado) são 25 de 27, e 5 regiões e 27 UFs são poucas para estimar cobertura;
* **Seleção:** 4 dos 18 modelos ficam dentro de 1 EP, e a escolha repousa na parcimônia, não numa vantagem demonstrada;
* **Pouca base regional:** só 5 regiões e 27 UFs para avaliar o erro; o erro regional não escolhe modelo;
* **Regras tardias:** o Wald robusto, o SLX e o corte de Conley foram fixados antes de cada resultado, mas não antes das Partes 13 a 17 inteiras; uma versão anterior da validação vazava informação da UF de teste (corrigida);
* **Covariáveis defasadas:** o comparecimento é o do 1º turno e nenhuma covariável do 2º turno entra; `pct_vab_agro` usa 2021;
* **Dados:** Bolsa Família 2022 com 20,07 milhões de famílias, 2,81% abaixo dos 20,65 milhões da folha de setembro de 2022 (Ministério da Cidadania, nota de 16/09/2022, republicada pelo TNH1; o original no gov.br não foi encontrado; a base é família na folha, não NIS distinto pago); a API do MDS está indisponível e a fonte é a CGU;
* **Apuração parcial:** o ganho sobre a contagem bruta e o swing vem do voto do 1º turno do mesmo ano, que é conhecido antes do 2º turno; 12 cortes (6 por ano) e 2 eleições são pouca base para estimar a regra de utilidade, e o mínimo de 4 em 6 foi fixado antes da execução;
* **Modelos de 2022 e 2026 diferem na apuração:** o ajuste parcial de 2022 não tem `delta_comp` nem `w_delta_comp`. O análogo em 2022 exigiria o 2º turno de 2014, que não temos, e usar o valor do próprio ano seria vazamento; o de 2026 mantém o modelo completo.

Pendências abertas, com efeito e correção possível: [`docs/pressupostos/12-pendencias.md`](docs/pressupostos/12-pendencias.md).

## Explicar × prever

Os 5.570 municípios são um censo, não uma amostra. O ajuste com todos os dados serve para interpretar os coeficientes e fazer o diagnóstico. Ele não mede a previsão: com ligação canônica e dummies de região, o modelo reproduz os totais regionais dentro da amostra.

A previsão é medida fora da amostra de duas formas:

* **LOSO:** a UF inteira sai do ajuste e recebe previsão só com covariáveis anteriores ao turno. A seleção de variáveis é refeita em cada dobra;
* **Apuração parcial (Parte 18):** o modelo é ajustado só nos municípios já apurados, na ordem real do TSE, e o desenho (covariáveis, fusão de regiões, dispersão, SLX, Conley e σ_UF) é reescolhido em cada corte só com eles. Ele prevê o restante e a previsão é comparada com o resultado final, com a contagem bruta, com o swing uniforme e com o swing uniforme do 1º turno.

Não se sorteia uma amostra aleatória de municípios, por dois motivos:

1. Os votos não chegam ao acaso: capitais e seções urbanas são apuradas primeiro;
2. Municípios vizinhos são correlacionados (I de Moran dos resíduos). Com sorteio, vizinhos ficariam no treino e no teste, e o erro seria subestimado.

Referência: Shmueli, G. (2010). To Explain or to Predict? *Statistical Science*, 25(3), 289–310.

---


## 📚 Referências

* Bolognesi, B.; Ribeiro, E.; Codato, A. (2023). Uma nova classificação ideológica dos partidos políticos brasileiros. *Dados*, 66(2). https://doi.org/10.1590/dados.2023.66.2.303
* Paula, G. A. (2013). *Modelos de Regressão com Apoio Computacional*. IME-USP. https://www.ime.usp.br/~giapaula/texto_2013.pdf
* McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models* (2ª ed.). Chapman & Hall.
* Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics*, 31(4), 949-952 (beta-binomial).
* Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
* Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
* Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
* Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
* Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review*, 15(3), 351-357.
* Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
* Tiefelsdorf, M.; Griffith, D. A. (2007). Semiparametric filtering of spatial autocorrelation: the eigenvector approach. *Environment and Planning A*, 39(5), 1193-1221.
* Breslow, N. E.; Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *Journal of the American Statistical Association*, 88(421), 9-25.
* Rey, S. J.; Anselin, L. (2010). PySAL: a Python library of spatial analytical methods. In *Handbook of Applied Spatial Analysis*. Springer, 175-193.
* TSE. Dados abertos (resultados, perfil do eleitorado e candidaturas). https://dadosabertos.tse.jus.br
* IBGE. Censo Demográfico 2022 e PIB dos Municípios. https://www.ibge.gov.br
* CGU. Portal da Transparência (Auxílio Brasil e Bolsa Família). https://portaldatransparencia.gov.br

Lista completa e conferida: [`docs/pressupostos/referencias.md`](docs/pressupostos/referencias.md).


---

## 👥 Autores

**José de Ribamar Ferreira Neto**

**Matheus Finger Ramos de Oliveira**

---

## 🎓 Contexto acadêmico

Projeto desenvolvido como parte das atividades da disciplina de **Modelagem Estatística**, com foco na aplicação de **Modelos Lineares Generalizados (MLGs)**, inferência estatística e dependência espacial.

