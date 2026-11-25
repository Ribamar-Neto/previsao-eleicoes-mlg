# Apuração parcial na ordem real do TSE (Parte 18)

`notebooks/03-apuracao-parcial.ipynb`. Tabelas `results/tables/apuracao_*.csv` e figuras
`results/figures/apuracao_*.png`. Cenário A do [09](09-modelo-final.md). Fonte da ordem:
`tse_bweb` ([consolidação](../dados/consolidacao.md)).

**Os modelos de 2022 e de 2026 diferem em um termo.** `delta_comp` (e o `w_delta_comp` do SLX) é a variação do comparecimento do 2º turno de 2018 para 2022; seu análogo em 2022 exigiria o 2º turno de 2014 (ausente), e usar `delta_comparecimento_2t_2022` em 2022 seria vazamento. Por isso o ajuste parcial de 2022 descarta os dois termos (`SEM_2022`), a mesma regra de `VARS_EXANTE` em `src/backtest.py`; o de 2026 mantém o modelo completo ([06](06-backtest.md)).

## 1. Pergunta e regra (fixada antes do resultado)

Com x % dos votos válidos recebidos, em que ordem real de chegada das seções, quanto dá o total
do Brasil e de cada região no 2º turno? Anos: 2022 (Lula x Bolsonaro) e 2026 (Lula x Flávio).

Cortes: 5, 10, 25, 50, 67 e 90 % dos votos válidos recebidos no país. O corte é o primeiro
instante de recebimento cuja fração acumulada chega a x %; seções do mesmo instante entram
juntas, então o percentual efetivo fica um pouco acima de x.

O modelo é útil em um ano se, em pelo menos 4 dos 6 cortes, ele tem **ao mesmo tempo** erro
absoluto do Brasil e MAE regional menores que os da contagem bruta e os do swing uniforme.
Veredito por ano; a regra não muda depois de ver os números.

## 2. Métodos comparados

- **Bruta:** parcela apurada da região (ou do Brasil) tomada como resultado final.
- **Swing uniforme:** variação média nacional, ponderada, entre a parcela apurada e a do turno
  anterior (2018 para 2022; 2022 para 2026), aplicada à parcela anterior de cada município,
  limitada a [0, 1].
- **Modelo:** `bb_slx` do [09](09-modelo-final.md), ajustado nos municípios com votos já apurados
  (padronização só nessas linhas) e usado para prever o restante de cada município.

Regionais e Brasil somam o apurado com o restante estimado. Alvo: resultado final do 2º turno.

## 3. Votos do restante (n_u)

n_u = aptos das seções ainda não recebidas x (válidos / aptos do 1º turno do mesmo município e
ano). É uma definição **pré-turno**: no 2º turno os votos dos demais candidatos migram, então o
total de válidos do 1º turno estima o total do 2º melhor do que só os dois finalistas.

Leitura literal (só os votos dos dois finalistas no 1º turno) versus todos os válidos, erro
percentual absoluto médio por município contra o n real do restante:

| Ano | Dois candidatos | Todos os válidos | Soma estimada / real (todos) |
|---|---|---|---|
| 2022 | 6,56 % | 2,10 % | 0,997 |
| 2026 | 9,47 % | 4,10 % | 0,962 |

**Declaração de honestidade:** a razão a priori é a da migração, mas a escolha entre as duas
leituras foi feita olhando o erro contra o n final. n_u ainda subestima o restante em 0,3 a 0,5 % (2022)
e 3,9 a 4,8 % (2026), conforme o corte. O veredito da regra é o mesmo nas duas
definições de n_u (2022 4 de 6; 2026 5 de 6).

**Incerteza de n_u nos intervalos (seção 6).** O centro de n_u não muda (estimativas e regra de
utilidade iguais); só os intervalos o sorteiam. Regra fixada antes do resultado, usando só as
seções já recebidas (sem dado do resultado final): por município, r = n_c / (aptos recebidos x
taxa), válido onde n_c > 0 e o denominador > 0; R_UF e R_BR são o log da razão das somas na UF e no
país; sd_UF é o desvio de R_UF - R_BR entre UFs; o resíduo municipal é e = log r - R_UF. Em cada
simulação, n_u do município é multiplicado por k = exp(choque da UF ~ N(0, sd_UF²) + e sorteado
com reposição da UF, centrado pela mediana do pool). UF com menos de 5 municípios válidos sorteia
do pool nacional. k tem centro 1 em log; o desvio entre UFs capta o erro de nível regional da taxa
e o resíduo, o erro municipal. O n_u simulado entra nos votos e no denominador do share; θ
continua avaliado no n_u fixo. Vale também para municípios fora do modelo.

## 4. Resultados

Erro absoluto do Brasil / MAE regional (p.p.). Em negrito, o menor de cada coluna do corte.

| Ano | Corte | Bruta | Swing | Modelo |
|---|---|---|---|---|
| 2022 | 5 | 4,57 / **0,77** | 0,60 / 3,45 | **0,34** / 0,98 |
| 2022 | 10 | 3,05 / **0,52** | 1,53 / 3,43 | **0,74** / 0,83 |
| 2022 | 25 | 2,20 / 0,74 | 1,62 / 2,88 | **0,67** / **0,52** |
| 2022 | 50 | 1,23 / 0,66 | 1,09 / 1,81 | **0,52** / **0,40** |
| 2022 | 67 | 0,93 / 0,60 | 0,61 / 1,16 | **0,33** / **0,25** |
| 2022 | 90 | 0,38 / 0,31 | 0,13 / 0,34 | **0,09** / **0,08** |
| 2026 | 5 | **0,97** / 1,78 | 1,19 / **1,49** | 1,11 / 1,62 |
| 2026 | 10 | 1,48 / 1,04 | 0,53 / 1,00 | **0,14** / **0,92** |
| 2026 | 25 | 2,37 / 0,83 | 0,39 / 0,46 | **0,23** / **0,27** |
| 2026 | 50 | 1,67 / 0,60 | 0,53 / 0,33 | **0,15** / **0,14** |
| 2026 | 67 | 0,95 / 0,49 | 0,40 / 0,22 | **0,08** / **0,08** |
| 2026 | 90 | 0,22 / 0,15 | 0,14 / 0,07 | **0,02** / **0,03** |

## 5. Veredito pela regra

| Ano | Cortes vencidos | Mínimo | Veredito |
|---|---|---|---|
| 2022 | 4 de 6 (25, 50, 67, 90) | 4 | modelo útil |
| 2026 | 5 de 6 (10 a 90) | 4 | modelo útil |

O modelo vence o erro do Brasil em 11 dos 12 cortes (em 2026 a 5 %, 1,11 contra 0,97 da bruta; antes vencia nos 12). Perde o MAE regional nos cortes iniciais
(2022 a 5 e 10 %; 2026 a 5 %). Os vereditos não mudaram, mas a margem de 2022 é a mínima da regra: a conclusão de 2022 é frágil.

## 6. Intervalos

Restante simulado pelo beta-binomial com grupos por UF e covariância de Conley (200 km),
σ_UF = 0,0558 (Parte 16; antes das covariáveis novas, 0,0564), B = 1000, semente 7; soma ao apurado por região. Cobertura nominal 95 %
(5 regiões + Brasil, 6 itens por corte):

| Corte | 5 | 10 | 25 | 50 | 67 | 90 |
|---|---|---|---|---|---|---|
| 2022 | 0,83 | 1,00 | 1,00 | 1,00 | 1,00 | 0,83 |
| 2026 | 0,33 | 0,67 | 1,00 | 1,00 | 0,83 | 0,83 |

O Brasil fica dentro do intervalo em 11 dos 12 cortes (fora em 2026 a 5 %, onde o modelo também erra mais). Largura do Brasil: no máximo 2,2 p.p. (2022 a 5 %; 1,9 em 2026) e no mínimo
0,23 p.p. (2022 a 90 %).

**Cobertura por unidade e ano, n_u fixo contra n_u simulado** (cortes cobertos de 6; variação mediana da largura em p.p.;
`results/tables/apuracao_cobertura_unidade.csv`). Os 6 cortes são aninhados no tempo e as unidades são correlacionadas:
a tabela é descritiva, não um teste de calibração, e não se soma entre unidades.

| Unidade | 2022 fixo | 2022 simulado | Δ largura | 2026 fixo | 2026 simulado | Δ largura |
|---|---|---|---|---|---|---|
| Brasil | 6 | 6 | +0,028 | 5 | 5 | +0,043 |
| Centro-Oeste | 6 | 6 | -0,012 | 5 | 5 | -0,021 |
| Nordeste | 6 | 6 | +0,018 | 4 | 4 | +0,015 |
| Norte | 4 | 4 | +0,005 | 2 | 1 | +0,009 |
| Sudeste | 6 | 6 | -0,019 | 6 | 6 | -0,074 |
| Sul | 6 | 6 | +0,060 | 6 | 6 | +0,033 |

Resultado: a incerteza de n_u quase não muda os intervalos (|Δ| < 0,08 p.p. contra larguras de 0,23 a 2,2 p.p.); só o Norte
de 2026 perde um corte coberto ao sortear n_u (2 para 1). sd_UF no corte de 50 %: 0,031 (2022) e 0,044 (2026). A cobertura
baixa do Norte (2026: 2 de 6 cortes) e do Nordeste (2026: 4 de 6; em 2022 passou de 3 para 6)
vem de outra fonte que n_u (θ, σ_UF reusado, representatividade, seção 7). A contagem agregada antiga (0,83/1,00/... por corte)
foi substituída por esta tabela.

Os votos já apurados dos municípios fora do modelo entram uma vez só, no apurado; o termo fixo
desses municípios é só π_UF x n_u. Antes eram somados duas vezes (só Boa Esperança do Norte, MT,
em 2026): os limites de Centro-Oeste e Brasil em 2026 mudaram até 0,01 p.p. e nenhuma cobertura mudou.

## 7. Representatividade das seções apuradas

Pressuposto: as seções já apuradas representam as do mesmo município que faltam. Teste: lacuna
entre a parcela apurada (ou π do modelo) e a parcela real do restante, ponderada pelos votos
reais do restante, nos 20 maiores municípios (capitais e grandes cidades) contra os demais.

- Os 20 maiores concentram 24 % a 46 % do restante nacional real.
- Lacuna absoluta do apurado: 0,7 a 4,4 p.p. nos maiores e 2,8 a 6,9 p.p. nos demais (2026 a
  10 %: apurado 3,6 p.p. abaixo do restante nos maiores). Com o modelo: 2026, 0,5 a 1,4 p.p. nos
  maiores e 1,6 a 2,5 nos demais; 2022, 1,3 a 1,6 e 2,4 a 2,9.
- Em 2026 o modelo reduz a lacuna nos dois grupos na maioria dos cortes; em 2022 reduz nos demais, mas não nos maiores (1,3 a 1,6 p.p. contra 0,7 a 2,1 do apurado). A diferença interna ao
  município (seção rural contra urbana) só é corrigida em parte pelas covariáveis.

## 8. Limitações

- σ_UF = 0,0558 (Parte 16) e o corte de Conley (200 km, Parte 15) vêm de 2026 e foram reusados em 2022, que não tem
  LOSO próprio. [assumed] que valham; a cobertura menor de 2022 nos cortes tardios sugere que não.
- σ_UF e o corte também foram ajustados nas mesmas 27 UFs ([12](12-pendencias.md)).
- A dispersão por log(n) é ajustada com n apurado e usada para prever com n_u (descasamento):
  θ é avaliado em log(n_u), fora da faixa de log n do ajuste; o erro de θ é ignorado (só o
  bloco β da covariância de Conley é sorteado); σ_UF da Parte 16 é reusado, não reestimado
  por corte ([12](12-pendencias.md)).
- n_u é sorteado nos intervalos (seção 3), mas θ segue avaliado no n_u fixo. A cobertura é
  reportada por unidade e ano, de forma descritiva (seção 6).
- Escolha do n_u feita olhando o n final (seção 3).
- Boa Esperança do Norte (TSE 73709) existe só nos dados do bweb de 2026; entra no apurado e no
  alvo, e seu restante usa a parcela e a taxa de válidos no nível da UF.
- Sem voto no exterior nas seções do modelo; é tratado como no restante do trabalho.
- Só dois anos e seis cortes: poucos graus de liberdade para a regra de 4 de 6.
