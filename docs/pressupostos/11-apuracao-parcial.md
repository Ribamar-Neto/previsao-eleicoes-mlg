# Apuração parcial na ordem real do TSE (Parte 18)

`notebooks/03-apuracao-parcial.ipynb`. Tabelas `results/tables/apuracao_*.csv` e figuras
`results/figures/apuracao_*.png`. Família `bb_slx` do [09](09-modelo-final.md), com o desenho refeito em cada corte (seção 9). Fonte da ordem:
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
- **Modelo:** família `bb_slx` do [09](09-modelo-final.md), com o **desenho escolhido em cada corte**
  só nos municípios com votos já apurados (seção 9), ajustado nessas linhas (padronização só nelas)
  e usado para prever o restante de cada município.

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
| 2022 | 5 | 4,57 / **0,77** | **0,60** / 3,45 | 0,72 / 1,35 |
| 2022 | 10 | 3,05 / **0,52** | 1,53 / 3,43 | **1,19** / 1,05 |
| 2022 | 25 | 2,20 / 0,74 | 1,62 / 2,88 | **0,57** / **0,57** |
| 2022 | 50 | 1,23 / 0,66 | 1,09 / 1,81 | **0,53** / **0,41** |
| 2022 | 67 | 0,93 / 0,60 | 0,61 / 1,16 | **0,33** / **0,27** |
| 2022 | 90 | 0,38 / 0,31 | 0,13 / 0,34 | **0,09** / **0,08** |
| 2026 | 5 | **0,97** / 1,78 | 1,19 / 1,49 | 1,02 / **1,36** |
| 2026 | 10 | 1,48 / 1,04 | 0,53 / 1,00 | **0,16** / **0,83** |
| 2026 | 25 | 2,37 / 0,83 | 0,39 / 0,46 | **0,16** / **0,24** |
| 2026 | 50 | 1,67 / 0,60 | 0,53 / 0,33 | **0,13** / **0,15** |
| 2026 | 67 | 0,95 / 0,49 | 0,40 / 0,22 | **0,07** / **0,07** |
| 2026 | 90 | 0,22 / 0,15 | 0,14 / 0,07 | **0,03** / **0,03** |

Fonte: `results/tables/apuracao_2022.csv` e `apuracao_2026.csv`.

## 5. Veredito pela regra

| Ano | Cortes vencidos | Mínimo | Veredito |
|---|---|---|---|
| 2022 | 4 de 6 (25, 50, 67, 90) | 4 | modelo útil |
| 2026 | 5 de 6 (10 a 90) | 4 | modelo útil |

O modelo vence a bruta no erro do Brasil em 11 dos 12 cortes (perde só em 2026 a 5 %, 1,02 contra 0,97). No MAE
regional perde para a bruta em 2022 a 5 e 10 % e vence nos 6 cortes de 2026 (4 de 6 em 2022). Em 2022 a 5 % perde
também o erro do Brasil para o swing (0,72 contra 0,60). Os vereditos não mudaram com o desenho por corte (seção 9), mas
a margem de 2022 é a mínima da regra: a conclusão de 2022 é frágil.

## 6. Intervalos

Restante simulado pelo beta-binomial com grupos por UF e covariância de Conley, com o corte de Conley (1 a 200 km)
e o σ_UF (2022: 0,106 a 0,133; 2026: 0,052 a 0,137) **do desenho de cada corte** (seção 9), B = 1000, semente 7; soma ao
apurado por região. Cobertura nominal 95 % (5 regiões + Brasil, 6 itens por corte):

| Corte | 5 | 10 | 25 | 50 | 67 | 90 |
|---|---|---|---|---|---|---|
| 2022 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 0,83 |
| 2026 | 0,83 | 1,00 | 1,00 | 0,83 | 0,83 | 0,83 |

O Brasil fica dentro do intervalo nos 12 cortes. As regiões (5 por corte) cobrem 29 de 30 em 2022 e 27 de 30 em 2026
(56 de 60). Largura média do Brasil: 2,18 p.p. em 2022 e 1,47 em 2026; máxima 3,45 (2022 a 10 %) e 3,53 (2026 a 5 %);
mínima 0,42 (2022 a 90 %) e 0,23 (2026 a 90 %). Largura média regional: 3,14 (2022) e 2,21 (2026).

**Cobertura por unidade e ano, n_u fixo contra n_u simulado** (cortes cobertos de 6; variação mediana da largura em p.p.;
`results/tables/apuracao_cobertura_unidade.csv`). Os 6 cortes são aninhados no tempo e as unidades são correlacionadas:
a tabela é descritiva, não um teste de calibração, e não se soma entre unidades.

| Unidade | 2022 fixo | 2022 simulado | Δ largura | 2026 fixo | 2026 simulado | Δ largura |
|---|---|---|---|---|---|---|
| Brasil | 6 | 6 | +0,037 | 6 | 6 | +0,073 |
| Centro-Oeste | 6 | 6 | -0,014 | 6 | 6 | +0,007 |
| Nordeste | 6 | 6 | +0,015 | 6 | 5 | -0,019 |
| Norte | 5 | 5 | +0,062 | 3 | 3 | +0,057 |
| Sudeste | 6 | 6 | +0,046 | 6 | 6 | +0,035 |
| Sul | 6 | 6 | -0,033 | 6 | 6 | +0,023 |

Resultado: a incerteza de n_u muda uma só contagem (Nordeste 2026: 6 com n_u fixo, 5 com n_u simulado) e quase não muda os intervalos (|Δ| < 0,08 p.p.
contra larguras de 0,23 a 3,5 p.p.). sd_UF no corte de 50 %: 0,031 (2022) e 0,044 (2026). A cobertura baixa do Norte
(2026: 3 de 6 cortes; 2022: 5) vem de outra fonte que n_u (θ, representatividade, seção 7).

Os votos já apurados dos municípios fora do modelo entram uma vez só, no apurado; o termo fixo
desses municípios é só π_UF x n_u. Antes eram somados duas vezes (só Boa Esperança do Norte, MT,
em 2026): os limites de Centro-Oeste e Brasil em 2026 mudaram até 0,01 p.p. e nenhuma cobertura mudou.

## 7. Representatividade das seções apuradas

Pressuposto: as seções já apuradas representam as do mesmo município que faltam. Teste: lacuna
entre a parcela apurada (ou π do modelo) e a parcela real do restante, ponderada pelos votos
reais do restante, nos 20 maiores municípios (capitais e grandes cidades) contra os demais.

- Os 20 maiores concentram 24 % a 46 % do restante nacional real.
- Lacuna absoluta do apurado: 0,7 a 4,4 p.p. nos maiores e 2,8 a 6,9 p.p. nos demais (2026 a
  10 %: apurado 3,6 p.p. abaixo do restante nos maiores). Com o modelo: 2026, 0,8 a 1,9 p.p. nos
  maiores e 1,6 a 2,5 nos demais; 2022, 1,2 a 3,1 e 2,4 a 2,8.
- O modelo reduz a lacuna nos demais municípios nos 12 cortes. Nos maiores reduz em 4 de 6 cortes de 2026 e em
  1 de 6 de 2022 (1,2 a 3,1 p.p. contra 0,7 a 2,1 do apurado). A diferença interna ao
  município (seção rural contra urbana) só é corrigida em parte pelas covariáveis.

## 8. Limitações

- O desenho de cada corte vem de poucos municípios nos cortes iniciais (1,4 mil a 2,1 mil de 5,5 mil), com LOSO
  em 24 a 26 UFs; σ_UF e o corte de Conley têm mais ruído ali ([12](12-pendencias.md)).
- **Corte de referência de 200 km (substitui o piso de 1 km).** Quando a regra da Parte 15 dá 0 km num desenho parcial, o
  corte é 200 km: a regra aplicada à apuração completa de 2022, conhecida antes do 2º turno de 2026 (sem vazamento).
  Razão: nenhuma faixa com correlação ≥ 0,05 numa amostra apurada pequena indica poucos pares próximos, não ausência de
  dependência; com 1 km o Conley seria HC0. Vale para R1, Wald A/B e corte final. Só 2026 a 5 % (corte final: 1 para
  200 km) e a 10 % (R1 e Wald; o corte final segue 50 km) usaram a regra. A 5 % de 2026 a largura do IC95 (p.p.) foi de
  3,79 para 3,53 no Brasil, 6,77 para 6,53 no Centro-Oeste, 4,10 para 4,32 no Nordeste (agora dentro do IC), 5,50 para
  5,50 no Norte, 8,29 para 7,77 no Sudeste e 6,23 para 6,37 no Sul: o corte maior não alargou todos os IC, porque com o
  núcleo de Bartlett os termos cruzados de municípios próximos podem ser negativos [deduced].
- A dispersão por log(n) é ajustada com n apurado e usada para prever com n_u (descasamento):
  θ é avaliado em log(n_u), fora da faixa de log n do ajuste; o erro de θ é ignorado (só o
  bloco β da covariância de Conley é sorteado) ([12](12-pendencias.md)).
- n_u é sorteado nos intervalos (seção 3), mas θ segue avaliado no n_u fixo. A cobertura é
  reportada por unidade e ano, de forma descritiva (seção 6).
- Escolha do n_u feita olhando o n final (seção 3).
- Boa Esperança do Norte (TSE 73709) existe só nos dados do bweb de 2026; entra no apurado e no
  alvo, e seu restante usa a parcela e a taxa de válidos no nível da UF.
- Sem voto no exterior nas seções do modelo; é tratado como no restante do trabalho.
- Só dois anos e seis cortes: poucos graus de liberdade para a regra de 4 de 6.

## 9. Desenho escolhido em cada corte (vazamento corrigido)

**O vazamento.** A versão anterior da Parte 18 usava o desenho final do [09](09-modelo-final.md) (covariáveis,
regiões fundidas, dispersão, SLX, corte de Conley e σ_UF), escolhido no `02-MLG` com o resultado final de 2026. As
estimativas parciais de 2026 já traziam esse resultado. O teste de 2022 sempre foi honesto: o desenho nunca viu 2022.
O vazamento favorecia 2026.

**A correção ("correção completa por corte", escolha do usuário em 2026-11-25).** Em cada (ano, corte) o desenho é
refeito só com D, os municípios com votos apurados (alvo = parcela apurada). As regras R1 a R7 (`src/desenho_parcial.py`,
`VERSAO = 1`) foram fixadas antes de ver o resultado:

| Regra | Decisão e razão |
|---|---|
| R1 | Covariáveis e fusão de regiões por LRT + BIC passo a passo, com as mesmas funções da seleção aninhada do 02: mesmo critério, só muda o conjunto de dados. Padronização só em D; 2022 sem `delta_comp` (parágrafo inicial). |
| R2 | Corte de Conley pela regra da Parte 15, nos resíduos do desenho final do corte; corte de 200 km quando a regra dá 0 (o ajuste exige um corte positivo; seção 8). |
| R3 | LOSO em D (UFs com 5 ou mais municípios apurados, alvo = parcela apurada da UF). L = MAE do R1 + erro-padrão: mesmo limite da seleção aninhada. |
| R4 | Dispersão pela regra da Parte 13 (Wald de Conley com o corte do próprio ajuste), aceita só se o LOSO fica abaixo de L. |
| R5 | SLX pela regra da Parte 14 (Moran cai), aceito só se o LOSO fica abaixo de L. Em 2026 a 50 e 67 % e em 2022 a 67 % o MAE com SLX ficou acima do MAE sem SLX; [fact: `src/desenho_parcial.py`, R5] o SLX foi aceito porque a regra exige queda do I de Moran e LOSO ≤ L, não queda do MAE. |
| R6 | ESF, krigagem e CAR não são retestados: o 02 os rejeitou e eles só acrescentariam termos. |
| R7 | σ_UF = RMS dos erros em logit do LOSO do desenho final do corte (cai de 0,137 em 2026 a 5 % para 0,052 a 90 %; 2022: 0,133 a 0,106). |

**Falhas.** Uma dobra com `LinAlgError` ou separação perfeita é pulada e registrada; um corte sem desenho conta como
falha do modelo na regra de utilidade (seção 1). Nenhuma UF foi pulada nos 12 cortes e disp e SLX foram aceitos em todos.

**Entradas.** O 03 não lê mais `final_decisao.json` nem `final_cobertura.csv`; lê só a base municipal (covariáveis
pré-turno) e `tse_bweb`. O desenho de cada corte fica em `results/tables/apuracao_desenho.csv`.

**Por que manter o modelo por corte.** Ele responde à pergunta da noite da apuração: o que se pode saber naquele
momento, só com o que já foi contado. Um desenho fixo escolhido depois do resultado não responde a isso.

**Efeito do vazamento (antes → depois).**

| Medida | Antes | Depois |
|---|---|---|
| Brasil dentro do IC (12 cortes) | 11 | 12 |
| Regiões dentro do IC (60 itens) | 50 | 56 |
| Largura média do IC do Brasil, 2022 / 2026 (p.p.) | 1,29 / 1,15 | 2,18 / 1,47 |
| Vitórias no MAE regional contra bruta e swing, 2022 / 2026 | 4 de 6 / 5 de 6 | 4 de 6 / 6 de 6 |
| 2022 a 5 %: erro do Brasil (p.p.) | 0,324 | 0,725 |
| 2022 a 10 %: erro do Brasil (p.p.) | 0,723 | 1,187 |
| 2026 a 5 %: erro do Brasil / largura do IC (p.p.) | 1,119 / 1,85 | 1,022 / 3,53 |

- [fact] De 50 % em diante as estimativas mudaram no máximo 0,010 p.p. (2022 a 90 %: 0,094 → 0,086).
- [fact] A 5 % de 2026 o modelo ainda perde para a bruta no Brasil (1,022 contra 0,965 p.p.); a perda vem do viés de
  Centro-Oeste (erro regional +2,36) e Nordeste (+2,20, agora dentro do IC), não do vazamento: o erro mudou só 0,097 p.p.
- [deduced] Os erros baixos de 2022 a 5 e 10 % vinham do desenho escolhido com o resultado final. [assumed] Parte pode
  ser acaso de um só ano.
- [deduced] O IC largo vem de σ_UF e da dispersão do desenho do corte. [assumed] A causa de fundo do viés a 5 %
  (composição da apuração inicial, que não cobre Centro-Oeste e Nordeste) não foi testada.
