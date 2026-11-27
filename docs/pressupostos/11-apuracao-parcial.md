# Apuração parcial na ordem real do TSE (Parte 18)

`notebooks/03-apuracao-parcial.ipynb`. Tabelas `results/tables/apuracao_*.csv` e figuras
`results/figures/apuracao_*.png`. Família `bb_slx` do [09](09-modelo-final.md), com o desenho refeito em cada corte (seção 9) e a covariável do 1º turno do mesmo ano (seção 10). Fonte da ordem:
`tse_bweb` ([consolidação](../dados/consolidacao.md)).

**Os modelos de 2022 e de 2026 diferem em um termo.** `delta_comp` (e o `w_delta_comp` do SLX) é a variação do comparecimento do 2º turno de 2018 para 2022; seu análogo em 2022 exigiria o 2º turno de 2014 (ausente), e usar `delta_comparecimento_2t_2022` em 2022 seria vazamento. Por isso o ajuste parcial de 2022 descarta os dois termos (`SEM_2022`), a mesma regra de `VARS_EXANTE` em `src/backtest.py`; o de 2026 mantém o modelo completo ([06](06-backtest.md)).

## 1. Pergunta e regra

Com x % dos votos válidos recebidos, em que ordem real de chegada das seções, quanto dá o total
do Brasil e de cada região no 2º turno? Anos: 2022 (Lula x Bolsonaro) e 2026 (Lula x Flávio).

Cortes: 5, 10, 25, 50, 67 e 90 % dos votos válidos recebidos no país. O corte é o primeiro
instante de recebimento cuja fração acumulada chega a x %; seções do mesmo instante entram
juntas, então o percentual efetivo fica um pouco acima de x.

O modelo vence um corte se tem **ao mesmo tempo** erro absoluto do Brasil e MAE regional menores que os de **todas**
as linhas de base: contagem bruta, swing uniforme e swing uniforme do 1º turno do mesmo ano (`swing_1t`). É útil em um
ano com vitória em pelo menos 4 dos 6 cortes. Veredito por ano.

**O `swing_1t` entrou depois de ver o resultado.** A regra nasceu com duas linhas de base (bruta e swing). Como o
modelo usa o 1º turno do mesmo ano (seção 10), a comparação justa exige uma linha de base que também o use; com ela a
regra ficou mais exigente. A mudança não foi fixada antes do resultado e o leitor deve pesar isso. Com as duas linhas
originais o veredito era útil nos dois anos (6 de 6).

## 2. Métodos comparados

- **Bruta:** parcela apurada da região (ou do Brasil) tomada como resultado final.
- **Swing uniforme:** variação média nacional, ponderada, entre a parcela apurada e a do turno
  anterior (2018 para 2022; 2022 para 2026), aplicada à parcela anterior de cada município,
  limitada a [0, 1].
- **Swing do 1º turno (`swing_1t`):** variação nacional ponderada entre o 1º turno do mesmo ano e a parcela apurada do
  2º turno, aplicada ao 1º turno do restante de cada município. Usa a mesma informação do 1º turno que o modelo.
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
e 3,9 a 4,8 % (2026), conforme o corte. Sensibilidade (`apuracao_sensibilidade_nu.csv`): com a leitura literal o
veredito de 2022 é o mesmo (3 de 6, não útil) e o de 2026 é 5 de 6 (perde o Brasil aos 10 %), ainda útil. A conclusão não
depende da taxa.

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

| Ano | Corte | Bruta | Swing | Swing 1º turno | Modelo |
|---|---|---|---|---|---|
| 2022 | 5 | 4,57 / 0,77 | 0,60 / 3,45 | 0,39 / 0,39 | **0,08** / **0,14** |
| 2022 | 10 | 3,05 / 0,52 | 1,53 / 3,43 | 0,24 / 0,24 | **0,13** / **0,10** |
| 2022 | 25 | 2,20 / 0,74 | 1,62 / 2,88 | **0,13** / 0,13 | 0,15 / **0,10** |
| 2022 | 50 | 1,23 / 0,66 | 1,09 / 1,81 | **0,08** / 0,07 | 0,10 / **0,06** |
| 2022 | 67 | 0,93 / 0,60 | 0,61 / 1,16 | **0,06** / 0,05 | 0,07 / **0,04** |
| 2022 | 90 | 0,38 / 0,31 | 0,13 / 0,34 | 0,022 / 0,015 | **0,022** / **0,011** |
| 2026 | 5 | 0,97 / 1,78 | 1,19 / 1,49 | 0,27 / 0,52 | **0,16** / **0,16** |
| 2026 | 10 | 1,48 / 1,04 | 0,53 / 1,00 | 0,23 / 0,50 | **0,11** / **0,13** |
| 2026 | 25 | 2,37 / 0,83 | 0,39 / 0,46 | 0,20 / 0,42 | **0,08** / **0,09** |
| 2026 | 50 | 1,67 / 0,60 | 0,53 / 0,33 | 0,14 / 0,27 | **0,04** / **0,04** |
| 2026 | 67 | 0,95 / 0,49 | 0,40 / 0,22 | 0,11 / 0,17 | **0,02** / **0,02** |
| 2026 | 90 | 0,22 / 0,15 | 0,14 / 0,07 | 0,04 / 0,04 | **0,00** / **0,01** |

Fonte: `apuracao_2022.csv` e `apuracao_2026.csv`. Média dos 12 cortes (Brasil + MAE regional): modelo
0,157 p.p. (0,081 + 0,075); swing do 1º turno 0,394 (0,160 + 0,234); antes do 1º turno do mesmo ano (seção 10), o
modelo tinha 0,951 (0,415 + 0,536). Em 2022 a 90 % o modelo vence o swing do 1º turno no Brasil por 0,0003 p.p. (0,0219
contra 0,0222), um empate prático. No MAE regional o modelo vence o swing do 1º turno nos 12 cortes.

## 5. Veredito pela regra

| Ano | Cortes vencidos | Mínimo | Veredito |
|---|---|---|---|
| 2022 | 3 de 6 | 4 | modelo não útil |
| 2026 | 6 de 6 | 4 | modelo útil |

2022: o modelo perde no erro do Brasil aos 25, 50 e 67 % para o `swing_1t` (0,15 contra 0,13; 0,10 contra 0,08; 0,07
contra 0,06 p.p.) e vence o MAE regional nos 6 cortes. Portanto, em 2022 a vantagem do modelo sobre o swing do 1º turno é
regional, não nacional. 2026: vence os 12 critérios dos 6 cortes. Contra a bruta e o swing uniforme o modelo vence os 12
cortes nos dois critérios (`results/tables/apuracao_regra_cortes.csv`, `apuracao_regra.csv`). Antes do `swing_1t` o
veredito era útil nos dois anos (6 de 6; no HEAD sem o 1º turno, 4 de 6 e 5 de 6). Que o `swing_1t` explique a derrota de
2022 é uma leitura dos números [deduced]: o diagnóstico não isola a causa ([14](14-apuracao-diagnostico-1t.md), seção 5).

## 6. Intervalos

Restante simulado pelo beta-binomial com grupos por UF e covariância de Conley, com o corte de Conley (75 a 150 km)
e o σ_UF (2022: 0,018 a 0,023; 2026: 0,011 a 0,027) **do desenho de cada corte** (seção 9), B = 1000, semente 7; soma ao
apurado por região. Cobertura nominal 95 % (5 regiões + Brasil, 6 itens por corte): **todos os 72 itens (12 cortes x 6)
ficam dentro do IC**, Brasil em 12 de 12 e regiões em 60 de 60 (antes 56 de 60).

**Choque de UF: σ̂·T_k, T de Student com k graus de liberdade, k = número de UFs que estimam σ̂; no 03, as UFs do desenho do corte**
(`gl_sigma_uf`, 24 a 26). Razão: σ̂ é a RMS de k resíduos de UF de média zero; tratar σ̂ como conhecido estreita o intervalo. É a mesma
regra do 02 (lá, 27 UFs), e a fórmula de variância do modelo usa σ²·k/(k−2). Fundamento: distribuição preditiva t com variância
desconhecida (Gelman et al. 2013, seção 3.2; [referências](referencias.md)). O efeito foi alargar um pouco o IC sem mudar nenhuma contagem
de cobertura: a largura média do Brasil foi de 0,37 e 0,36 (choque normal; execução de 2e1f30d não guardada) para 0,39 e 0,38 p.p.

Largura média do Brasil: 0,39 p.p. em 2022 e 0,38 em 2026; máxima 0,66 (2022 a 5 %) e 0,75 (2026 a 10 %); mínima 0,07
nos dois anos (a 90 %). Largura média regional: 0,56 (2022) e 0,54 (2026). Antes eram 1,82 (Brasil) e 2,67 (regiões)
em média dos 12 cortes, contra 0,39 e 0,55 agora. Com 72 de 72 dentro e IC 5 vezes mais estreito, a cobertura de 95 %
nominal pode estar acima da real [assumed: a cobertura não foi calibrada; 72 de 72 não separa 95 % de valores bem maiores].

**Cobertura por unidade e ano, n_u fixo contra n_u simulado** (cortes cobertos de 6; variação mediana da largura em p.p.;
`results/tables/apuracao_cobertura_unidade.csv`). Os 6 cortes são aninhados e as unidades são correlacionadas:
a tabela é descritiva, não um teste de calibração.

| Unidade | 2022 fixo | 2022 simulado | Δ largura | 2026 fixo | 2026 simulado | Δ largura |
|---|---|---|---|---|---|---|
| Brasil | 6 | 6 | +0,058 | 6 | 6 | +0,149 |
| Centro-Oeste | 6 | 6 | −0,007 | 6 | 6 | +0,010 |
| Nordeste | 6 | 6 | +0,018 | 6 | 6 | +0,020 |
| Norte | 6 | 6 | +0,102 | 6 | 6 | +0,144 |
| Sudeste | 6 | 6 | +0,009 | 6 | 6 | +0,046 |
| Sul | 6 | 6 | +0,050 | 6 | 6 | +0,104 |

Resultado: a incerteza de n_u não muda nenhuma contagem (todas 6 de 6) e alarga os intervalos em até 0,15 p.p. (mediana),
o que é maior em termos relativos agora que o IC é estreito (Brasil 2026: +0,15 sobre largura média de 0,38).
sd_UF de n_u no corte de 50 %: 0,031 (2022) e 0,044 (2026).

Os votos já apurados dos municípios fora do modelo entram uma vez só, no apurado; o termo fixo
desses municípios é só π_UF x n_u. Antes eram somados duas vezes (só Boa Esperança do Norte, MT,
em 2026): os limites de Centro-Oeste e Brasil em 2026 mudaram até 0,01 p.p. e nenhuma cobertura mudou.

## 7. Representatividade das seções apuradas

Pressuposto: as seções já apuradas representam as do mesmo município que faltam. Teste: lacuna
entre a parcela apurada (ou π do modelo) e a parcela real do restante, ponderada pelos votos
reais do restante, nos 20 maiores municípios (capitais e grandes cidades) contra os demais.

- Os 20 maiores concentram 24 % a 46 % do restante nacional real.
- Lacuna absoluta do apurado: 0,7 a 4,4 p.p. nos maiores e 2,8 a 6,9 p.p. nos demais (2026 a
  10 %: apurado 3,6 p.p. abaixo do restante nos maiores). Com o modelo: 0,3 a 0,7 p.p. nos maiores e 0,5 a 0,6 nos
  demais em 2026; 0,4 a 0,5 e 0,4 a 0,5 em 2022.
- O modelo reduz a lacuna absoluta nos dois grupos nos 12 cortes (nos maiores em 2022: 0,4 a 0,5 p.p. contra 0,7 a 2,1
  do apurado). A diferença interna ao município (seção rural contra urbana) é medida à parte com o 1º turno por seção
  ([14](14-apuracao-diagnostico-1t.md), D4).

## 8. Limitações

- O desenho de cada corte vem de poucos municípios nos cortes iniciais (1,4 mil a 2,1 mil de 5,5 mil), com LOSO
  em 24 a 26 UFs; σ_UF e o corte de Conley têm mais ruído ali ([12](12-pendencias.md)).
- **Corte de referência de 200 km (substitui o piso de 1 km).** Quando a regra da Parte 15 dá 0 km num desenho parcial, o
  corte é 200 km: a regra aplicada à apuração completa de 2022, conhecida antes do 2º turno de 2026 (sem vazamento).
  Razão: nenhuma faixa com correlação ≥ 0,05 numa amostra apurada pequena indica poucos pares próximos, não ausência de
  dependência; com 1 km o Conley seria HC0. Vale para R1, Wald A/B e corte final. Com o 1º turno do mesmo ano nenhum
  dos 12 desenhos precisou da regra (cortes de 75 a 150 km; `piso_corte_aplicado` falso em todos).
- Se um desenho adotar dispersão por log(n), θ ajustado com n apurado é avaliado em log(n_u), fora da faixa
  do ajuste, e seu erro é ignorado ([12](12-pendencias.md)). Nos 12 desenhos atuais não se aplica: a dispersão é
  `const` ou `regiao`. O erro de θ é ignorado em todos (só o bloco β da covariância de Conley é sorteado).
- n_u é sorteado nos intervalos (seção 3). A cobertura é reportada por unidade e ano, de forma descritiva (seção 6).
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
`VERSAO = 5`: a versão acompanha revisões posteriores, ver abaixo) foram fixadas antes de ver o resultado:

| Regra | Decisão e razão |
|---|---|
| R1 | Covariáveis e fusão de regiões por LRT + BIC passo a passo, com as mesmas funções da seleção aninhada do 02: mesmo critério, só muda o conjunto de dados. Padronização só em D; 2022 sem `delta_comp` (parágrafo inicial). |
| R2 | Corte de Conley pela regra da Parte 15, nos resíduos do desenho final do corte; corte de 200 km quando a regra dá 0 (o ajuste exige um corte positivo; seção 8). |
| R3 | LOSO em D (UFs com 5 ou mais municípios apurados, alvo = parcela apurada da UF). L = MAE do R1 + erro-padrão: mesmo limite da seleção aninhada. |
| R4 | Dispersão pela regra da Parte 13 (Wald de Conley com o corte do próprio ajuste), aceita só se o LOSO fica abaixo de L. θ no limite binomial: ver o teto de θ abaixo. |
| R5 | SLX pela regra da Parte 14 (Moran cai), aceito só se o LOSO fica abaixo de L. A regra exige queda do I de Moran e LOSO ≤ L, não queda do MAE; por isso um SLX pode ser aceito com MAE acima do do desenho sem ele [fact: `src/desenho_parcial.py`, R5]. |
| R6 | ESF, krigagem e CAR não são retestados: o 02 os rejeitou e eles só acrescentariam termos. |
| R7 | σ_UF = RMS dos erros em logit do LOSO do desenho final do corte (com o 1º turno, 2022: 0,018 a 0,023; 2026: 0,011 a 0,027; sem ele, 0,052 a 0,137). Os k graus de liberdade do choque t (seção 6) são as UFs desse LOSO. |

**Falhas.** Uma dobra com `LinAlgError` ou separação perfeita é pulada e registrada; um corte sem desenho conta como
falha do modelo na regra de utilidade (seção 1). Nenhuma UF foi pulada nos 12 cortes. Com o 1º turno do mesmo ano, a
dispersão foi rejeitada nos cortes de 5 e 10 % dos dois anos (beta-binomial simples, sem θ por região) e o SLX em 2022 a 5 e
10 % e em 2026 a 5, 10 e 25 % (`results/tables/apuracao_desenho.csv`).

**Teto de θ (versão 5 do desenho).** Quando o θ beta-binomial de um grupo de dispersão vai ao limite binomial (θ ≥ 1e5
em todas as linhas do grupo), ele não é identificado: o contraste de Wald de θ é sem sentido e a verossimilhança é plana
nessa direção. O θ é fixado em `THETA_TETO` = 1e9, com teste da razão de verossimilhanças como guarda, e sai dos
contrastes de Wald e da contagem de parâmetros do BIC. Razão do valor: com n = 7e6 por grupo, o fator de variância
1 + (n − 1)/(1 + θ) com θ = 1e9 fica dentro da tolerância de 1 % adotada, então o teto é indistinguível do binomial. Efeito:
o ajuste de 2026 a 10 % (Centro-Oeste), que antes parava fora do ótimo numa crista plana (|g| 0,73), converge
(|g| 1,4e-5); o Wald A dá p = 0,0708 e a decisão continua dispersão única ([14](14-apuracao-diagnostico-1t.md),
seção 6). Todos os 12 desenhos foram refeitos na versão 5.

**Entradas.** O 03 não lê mais `final_decisao.json` nem `final_cobertura.csv`; lê só a base municipal (covariáveis
pré-turno) e `tse_bweb`. O desenho de cada corte fica em `results/tables/apuracao_desenho.csv`.

**Por que manter o modelo por corte.** Ele responde à pergunta da noite da apuração: o que se pode saber naquele
momento, só com o que já foi contado. Um desenho fixo escolhido depois do resultado não responde a isso.

**Efeito do vazamento (antes → depois), estado de 2026-11-25, antes do 1º turno do mesmo ano (seção 10).**

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
- Posteriormente, a covariável do 1º turno do mesmo ano trouxe os erros de 2022 e 2026 a 0,2 p.p. ou menos em todos os
  cortes, o que torna esta tabela histórica; o diagnóstico do IC largo e da causa do viés a 5 % está em
  [14](14-apuracao-diagnostico-1t.md).
- Encolher o restante pelo desvio do apurado na UF estreita o IC e perde cobertura em 2026: rejeitado
  ([13](13-apuracao-extensoes-rejeitadas.md)).

## 10. Covariável do 1º turno do mesmo ano (estado atual)

As seções 4 a 8 e a tabela da seção 9 refletem o modelo com `lag1t` e `outros_1t` (1º turno do mesmo ano, medido por
seção nas seções já apuradas e nas restantes). A pergunta, o diagnóstico D1 a D4, a regra de aceitação fixada antes e
o resultado estão em [14](14-apuracao-diagnostico-1t.md). A "Efeito do vazamento" acima é o estado de 2026-11-25.
