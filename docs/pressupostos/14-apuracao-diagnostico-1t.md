# 14 - Apuração parcial: diagnóstico do IC largo e covariável do 1º turno do mesmo ano (Parte 18)

Notebook `notebooks/03-apuracao-parcial.ipynb` (seção de diagnóstico D1 a D4). Tabelas
`results/tables/apuracao_diag_*.csv` e figura `results/figures/apuracao_diagnostico.png`. O resultado entra em
[11](11-apuracao-parcial.md), seção 10. "HEAD" é o modelo antes desta mudança (desenho por corte do
[11](11-apuracao-parcial.md), seção 9, sem o 1º turno do mesmo ano).

## 1. Pergunta e restrição

Por que o IC da apuração parcial era largo e a estimativa tão sensível à região dos municípios já apurados? Melhorar
**sem mudar a metodologia**: MLGs binomiais e beta-binomiais, na linha de Paula (2013). O escopo escolhido foi a
covariável do 1º turno do mesmo ano; nada na família do modelo, na regra de seleção ou no IC mudou.

## 2. Diagnóstico D1 a D4 (antes → depois)

Quatro hipóteses para a largura, testadas uma a uma sobre os 12 cortes (cada uma gera uma tabela em `apuracao_diag_*.csv`).

| Hipótese | O que mede | HEAD | Agora | Leitura |
|---|---|---|---|---|
| D1 variância por componente | Parte do IC vinda do piso binomial, de β̂, do choque de UF (σ_UF), de θ e de n_u | choque de UF explica em média 88 % da variância do Brasil; σ_UF de 0,052 a 0,137; meia-largura média do IC (unidades) 1,49 (2022) e 1,04 (2026) p.p. | UF 66 %, n_u 23 %; σ_UF de 0,011 a 0,027; meia-largura 0,26 nos dois anos, em p.p. | **Causa principal.** O σ_UF absorvia o que o desenho sem o 1º turno não explicava |
| D2 composição do apurado | Fração do final já apurada, municípios com apuração, parcela de capitais | Brasil a 5 % em 2022: 5 % do final, 25 % dos municípios | Igual (a ordem do TSE não mudou) | Propriedade dos dados; não se corrige no modelo |
| D3 extrapolação de n_u | Alavancagem do restante contra o máximo do ajuste (Paula 1.7.2 e 2.8.1) | 4 de 72 casos fora do suporte (2 de 60 nas regiões); erro máximo 2,56 p.p. | 2 de 72 casos fora do suporte (n_u extrapolado ≤ 0,015 %); erro máximo 0,35 p.p. | Marginal em ambos |
| D4 seleção dentro do município | Lacuna do 1º turno entre as seções apuradas e as restantes (logit), por município e por UF | Igual: a lacuna não usa o modelo (votos do 1º turno por seção, ordem do TSE e n_u = aptos pendentes × taxa) | município -0,21 a 0,28 (mediana -0,01); UF -0,47 a 0,19, negativa em 58 de 72 | Medida direta de que o apurado difere do restante |

D1 soma as partes dentro de 10 % da variância total em 90 % das unidades (antes 81 %); o maior componente é a UF em 61
de 72 unidades, n_u em 10 e θ em 1 (antes UF em todas). O choque de UF é σ̂·T_k ([11](11-apuracao-parcial.md), seção 6).
Na coluna HEAD, as larguras, o σ_UF e o erro máximo do D3 estão no git (`2e1f30d~1`); as frações do D1 e a contagem do D3
vêm do diagnóstico atual rodado sobre o modelo anterior (refeito em 2026-11-27; larguras e erro máximo idênticos aos do
git) e não ficam em `results/`.

## 3. Decisão: 1º turno do mesmo ano por seção

- **O que entra.** `lag1t` (logit de Lula / (Lula + Bolsonaro) no 1º turno do mesmo ano, por município) e `outros_1t`
  (parcela dos votos válidos do 1º turno que foi para os demais candidatos). Denominador zero dá logit 0 (50 %) e
  `outros_1t` 0, nunca NaN.
- **Por que é legítimo.** O 1º turno é conhecido antes do 2º, então segue a regra de informação do
  [09](09-modelo-final.md): só covariável pré-turno. No `02-MLG` é um dado de cada município como qualquer outro.
- **Por que mantém a metodologia.** É um acréscimo de colunas ao mesmo MLG; a seleção por BIC, a regra de Conley, a
  dispersão e o SLX continuam as mesmas regras R1 a R7 ([11](11-apuracao-parcial.md), seção 9). Quem escolhe o termo é a
  seleção por BIC, não o analista: o BIC escolhe `lag1t` e não `outros_1t` no `02-MLG` (conjunto selecionado com BIC
  62.571,4 e 18 contínuas com a defasagem; [03](03-modelos-candidatos.md)).
- **Apuração parcial.** O 03 mede o 1º turno por seção, separadamente nas seções já apuradas e nas restantes de cada
  município (bweb do 1º turno, [dicionário](../dados/dicionario.md)). Assim o desvio de seleção dentro do município
  (D4) entra na previsão do restante, e não só o nível do município.
- **Efeito no 02-MLG.** O LOSO por UF do `bb_selecao` cai de 1,005 para 0,213 p.p. (0,250 com a seleção aninhada), o `bb_slx` segue o modelo final
  (0,194) e a fusão Nordeste + Sudeste é mantida; o limite L cai de 1,209 para 0,290. Detalhes em [03](03-modelos-candidatos.md),
  [05](05-validacao-cruzada.md) e [09](09-modelo-final.md).

## 4. Regra de aceitação (fixada antes da rodada) e resultado

Regra fixada antes da rodada, avaliada somando 2022 e 2026 (12 cortes) contra o estado anterior (HEAD):

1. Brasil dentro do IC95 em 12 de 12 cortes e regiões em pelo menos 56 de 60;
2. média dos 12 cortes de (erro do Brasil + MAE regional) menor que a do HEAD;
3. nenhum veredito de utilidade por ano (seção 1 do [11](11-apuracao-parcial.md)) pior.

A largura do IC não entra na regra: a condição 1 impede que um IC mais estreito passe com menos cobertura.

| Medida (2022 + 2026) | HEAD | Agora |
|---|---|---|
| Brasil dentro do IC95 | 12/12 | 12/12 |
| Regiões dentro do IC95 | 56/60 | 60/60 |
| Média dos 12 cortes de (erro do Brasil + MAE regional), p.p. | 0,951 | 0,157 |
| Erro médio do Brasil, p.p. | 0,415 | 0,081 |
| MAE regional médio, p.p. | 0,536 | 0,075 |
| Largura média do IC, Brasil, p.p. | 1,824 | 0,387 |
| Largura média do IC, regiões, p.p. | 2,673 | 0,548 |
| Veredito 2022 (regra com bruta e swing) | útil (4 de 6) | útil (6 de 6) |
| Veredito 2026 (regra com bruta e swing) | útil (5 de 6) | útil (6 de 6) |

**Aceita**: as três condições valem (12/12 e 60/60; 0,157 < 0,951; 6 de 6 nos dois anos, pela regra de então, sem o `swing_1t`; com ele o veredito é 2022 não útil, seção 5). Duas rodadas do 03 (fria e depois com cache) geraram `results/` idênticos (regra de
determinismo em [12](12-pendencias.md)).

Desenho por corte agora: termos do 1º turno em todos os 12 cortes; SLX rejeitado em 2022 a 5 e 10 % e em 2026 a 5, 10 e
25 %; dispersão `regiao` onde aceita (antes `regiao_logn`) e beta-binomial simples em 2022 e 2026 a 5 e 10 %
(`results/tables/apuracao_desenho.csv`, `results/cv/desenho_parcial_*.json`).

## 5. Swing uniforme do 1º turno como linha de base do veredito

Referência: aplicar a variação nacional ponderada entre o 1º turno do mesmo ano e a parcela apurada do 2º turno a cada
município (linhas `tipo=swing_1t` de `results/tables/apuracao_diag_selecao.csv`). **Entra no veredito** como terceira linha
de base ([11](11-apuracao-parcial.md), seção 1): o modelo só vence um corte se o bater nos dois critérios. Resultado:
2022 não útil (3 de 6), 2026 útil (6 de 6).

| Medida | Swing do 1º turno | Modelo |
|---|---|---|
| Erro médio do Brasil, 12 cortes (p.p.) | 0,160 | 0,081 |
| MAE regional médio, 12 cortes (p.p.) | 0,234 | 0,075 |
| Cortes em que o swing tem erro do Brasil menor | 3 de 12 (2022 a 25, 50 e 67 %), por até 0,02 p.p. | nos outros 9 (2022 a 90 % por 0,0003) |
| Cortes em que o swing tem MAE regional menor | 0 de 12 | 12 de 12 |

- O swing de 2022 a 5 % erra 0,39 p.p. no Brasil contra 0,08 do modelo; em 2026 a 5 %, 0,27 contra 0,16.
- O swing do 1º turno é muito melhor que o swing do turno anterior da seção 2 do [11](11-apuracao-parcial.md) (Brasil
  0,13 a 1,62 p.p. em 2022): a maior parte do ganho do modelo vem da própria informação do 1º turno. O modelo só
  acrescenta sobre o swing nas regiões, e em 2022 é isso que separa o veredito "não útil" do "útil" anterior. É uma
  leitura, não um teste de significância [deduced].

## 6. Hessiana analítica do beta-binomial com dispersão regional

- **Achado.** A Hessiana numérica (diferenças finitas da log-verossimilhança somada, n até 1e6) era indefinida em
  2026 a 5 e a 10 % (maior autovalor +14.742 e +0,67; a analítica é negativa definida nos dois): `cov_params`
  falhava e o pão do Conley e os Wald do desenho usavam uma matriz errada. Passou-se a usar a Hessiana analítica
  (trigama); cache do desenho passou da versão 3 para a 4. Os números sem teto (θ 4,3e8, autovalores, |g| 0,73, EPs)
  e o |g| com teto foram refeitos em 2026-11-27 com o código atual e o teto desligado (o erro de 1e-3 e os EPs de
  15 % de 2e1f30d não se reproduziram) e não ficam em `results/`.
- **Dispersão no limite binomial: resolvido com um teto de θ.** Em 2026 a 5 % o θ do Centro-Oeste chega a 4,3e8, o limite
  binomial; nesse ponto θ não é identificado e seu contraste de Wald não tem sentido. Regra (versão 5 do desenho, [11](11-apuracao-parcial.md),
  seção 9): se todas as linhas do grupo têm θ ≥ 1e5, fixa-se θ = 1e9, com guarda por razão de verossimilhanças, fora
  dos Wald e da contagem do BIC. O fator de variância fica dentro da tolerância de 1 % com n = 7e6.
- **Crista plana em 2026 a 10 %: resolvida.** O θ do Centro-Oeste ficava numa crista plana e o otimizador parava fora do
  ótimo (|g| 0,73 contra a tolerância de 1e-3; ao liberá-lo, os EPs Conley de β mudavam até 72 % e os do modelo até
  21 %). Com o teto o ajuste converge (|g| 1,4e-5); o Wald A dá p = 0,0708 e a decisão continua dispersão única (BIC
  18.724,9 com `regiao` contra 18.751,0 constante, sem teste de Wald que a sustente).

## 7. Outros defeitos corrigidos (regras, não código)

- O ramo do beta-binomial simples (sem dispersão por região) não tinha o n do restante, e o 03 falhava nos cortes cujo
  desenho rejeita a dispersão (2022 e 2026 a 5 e 10 %). Caso de borda agora coberto por teste.
- A seleção aninhada do 02 lia o quadro sem as colunas do 1º turno, que agora são candidatas. Coberto por teste.

## 8. Limitações

- **Um experimento, dois anos.** A regra de aceitação foi fixada antes, mas o IC cobre 72 de 72 itens com largura 5
  vezes menor; a cobertura real não foi calibrada ([11](11-apuracao-parcial.md), seção 6). O diagnóstico ex post da
  largura está em [15](15-apuracao-corte-precoce-calibracao.md), seção 3.
- **Referências comparáveis.** A literatura de apuração em noite de eleição com amostra não aleatória de seções
  (Pavía-Miralles 2005; Pavía-Miralles e Larraz-Iribas 2008, [referências](referencias.md)) trata a mesma
  dificuldade de seleção; não replicamos seus estimadores.
