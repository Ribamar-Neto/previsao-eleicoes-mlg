# 13 - Extensões rejeitadas da apuração parcial: encolhimento por UF e desenho fixo (Parte 18)

Notebook `notebooks/03-apuracao-parcial.ipynb`. Testado em 2026-11-26 e **rejeitado**: a Parte 18 segue como em
[11-apuracao-parcial.md](11-apuracao-parcial.md). O código da tentativa não foi mantido: ficam as regras e os
resultados. Com λ = ρ = 0 ele reproduzia o modelo atual.

**Estado datado.** "Modelo atual" e "antes" neste arquivo são o modelo de 2026-11-26, sem a covariável do 1º turno do
mesmo ano. Depois dela ([14](14-apuracao-diagnostico-1t.md)) o IC do modelo é de 0,39 p.p. (Brasil, já com o choque de
UF t de Student, [11](11-apuracao-parcial.md), seção 6) e cobre 72 de 72 itens; o encolhimento por UF não foi retestado
nesse modelo e a rejeição vale para o estado de 26/11. A coluna "Vitórias pela regra" da seção 4 usa a regra daquele dia
(bruta e swing): com o `swing_1t` como terceira linha de base ([11](11-apuracao-parcial.md), seção 1) o veredito atual é
2022 não útil (3 de 6) e 2026 útil (6 de 6), e as comparações abaixo não foram refeitas com ele.

## 1. Pergunta

Por que o IC é tão largo e a estimativa tão sensível à região dos municípios já apurados? Daria para usar o
desvio que o apurado mostra na UF para corrigir o restante e estreitar o IC?

## 2. Regras (fixadas antes de olhar 2026)

Notação: η̂ᵢ é o preditor linear do modelo do corte; rᵢ = g(y_c/n_c, limitado a [0,5/n_c, 1 − 0,5/n_c]) − η̂ᵢ nos
municípios apurados; g é a ligação do corte; σ é o σ_UF do desenho do corte.

- **R8, encolhimento por UF (BLUP).** Em cada UF: ū é a média de rᵢ ponderada por n_c; τ² é a variância de rᵢ
  dentro da UF (agrupada entre UFs, ponderada); m_eff = (Σn_c)²/Σn_c² (contagem efetiva de Kish).
  k = m_eff·σ²/(m_eff·σ² + τ²); û_UF = k·ū; v_UF = σ²·τ²/(m_eff·σ² + τ²).
  No restante, η = η̂ + λ·û_UF, e o choque simulado da UF tem sd_UF = √(λ·v_UF + (1 − λ)·σ²).
  UF sem município apurado: û = 0 e sd = σ.
- **R9, parcela já apurada do próprio município.** ηᵢ += ρ·fᵢ·(rᵢ − λ·û_UF), com fᵢ = n_c/(n_c + n_u). Só move o
  centro. ρ limita o peso porque as primeiras seções são mais urbanas ([11](11-apuracao-parcial.md), seção 7).
- **R10, calibração só em 2022.** Grade λ, ρ ∈ {0; 0,25; 0,5; 0,75; 1}. Critério: média nos 6 cortes de 2022 de
  (erro absoluto do Brasil + MAE regional), só nas estimativas pontuais. Empate vai para o valor menor. O par
  vencedor vale para 2026 sem mudança. Os números de 2022 ficam dentro da amostra para (λ, ρ).
- **R11, desenho fixo em 2026.** Todo corte de 2026 usa o desenho que R1 a R7 escolhem na apuração completa de
  2022 ([11](11-apuracao-parcial.md), seção 9), com seu corte de Conley e σ_UF. Ele era conhecido antes do 2º turno
  de 2026 (sem vazamento). β é reajustado em cada corte. 2022 segue com o desenho de cada corte (não há apuração
  de 2018 para fixá-lo).
- **Aceitação.** Substituir o modelo atual só se 2026, o teste honesto, não piorar a cobertura do IC95 nem o erro.

## 3. Calibração (R10)

Venceu λ = 1, ρ = 0 (critério 0,970 contra 1,191 com λ = ρ = 0). Os cinco melhores pares têm λ = 1 (0,970 a
0,976): o ganho vem quase todo do encolhimento por UF, e a parcela do próprio município quase não ajuda.
O critério só olha a estimativa pontual, por isso não pune um IC estreito demais.

## 4. Resultado: antes × depois (λ = 1, ρ = 0, com R11 em 2026)

| Ano | Largura IC Brasil (p.p.) | Largura IC regiões | Brasil dentro | Regiões dentro | Vitórias pela regra |
|---|---|---|---|---|---|
| 2022 antes | 2,18 | 3,14 | 6/6 | 29/30 | 4/6 |
| 2022 depois | 0,89 | 1,48 | 1/6 | 21/30 | 4/6 |
| 2026 antes | 1,47 | 2,21 | 6/6 | 27/30 | 5/6 |
| 2026 depois | 0,91 | 1,46 | 4/6 | 20/30 | 4/6 |

- Total: Brasil dentro do IC em 5 de 12 cortes (antes 12 de 12), regiões em 41 de 60 (antes 56 de 60).
- Erro do Brasil em 2026: 1,02 → 1,30 p.p. a 5 %, 0,16 → 0,48 a 25 %. Em 2022 (dentro da amostra) caiu a 5 %
  (0,72 → 0,36) e a 10 % (1,19 → 0,80) e ficou igual nos demais cortes.
- A cobertura caiu também em 2022, o ano da calibração: o IC estreitou muito mais do que o erro caiu.

## 5. Diagnóstico: R8 e R11 em separado (2026)

Quatro variantes, mesma semente (7) e B = 1000. V0 reproduz o modelo atual, e V3 reproduz a seção 4, com
diferença 0,0 no erro e nos limites do IC do Brasil.

| Variante | Erro Brasil (média, p.p.) | MAE regional | Largura Brasil | Largura regional | Brasil dentro | Regiões dentro |
|---|---|---|---|---|---|---|
| V0: desenho por corte, λ = 0 (atual) | 0,26 | 0,45 | 1,47 | 2,21 | 6/6 | 27/30 |
| V1: só R11 (desenho fixo) | 0,27 | 0,55 | 2,05 | 2,95 | 6/6 | 28/30 |
| V2: só R8 (λ = 1) | 0,34 | 0,53 | 0,81 | 1,29 | 5/6 | 19/30 |
| V3: R8 + R11 | 0,39 | 0,55 | 0,91 | 1,46 | 4/6 | 20/30 |

Por corte (erro Brasil / largura Brasil, p.p.):

| Corte | V0 | V1 | V2 | V3 |
|---|---|---|---|---|
| 5 | 1,02 / 3,53 | 0,99 / 3,17 | 1,30 / 2,12 | 1,30 / 2,11 |
| 10 | 0,16 / 1,90 | 0,02 / 2,65 | 0,07 / 1,14 | 0,22 / 1,47 |
| 25 | 0,16 / 1,43 | 0,34 / 2,64 | 0,36 / 0,78 | 0,48 / 0,99 |
| 50 | 0,13 / 1,02 | 0,17 / 1,86 | 0,20 / 0,42 | 0,23 / 0,47 |
| 67 | 0,07 / 0,68 | 0,08 / 1,48 | 0,11 / 0,28 | 0,11 / 0,32 |
| 90 | 0,03 / 0,23 | 0,03 / 0,51 | 0,01 / 0,11 | 0,02 / 0,13 |

- **R8 tira a cobertura.** Ele reduz o IC pela metade ou mais e aumenta o erro.
- **R11 sozinho não ajuda.** O erro fica igual ou pouco maior, o IC alarga (Brasil 1,47 → 2,05 p.p.) e a
  cobertura não muda. Logo não há ganho que justifique trocar o desenho de cada corte.
- **Por UF (V2, 5 %).** sd_UF fica entre 0,25 e 0,83 de σ (σ = 0,137), e na maioria das UFs entre 0,4 e 0,6.
  No Centro-Oeste e no Nordeste, onde o modelo superestima Lula (+2,4 e +2,2 p.p.), û é negativo (GO −0,068,
  MA −0,081, PI −0,067): o centro vai na direção certa, mas pouco. O erro cresce no Norte, no Sudeste e no Sul,
  com û positivo em AP (+0,46) e RJ (+0,14) [assumed: efeito não isolado por UF].

## 6. Decisão e razão

- **Rejeitadas R8 a R11.** Pela regra da seção 2, 2026 piora na cobertura (Brasil 6/6 → 4/6, regiões 27/30 → 20/30)
  e no erro.
- **Razão [deduced].** O encolhimento trata os municípios apurados de uma UF como amostra aleatória dela. Na ordem
  real do TSE eles não são: capitais e grandes cidades entram primeiro ([11](11-apuracao-parcial.md), seção 7). O
  desvio deles diz pouco sobre o restante da UF, e o BLUP confia demais nele e tira variância que existe.
- **Resposta à pergunta da seção 1, no estado de 26/11.** O IC largo era lido como o preço da ordem de apuração não
  aleatória, e o encolhimento não o removia. Essa leitura ficou incompleta [fact]: o diagnóstico de
  [14](14-apuracao-diagnostico-1t.md) mostrou que a maior parte da largura vinha de um σ_UF inflado pela covariável do
  1º turno ausente, e não da ordem em si.
- **O que poderia funcionar.** Um encolhimento que modele a ordem (por exemplo, rᵢ ponderado pela semelhança entre
  o município apurado e o restante da UF), calibrado com critério que inclua a cobertura do IC. Fica como
  pendência ([12](12-pendencias.md)).
