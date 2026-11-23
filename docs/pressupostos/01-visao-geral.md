# Visão geral

## Problema

Estimar, por região, a votação de Lula e de Flávio Bolsonaro no segundo turno de 2026, a
partir de modelos lineares generalizados (MLG) da família binomial ajustados por município.
O trabalho é acadêmico; o objetivo é comparar especificações com critério definido antes dos
resultados e agregar as previsões às 5 regiões.

## Modelo

- Unidade: município (5.570 linhas). É um censo, não uma amostra.
- Yᵢ = votos de Lula; nᵢ = votos de Lula + Flávio (válidos nos dois candidatos, só Brasil;
  o exterior, UF `ZZ`, sai). Yᵢ ~ Bin(nᵢ, πᵢ), g(πᵢ) = xᵢᵀβ.
- π_Flávio = 1 − π_Lula: um modelo dá os dois candidatos.
- Participação regional: p̂_r = Σ π̂ᵢ nᵢ / Σ nᵢ sobre os municípios da região r.

## Decisões e razões

- **Binomial agrupado.** Tem a mesma verossimilhança da logística sobre votos individuais
  0/1 quando as covariáveis são municipais. Vale a falácia ecológica: o modelo descreve
  municípios, não eleitores (Robinson 1950).
- **Gama, normal inversa e gaussiana ficam fora.** O suporte não combina com proporção.
- **Superdispersão.** φ de Pearson = 21,8; na beta-binomial θ ≈ 624 e ρ ≈ 0,0016. Por isso
  entram quase-binomial, beta-binomial, GLMM e beta, além da binomial pura.
- **Defasagem.** O coeficiente da defasagem fica ≈ 0,90; o offset (coeficiente fixo em 1) é
  rejeitado (LRT 755). O σ_UF ≈ 0,05.
- **Estimativa regional só fora da amostra.** Uma binomial com ligação canônica e dummies de
  região reproduz exatamente os totais regionais dentro da amostra; por isso as estimativas
  vêm de previsões leave-one-state-out (LOSO). O LOSO não usa o 2º turno da UF retida: só
  covariáveis anteriores ao turno, e a UF retida é pesada por n̂ ([09](09-modelo-final.md)).
- **Nordeste e Norte têm a mesma média.** A região entra na média com os dois fundidos
  (teste de igualdade, p = 0,921); o bloco do prefeito não é fundido (p = 8e-4). A dispersão
  mantém as 5 regiões ([03](03-modelos-candidatos.md), [07](07-dispersao-regional.md)).
- **GLMM com PQL próprio**, porque o `BinomialBayesMixedGLM` do statsmodels só aceita
  resposta 0/1.
- **Quase-binomial com φ de Pearson próprio**, porque `scale='X2'` do statsmodels erra com
  endog de 2 colunas.

## Regra de seleção (fixada antes dos resultados)

1. Menor erro absoluto médio da participação por UF no LOSO (ponderado por n̂, nos notebooks 03 e 05).
2. Regra do 1 erro-padrão: entre os modelos dentro de 1 EP do melhor, o de menos parâmetros.
3. O MAE ponderado por votos da validação de 5 dobras é critério secundário.
4. Variáveis do modelo escolhido: BIC (backward, forward e both dão o mesmo conjunto).
   AIC, Hosmer-Lemeshow e todos os subconjuntos são comparações ([03](03-modelos-candidatos.md)).
5. Uma extensão do modelo escolhido só é aceita se melhora o seu diagnóstico **e** o MAE por UF
   no LOSO não passa de L = MAE aninhado de `bb_selecao` (seleção refeita em cada dobra) + EP do melhor modelo LOSO =
   1,1168 + 0,1347 = **1,2515 p.p.** (1,1695 sem aninhar; 1,1308 no LOSO vazado).

O erro regional LOSO não é critério, porque favorece por construção as binomiais canônicas
([05](05-validacao-cruzada.md), seção 6).

## Resultados finais

- Treze modelos ficam dentro de 1 EP do melhor (`bin_probit`, 0,998 p.p., EP 0,135; limite
  1,133). A regra escolhe `bb_selecao` (19 parâmetros, MAE por UF 1,035 p.p., EP 0,156); o
  segundo é `bin_probit` ([05](05-validacao-cruzada.md)).
- Resíduos quantílicos: DP 1,03 na beta-binomial e 3,45 na binomial.
- Sem os 10 municípios mais influentes (capitais), `bb_reglag` muda no máximo 0,15 EP;
  `quasi_reglag` muda 9,0 EP.

Refinamento (notebook 05): `bb_selecao` virou o modelo final `bb_slx` (36 parâmetros):
dispersão por região e log n (`bb_disp`; [07](07-dispersao-regional.md); aceita, LOSO 1,0345,
confirmada por Wald robusto) mais covariáveis defasadas espacialmente, SLX (LOSO 1,078 ≤ 1,2515,
acima do 1,035 de `bb_selecao`; I de Moran dos resíduos só de 0,342 para 0,336). O filtro de
autovetores foi rejeitado (LOSO 1,484) e a inferência usa EPs de Conley a 200 km, corte
conferido com 2022 ([08](08-dependencia-espacial.md)). Krigagem e CAR nos resíduos foram
rejeitadas (queda do I de 0,0003 e 0,0275, abaixo do mínimo de 0,05;
[10](10-extensoes-rejeitadas.md)). O modelo final segue `bb_slx`.

Estimativas regionais do modelo final (LOSO, Conley + σ_UF = 0,0564), Lula %, cenário "UF
inédita" ([09](09-modelo-final.md)):

| Região | Estimado | IC 95% | Real |
|---|---|---|---|
| Centro-Oeste | 40,09 | [38,6; 41,6] | 39,99 |
| Nordeste | 69,44 | [68,5; 70,3] | 69,37 |
| Norte | 49,18 | [47,7; 50,7] | 50,01 |
| Sudeste | 47,89 | [46,3; 49,6] | 47,24 |
| Sul | 36,93 | [35,4; 38,5] | 37,42 |
| Brasil | 51,76 | [51,1; 52,4] (90%) | 51,78 |

MAE por UF 1,078 p.p. no LOSO (0,846 no 5-fold); MAE regional 0,426 p.p.

Backtest (ajuste de 2022 em 2018, previsão de 2026): a melhor variante é `C_recal`
(participações do 1º turno de 2026, intercepto ancorado na participação nacional de 2022),
com erro absoluto médio de 0,58 p.p. por UF e 0,68 p.p. por região. A linha de base ingênua
(repetir 2022) dá 1,65 (UF) e 0,64 (região). Sem o 1º turno, os modelos ficam piores que a
linha de base (UF 3,3 a 3,9): as oscilações regionais de 2018 a 2022 não se transferem a
2026 ([06](06-backtest.md)).

Apuração parcial na ordem real das seções (Parte 18): [11](11-apuracao-parcial.md).

Pendências: [12-pendencias.md](12-pendencias.md). Referências: [referencias.md](referencias.md).
