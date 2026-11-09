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
- **Superdispersão.** φ de Pearson = 21,7; na beta-binomial θ ≈ 630 e ρ ≈ 0,0016. Por isso
  entram quase-binomial, beta-binomial, GLMM e beta, além da binomial pura.
- **Defasagem.** O coeficiente da defasagem fica ≈ 0,90; o offset (coeficiente fixo em 1) é
  rejeitado (LRT 755). O σ_UF ≈ 0,05.
- **Estimativa regional só fora da amostra.** Uma binomial com ligação canônica e dummies de
  região reproduz exatamente os totais regionais dentro da amostra; por isso as estimativas
  vêm de previsões leave-one-state-out (LOSO).
- **GLMM com PQL próprio**, porque o `BinomialBayesMixedGLM` do statsmodels só aceita
  resposta 0/1.
- **Quase-binomial com φ de Pearson próprio**, porque `scale='X2'` do statsmodels erra com
  endog de 2 colunas.

## Regra de seleção (fixada antes dos resultados)

1. Menor erro absoluto médio da participação por UF no LOSO.
2. Regra do 1 erro-padrão: entre os modelos dentro de 1 EP do melhor, o de menos parâmetros.
3. O MAE ponderado por votos da validação de 5 dobras é critério secundário.

O erro regional LOSO não é critério, porque favorece por construção as binomiais canônicas
([06](06-estimativas-regionais.md)).

## Resultados finais

- 12 modelos ficam dentro de 1 EP do melhor (`bin_probit`, 0,930 p.p., EP 0,130). A regra
  escolhe `bb_backward` (20 parâmetros, MAE por UF 1,000 p.p.); o segundo é `bin_probit`.
- Resíduos quantílicos: DP 1,03 na beta-binomial e 3,43 na binomial.
- Sem os 10 municípios mais influentes (capitais), `bb_reglag` muda no máximo 0,14 EP;
  `quasi_reglag` muda 8,4 EP.

Refinamento (notebook 05): `bb_backward` virou o modelo final `bb_disp`, com a dispersão
variando por região e log n ([09](09-dispersao-regional.md)): max |DP − 1| por região de
0,409 para 0,095, MAE por UF no LOSO 0,995. O filtro espacial de autovetores foi rejeitado
(LOSO 1,446 > 1,1308) e a inferência usa EPs de Conley a 200 km, cerca de 2 vezes os do
modelo ([10](10-dependencia-espacial.md)). Os resultados abaixo vêm do modelo final
([11](11-modelo-final.md)); as estimativas de `bb_backward` ficam em
[06](06-estimativas-regionais.md).

Estimativas regionais do modelo final (LOSO, Conley + σ_UF), Lula %:

| Região | Estimado | IC 95% | Real |
|---|---|---|---|
| Centro-Oeste | 40,12 | [38,7; 41,6] | 39,99 |
| Nordeste | 69,40 | [68,5; 70,2] | 69,37 |
| Norte | 49,53 | [48,1; 50,9] | 50,01 |
| Sudeste | 47,58 | [46,0; 49,2] | 47,24 |
| Sul | 37,02 | [35,5; 38,4] | 37,42 |
| Brasil | 51,84 | [51,1; 52,6] | 51,78 |

Backtest (ajuste de 2022 em 2018, previsão de 2026): a melhor variante é `C_recal`
(participações do 1º turno de 2026, intercepto ancorado na participação nacional de 2022),
com erro absoluto médio de 0,58 p.p. por UF e 0,68 p.p. por região. A linha de base ingênua
(repetir 2022) dá 1,65 (UF) e 0,64 (região). Sem o 1º turno, os modelos ficam piores que a
linha de base (UF 3,3 a 3,9): as oscilações regionais de 2018 a 2022 não se transferem a
2026 ([08](08-backtest.md)).

Pendências: [07-pendencias.md](07-pendencias.md). Referências: [referencias.md](referencias.md).
