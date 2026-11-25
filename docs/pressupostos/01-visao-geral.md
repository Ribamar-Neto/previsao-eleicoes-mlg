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
- **Superdispersão.** φ de Pearson = 21,24; na beta-binomial θ ≈ 651 e ρ ≈ 0,0015. Por isso
  entram quase-binomial, beta-binomial, GLMM e beta, além da binomial pura.
- **Defasagem.** O coeficiente da defasagem fica ≈ 0,89; o offset (coeficiente fixo em 1) é
  rejeitado (LRT 831). O σ_UF ≈ 0,05.
- **Estimativa regional só fora da amostra.** Uma binomial com ligação canônica e dummies de
  região reproduz exatamente os totais regionais dentro da amostra; por isso as estimativas
  vêm de previsões leave-one-state-out (LOSO). O LOSO não usa o 2º turno da UF retida: só
  covariáveis anteriores ao turno, e a UF retida é pesada por n̂ ([09](09-modelo-final.md)).
- **Nordeste e Norte têm a mesma média.** A região entra na média com os dois fundidos
  (a igualdade não é rejeitada, [03](03-modelos-candidatos.md) seção 8.1); no bloco do prefeito a regra funde centro e sem prefeito com a direita e deixa a esquerda separada (p = 8e-6). A dispersão
  mantém as 5 regiões ([03](03-modelos-candidatos.md), [07](07-dispersao-regional.md)).
- **GLMM com PQL próprio**, porque o `BinomialBayesMixedGLM` do statsmodels só aceita
  resposta 0/1.
- **Quase-binomial com φ de Pearson próprio**, porque `scale='X2'` do statsmodels erra com
  endog de 2 colunas.

## Organização dos notebooks

- **Sem funções nos notebooks.** Toda função auxiliar fica em `src/` (testada em `tests/`); a
  célula mostra só o passo estatístico. Razão: o leitor vê o método sem o código de apoio.
- **Três notebooks.** `01-AED` (Partes 1 a 4), `02-MLG` (Partes 5 a 11 e 13 a 17, num só
  Run-All) e `03-apuracao-parcial` (Parte 18). Razão: antes, cada notebook lia arquivos do
  anterior e exigia ordem de execução manual (incluindo rodar o 02 duas vezes). Os números das
  Partes são estáveis: a Parte 12 (backtest) foi retirada, não renumerada ([06](06-backtest.md)).
  O Passo 6 da Parte 10 roda depois da Parte 15, porque diagnostica `bb_selecao` e `bb_slx`.
- **Cada valor é calculado uma vez.** Entre as Partes do `02-MLG` os valores passam em memória
  (sem leitura de arquivo entre notebooks e sem "rode o 03 antes"); os CSV em `results/tables`
  continuam gravados para consulta. O I de Moran do `bb_slx`, por exemplo, é calculado na
  Parte 14 e vale para o Passo 6 da Parte 10; os ajustes parciais da apuração saem de um só
  laço na Parte 18.
- **Ajustes repetidos de propósito.** Os reajustes do `bb_slx` por dobra nas Partes 16 e 17
  ficam em cada parte, porque rodam em processos paralelos e o compartilhamento dos ajustes
  não foi testado.
- **Gráficos em PNG estático (kaleido).** Os gráficos Plotly são exportados em PNG para
  `results/figures`; os `.html` foram apagados. Razão: o GitHub não renderiza a saída JSON do
  Plotly nos notebooks, e o leitor veria células vazias.
- **Regra das tabelas.** Uma tabela fica no notebook só quando uma célula de interpretação
  cita seus números para uma decisão; senão vira gráfico, lista de indicadores ou nada (o CSV
  guarda o detalhe).
- **Cada explicação aparece uma vez.** A teoria (por exemplo, a sobredispersão na Parte 7) fica
  numa só célula; as outras Partes e os outros notebooks dão uma frase e um link. Razão: textos
  repetidos divergem quando um deles muda.
- **Texto curto e comentário de leitura.** As células de texto seguem frases curtas, com uma
  ideia por frase (princípios do ASD-STE100 em português). O comentário numa célula de código
  diz por que o passo existe ou como ler a saída, não o que a linha faz.

## Explicar × prever

O ajuste com todos os municípios serve para interpretar coeficientes e diagnosticar o modelo;
não mede previsão, porque a binomial canônica com dummies de região reproduz os totais
regionais dentro da amostra. A previsão é medida só fora da amostra: no LOSO e na apuração
parcial na ordem real do TSE ([11](11-apuracao-parcial.md)). Uma amostra aleatória de
municípios não é usada: os votos não chegam ao acaso (capitais primeiro) e os vizinhos são
correlacionados, então treino e teste teriam vizinhos e o erro ficaria subestimado
(Shmueli 2010).

## Regra de seleção (fixada antes dos resultados)

1. Menor erro absoluto médio da participação por UF no LOSO (ponderado por n̂, nas Partes 11 e 13 a 17 do `02-MLG`).
2. Regra do 1 erro-padrão: entre os modelos dentro de 1 EP do melhor, o de menos parâmetros.
3. O MAE ponderado por votos da validação de 5 dobras é critério secundário.
4. Variáveis do modelo escolhido: BIC (backward, forward e both dão o mesmo conjunto).
   AIC, Hosmer-Lemeshow e todos os subconjuntos são comparações ([03](03-modelos-candidatos.md)).
5. Uma extensão do modelo escolhido só é aceita se melhora o seu diagnóstico **e** o MAE por UF
   no LOSO não passa de L = MAE aninhado de `bb_selecao` (seleção refeita em cada dobra) + EP do melhor modelo LOSO =
   1,0669 + 0,1419 = **1,2088 p.p.** (1,1472 sem aninhar).

O erro regional LOSO não é critério, porque favorece por construção as binomiais canônicas
([05](05-validacao-cruzada.md), seção 6).

## Resultados finais

- Treze modelos ficam dentro de 1 EP do melhor (`bb_selecao`, 1,005 p.p., EP 0,158; limite
  1,164). A regra escolhe `bb_selecao` (19 parâmetros, MAE por UF 1,005 p.p., EP 0,158); o
  segundo é `bin_logit` (1,009) ([05](05-validacao-cruzada.md)).
- Resíduos quantílicos: DP 1,03 na beta-binomial e 3,42 na binomial.
- Sem os 10 municípios mais influentes (capitais), `bb_reglag` muda no máximo 0,18 EP;
  `quasi_reglag` muda 9,5 EP.

Refinamento (Partes 13 a 17 do `02-MLG`): `bb_selecao` virou o modelo final `bb_slx` (37 parâmetros):
dispersão por região e log n (`bb_disp`; [07](07-dispersao-regional.md); aceita, LOSO 0,996,
confirmada por Wald robusto) mais covariáveis defasadas espacialmente, SLX (LOSO 1,073 ≤ 1,2088,
acima do 1,005 de `bb_selecao`; I de Moran dos resíduos só de 0,353 para 0,348). O filtro de
autovetores foi rejeitado (LOSO 1,584) e a inferência usa EPs de Conley a 200 km, corte
conferido com 2022 ([08](08-dependencia-espacial.md)). Krigagem e CAR nos resíduos foram
rejeitadas (queda do I de 0,0005 e 0,0311, abaixo do mínimo de 0,05;
[10](10-extensoes-rejeitadas.md)). O modelo final segue `bb_slx`.

Estimativas regionais do modelo final (LOSO, Conley + σ_UF = 0,0570), Lula %, cenário "UF
inédita" ([09](09-modelo-final.md)):

| Região | Estimado | IC 95% | Real |
|---|---|---|---|
| Centro-Oeste | 39,96 | [38,4; 41,4] | 39,99 |
| Nordeste | 69,58 | [68,6; 70,5] | 69,37 |
| Norte | 49,28 | [47,8; 50,8] | 50,01 |
| Sudeste | 47,94 | [46,3; 49,6] | 47,24 |
| Sul | 36,92 | [35,4; 38,4] | 37,42 |
| Brasil | 51,82 | [51,2; 52,5] (90%) | 51,78 |

MAE por UF 1,073 p.p. no LOSO (0,846 no 5-fold); MAE regional 0,431 p.p.

Apuração parcial na ordem real das seções (Parte 18): [11](11-apuracao-parcial.md). O desenho do
modelo é refeito em cada corte só com os municípios já apurados (antes usava o desenho final,
escolhido com o resultado de 2026: vazamento). Resultado: o IC95 do Brasil cobre o final nos 12
cortes (antes 11), as regiões em 55 de 60 itens (antes 50), com IC mais largos (Brasil, média
2,18 p.p. em 2022 e 1,51 em 2026); o modelo é útil nos dois anos (4 de 6 e 5 de 6 cortes) e a
5 % de 2026 ainda perde para a contagem bruta no Brasil (1,02 contra 0,97 p.p.).

Pendências: [12-pendencias.md](12-pendencias.md). Referências: [referencias.md](referencias.md).
