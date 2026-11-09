# Pendências

Todas são [fact] salvo marca em contrário. Cada item: efeito e correção possível.

## Dados

- **Boa Esperança do Norte (MT) fora das 5.570 linhas** (2026: PT 945, Bolsonaro 2.353). Os
  totais reais em `results/tables/resultado_*.csv` vêm das linhas modeladas. Efeito:
  incluir o município muda o Centro-Oeste em −0,004 p.p. Correção: reportar à parte (já é).
- **Bolsa Família 2022** vem do arquivo de saques do Portal da Transparência: 20,07 M de
  famílias, cerca de 2,8% abaixo da referência. Efeito: cobertura subestimada. Correção:
  usar o agregado oficial do MDS, hoje indisponível.
- **Perfil do eleitorado 2026** gerado em 14/07/2026, sem confirmação de ser o perfil
  fechado. Efeito: pequenas diferenças de cadastro. Correção: rebaixar o perfil fechado.
- **`pct_vab_agro` usa 2021** e **`pib_per_capita_2023` divide pela população de 2022**.
  Efeito: defasagem de cinco anos e denominador aproximado. Correção: atualizar quando o
  IBGE publicar VAB posterior e estimativas de população.
- **Classificação ideológica de UP e Missão sem citação** (não constam em Bolognesi,
  Ribeiro e Codato 2023). Efeito: bloco de poucos votos com base fraca. Correção: citar
  fonte ou agrupar como "outros".

## Modelo

- **[resolvida para a inferência] Resíduos com autocorrelação espacial em todos os modelos**
  (I de Moran ≈ 0,29 a 0,35, p = 0,001). Os EPs de Conley a 200 km (cerca de 2 vezes os do
  modelo) corrigem a inferência ([10](10-dependencia-espacial.md), seção 3). O filtro de
  autovetores foi rejeitado (LOSO 1,446 > 1,1308). **Resta:** I de Moran 0,339 no modelo
  final; Conley não corrige as estimativas pontuais. Correção possível: CAR/BYM ou
  defasagem espacial, a validar por UF.
- **[resolvida] θ não é constante por região** (DP residual de 0,80 no Sudeste a 1,40 no
  Norte). Dispersão por região e log n: máximo |DP − 1| de 0,409 para 0,095
  ([09](09-dispersao-regional.md)). Resta o Norte (DP 1,10).
- **LRT da dispersão supõe municípios independentes** ([09](09-dispersao-regional.md), seção 3),
  o que a correlação residual contradiz. A diferença de BIC (450) e o ganho de DP seguem
  valendo; o p-valor é indicativo.
- **Corte de Conley (200 km) vem do correlograma de uma só eleição.** Correção: repetir com
  2018 e 2022.
- **Envelope meio-normal de `bb_reglag`: 93,9% dos pontos fora** (corpo estreito, caudas
  pesadas, curtose 3,8). Efeito: o envelope não valida a distribuição. Correção: modelo com
  caudas mais pesadas.
- **Pontos mais influentes são as capitais** (Cook: Brasília 5,1; São Paulo 3,9). Efeito:
  sensibilidade medida em [04](04-diagnostico.md).

## Validação

- **Cobertura dos intervalos (27/27 UFs, 5/5 regiões) é otimista**: σ_UF = 0,054 foi
  ajustado nas mesmas UFs; com σ = 0 a cobertura é 20/27 em `bb_backward` e 16/27 no modelo
  final com Conley ([11](11-modelo-final.md), seção 3). Correção: σ_UF por validação
  aninhada.
- **Erro regional LOSO favorece binomiais canônicas por construção** ([06](06-estimativas-regionais.md));
  não serve para seleção.
- **12 modelos dentro de 1 EP**: os dados não os separam com clareza [deduced]. A escolha
  de `bb_backward` repousa na regra de parcimônia, não numa vantagem demonstrada.

## Backtest

- **Sem perfil do eleitorado de 2018**: o ajuste usa o perfil de 2022.
- **Censo 2022 (publicado em 2023)** entra no ajuste de 2018: pequeno look-ahead.
- **Modelo entre turnos sem recalibração erra o nível nacional em −4,7 p.p.**; a causa não
  foi testada. Correção: testar efeito de turno e de abstenção diferencial.

## Código/testes

- **[resolvida] Teste de convergência fraco.** `tests/test_convergencia.py` usa dados reais
  (`data/processed/municipios.parquet`; pulado se ausente): a tolerância antiga (`gtol` =
  1e-4·N) parava 398 unidades de log-verossimilhança abaixo do ótimo polido; o padrão atual
  fica a 2e-6. O teste sintético foi removido de `tests/test_modelos.py`.
- **[resolvida] Referências com ✔m**: `referencias.md` tem 0 ✔m (36 entradas conferidas na
  web com DOI/ISBN/URL); detalhes residuais na legenda do arquivo.
- **Duas saídas do 02-MLG variam entre execuções.** Os coeficientes de `l1_logit` mudam na
  ordem de 1e-5 (log-verossimilhança ±5) porque o solver `liblinear` embaralha os dados sem
  semente. O `z_sim` de `diag_moran.csv` muda porque as 999 permutações do esda não têm
  semente. O I de Moran e as demais tabelas não mudam. Correção: `random_state=0` no
  `liblinear` e semente fixa nas permutações. Os números do `l1_logit` mudam, mas não a
  seleção.
- **Envelope da beta-binomial no 02-MLG consome a maior parte dos 7,5 min** (49 reajustes
  de ~4 s). Correção: partida a quente (warm start) e remover o passo de polimento.
