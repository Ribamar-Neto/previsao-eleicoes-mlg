# Pendências

Todas são [fact] salvo marca em contrário. Cada item: efeito e correção possível.

## Dados

- **Boa Esperança do Norte (MT) fora das 5.570 linhas** (2026: PT 945, Bolsonaro 2.353). Os
  totais reais em `results/tables/resultado_*.csv` vêm das linhas modeladas. Efeito:
  incluir o município muda o Centro-Oeste em −0,004 p.p. Correção: reportar à parte (já é).
- **Bolsa Família 2022** vem do arquivo de saques do Portal da Transparência: 20,07 M de
  famílias pagas, 2,81% abaixo dos 20,65 M de famílias da folha de setembro de 2022 do
  Ministério da Cidadania (nota de 16/09/2022, [referências](referencias.md)). A base da
  comparação é família na folha, não NIS distinto pago. O texto original no gov.br não foi
  encontrado; a fonte achada é a republicação do TNH1. Efeito: cobertura subestimada. Correção:
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

- **[fechada como limitação] Resíduos com autocorrelação espacial em todos os modelos**
  (I de Moran ≈ 0,34). Conley a 200 km corrige a inferência
  ([08](08-dependencia-espacial.md), seções 4 e 5); ESF foi rejeitado (LOSO 1,484 contra o
  limite 1,2515) e o SLX aceito, mas o SLX só baixou o I de 0,342 para 0,336. A Parte 17
  decompôs o I e testou krigagem dos resíduos sob LOSO: rejeitada (queda do I de 0,0003
  contra o mínimo de 0,05). Testou também CAR/BYM em dois estágios sob a mesma regra:
  rejeitado (queda do I de 0,0275; MAE por UF de 1,004, mas MAE regional pior)
  ([10](10-extensoes-rejeitadas.md)). A correlação não é removível com a informação
  disponível para uma UF nova. Não resta alternativa espacial na lista.
- **[resolvida] θ não é constante por região** (DP residual de 0,80 no Sudeste a 1,43 no
  Norte). Dispersão por região e log n: máximo |DP − 1| de 0,434 para 0,087
  ([07](07-dispersao-regional.md)). Resta o Norte (DP 1,09).
- **[resolvida] LRT da dispersão supõe municípios independentes** ([07](07-dispersao-regional.md),
  seção 3). O Wald robusto com Conley (e por UF) mantém `regiao` e `log n`, com p ≤ 1e-4
  em todas as covariâncias ([07](07-dispersao-regional.md), seção 5).
- **[resolvida] Corte de Conley (200 km) vinha do correlograma de uma só eleição.** O de 2022
  dá 200 km também; mantido ([08](08-dependencia-espacial.md), seção 5).
- **Regras adicionadas depois dos primeiros resultados da Parte 13** (Wald robusto, SLX e
  corte de 2022, em 2026-11-09): foram fixadas antes de cada resultado, mas não antes do
  notebook inteiro; o leitor deve pesar isso.
- **Fusão de Nordeste e Norte na média e rodada de seleção de variáveis** (BIC, AIC,
  Hosmer-Lemeshow e todos os subconjuntos) vieram da revisão da análise, depois dos
  resultados anteriores; o leitor deve pesar isso. A busca em todos os subconjuntos
  (triagem quase-binomial e reajuste dos 200 melhores) não achou o ótimo do stepwise (BIC
  72.848,8 contra 72.844,87): fraqueza da triagem. Correção possível: ampliar a triagem.
- **Envelope meio-normal fora da banda em todos os beta-binomiais**: `bb_reglag` 92,5%,
  `bb_selecao` 90,2%, `bb_slx` 84,2% dos pontos fora (corpo abaixo, caudas acima; curtose
  3,8 / 3,95 / 2,49). A dispersão por região melhora a curtose, mas o padrão persiste. Efeito:
  o envelope não valida a distribuição. Correção: modelo com caudas mais pesadas
  ([04](04-diagnostico.md), seção 8.3).
- **Influência no beta-binomial**: os maiores Cook de um passo são municípios pequenos
  (Uiramutã/RR, Jutaí/AM), não as capitais. O reajuste sem os 10 maiores move no máximo
  0,86 EP, sem troca de sinal. Efeito: inferência estável; a sensibilidade do quase-binomial
  às capitais fica restrita a ele ([04](04-diagnostico.md), seções 4 e 8.7).
- **Ligação sinalizada em `bb_selecao` e `bb_slx`** (p Conley 0,028 e 0,0003; P95 de
  \|Δπ̂\| 0,21 e 0,33 p.p.). Efeito: pequeno, abaixo de 0,6 p.p. Correção: ligação
  paramétrica ou η̂² na média, validado por validação cruzada
  ([04](04-diagnostico.md), seção 8.1).
- **Não linearidade de `pct_pretos_pardos`** (única material: P95 de \|Δπ̂\| 0,66 e
  0,65 p.p., máximo 1,64). Correção: spline nessa covariável, a validar por validação
  cruzada ([04](04-diagnostico.md), seção 8.2).
- **VIF > 10 nas defasagens SLX do `bb_slx`** (`w_pct_pretos_pardos` 21,5, `w_lag` 18,4,
  `w_cobertura_pbf` 17,2, `pct_pretos_pardos` 17,0, `w_pct_16_24` 12,5, `lag` 12,0; número
  de condição 19,5). Efeito: leitura de coeficientes individuais frágil, inclusive a tabela
  de coeficientes do README da raiz. Correção: ler os pares (x, Wx) por Wald de 2 gl
  ([04](04-diagnostico.md), seção 8.6).

## Validação

- **Cobertura dos intervalos (27/27 UFs, 5/5 regiões) é otimista**: σ_UF = 0,0564 foi
  ajustado nas mesmas UFs; com σ = 0 a cobertura é 14/27 no modelo final com Conley
  ([09](09-modelo-final.md), seção 4). Correção: σ_UF por validação aninhada.
- **Erro regional LOSO favorece binomiais canônicas por construção** ([05](05-validacao-cruzada.md), seção 6);
  não serve para seleção.
- **13 modelos dentro de 1 EP**: os dados não os separam com clareza [deduced]. A escolha
  de `bb_selecao` repousa na regra de parcimônia, não numa vantagem demonstrada.
  O modelo final `bb_slx` (LOSO 1,078) fica acima de `bb_selecao` (1,035), dentro da
  tolerância de 1,2515.

- **[resolvida] LOSO vazava o 2º turno da UF retida** (comparecimento e n reais de 2026).
  Agora o modelo usa só covariáveis anteriores ao turno e o LOSO pesa a UF retida por n̂
  ([09](09-modelo-final.md), seção 1); as métricas das Partes 11 e 13 a 17 são ponderadas
  por n̂. O limite de aceitação passou de 1,1308 para 1,1695 e, com a seleção aninhada, para 1,2515.
- **σ_UF (0,0564) também é ajustado nas mesmas 27 UFs**; mesma limitação da cobertura
  otimista acima. Correção: validação aninhada.
- **[resolvida] Seleção de variáveis e fusão Nordeste + Norte usavam o 2º turno de todas as UFs**,
  o que tornava otimista o LOSO do `bb_selecao` (1,0348). Refeitas dentro de cada dobra: 1,1168;
  a escolha não muda e L passa a 1,2515 ([05](05-validacao-cruzada.md), seção 8).
- **Extensões com a seleção dos dados completos.** `bb_disp`, `bb_slx` e as rejeitadas herdam
  as variáveis do `bb_selecao` escolhidas com todas as UFs; contra um L aninhado o limite é
  mais folgado. Nenhuma decisão está perto dele (`bb_slx` 1,078 contra 1,2515). Correção:
  refazer a seleção dentro de cada dobra para as extensões.
- **[resolvida] Cenário de apuração parcial (ordem real das seções)**: Parte 18, em
  [11](11-apuracao-parcial.md).
- **σ_UF e corte de Conley de 2026 reusados na apuração parcial de 2022** (sem LOSO próprio).
  Correção: ajustar σ_UF e o corte para 2022 ([11](11-apuracao-parcial.md)).

- **Krigagem: o corte do correlograma (CORTE) vem dos resíduos dentro da amostra com todos
  os dados** (Parte 15, c. 20) e é reusado como alcance da krigagem (c. 33). Efeito:
  vazamento leve que favorece aceitar a krigagem; ela foi rejeitada, então a decisão não
  muda ([10](10-extensoes-rejeitadas.md), seção 3). Correção: estimar o corte dentro de
  cada dobra.
- **[resolvida] ρ e o CAR usavam resíduos sem peso, embora a variância deles seja cerca de
  1/n**; o I de Moran fora da amostra também. Sensibilidade com resíduos de Pearson
  (beta-binomial), só informativa: nenhuma decisão muda. A queda do Moran fica abaixo de 0,05
  (krigagem −0,009, CAR 0,025) e o MAE por UF passa nas duas versões
  ([10](10-extensoes-rejeitadas.md), seção 6).
- **Moran sem peso superestima a correlação residual.** Padronizado, o I de Moran base cai de
  0,4224 para 0,3728: parte da dependência espacial vem de municípios pequenos. Com peso, o MAE
  regional piora (krigagem 0,624 contra 0,428 p.p.; CAR 0,607 contra 0,563).
- **[resolvida] Intervalos da apuração parcial tratavam n_u como fixo.** n_u passou a ser
  sorteado (choque de UF mais resíduo municipal, só com seções recebidas) e a cobertura é
  reportada por unidade e ano. Resultado: nenhuma cobertura mudou e as larguras variam menos de
  0,05 p.p. ([11](11-apuracao-parcial.md), seções 3 e 6).
- **Descasamento de θ na apuração parcial**: θ é avaliado em log(n_u), fora da faixa de
  log n do ajuste (n apurado); o erro de θ é ignorado (só o bloco β da covariância de
  Conley é sorteado); σ_UF da Parte 16 é reusado, não reestimado por corte
  ([11](11-apuracao-parcial.md), seção 8).

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
- **[resolvida] Saídas que variavam entre execuções.** `l1_logit` (solver `liblinear` sem
  semente, coeficientes ±1e-5) ganhou `random_state=0`. O `z_sim` de `diag_moran.csv` (999
  permutações do esda sem semente) ganhou semente fixa. As figuras (antes HTML, hoje PNG) não
  guardam mais id aleatório. Regra: duas
  execuções seguidas dos notebooks não mudam nenhum arquivo de `results/`.
- **[resolvida] Votos apurados contados duas vezes nos intervalos.** Nos municípios fora do
  modelo, o termo fixo somava y_c, que já estava no apurado. Só afetava 2026 (Boa Esperança do
  Norte): limites de Centro-Oeste e Brasil até 0,01 p.p.; nenhuma cobertura mudou
  ([11](11-apuracao-parcial.md), seção 6).
- **Envelope da beta-binomial no 02-MLG consome a maior parte dos 7,5 min** (49 reajustes
  de ~4 s). Correção: partida a quente (warm start) e remover o passo de polimento.
- **[resolvida] Checagem de tipos (`ty`).** Rodou em `src/`: 28 diagnósticos, depois 15. A
  primeira leitura ("os 15 são lacunas dos stubs") estava errada: 7 eram do nosso código
  (Optional sem estreitamento em `exog_names` e na ligação, assinatura de `fit` diferente da
  do statsmodels, retorno de `_logit`) e foram corrigidos. Restam 3 `ty: ignore` com motivo,
  todos de anotação de biblioteca: `LinearOperator` do scipy e `read_json(dtype=False)` do
  pandas (2). `ty check src/` dá 0 diagnósticos.
- **[resolvida] Referência do SLX.** Halleck Vega e Elhorst (2015) conferida (DOI
  10.1111/jors.12188) e incluída em `referencias.md` e em
  [08](08-dependencia-espacial.md), seção 3.
- **Agrupamento por UF não feito.** A seleção aninhada funde só níveis de região e de bloco de
  prefeito; UFs com comportamento parecido não são agrupadas. Correção: testar a fusão de UFs
  dentro de cada dobra, se o erro por UF justificar.
- **Binária `suplementar` (prefeito eleito em eleição suplementar).** Entra só se o LRT de adição
  der p <= 0,05 (`usa_suplementar`, `src/especificacoes.py`). Município sem o valor é tratado
  como 0 (não suplementar). Conferir quantos municípios estão sem o valor e se tratá-los como 0
  distorce o coeficiente.
