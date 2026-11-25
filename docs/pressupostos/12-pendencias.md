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

- **[resolvida] Triagem das covariáveis, funil registrado.** 24 numéricas na AED → 20 após o
  VIF ([02](02-aed.md), Parte 3) → 21 candidatas do modelo (`CONTINUAS` em
  `src/especificacoes.py`: as 20 mais `nota_dep`, que a AED descartou por medir o mesmo que
  `pct_dep_esq` e que a seleção volta a testar) → 13 selecionadas por BIC (`lag` e 12).
  As três candidatas que antes ficavam fora (`pct_ate_fund_incompleto_2026`,
  `pct_sem_religiao_2022`, `pct_indigenas_2022`) agora entram e as três são selecionadas.
  Nenhuma variável recomendada pela AED fica sem avaliação.
- **[fechada como limitação] Resíduos com autocorrelação espacial em todos os modelos**
  (I de Moran ≈ 0,35). Conley a 200 km corrige a inferência
  ([08](08-dependencia-espacial.md), seções 4 e 5); ESF foi rejeitado (LOSO 1,575 contra o
  limite 1,2088) e o SLX aceito, mas o SLX só baixou o I de 0,352 para 0,347. A Parte 17
  decompôs o I e testou krigagem dos resíduos sob LOSO: rejeitada (queda do I de 0,0006
  contra o mínimo de 0,05). Testou também CAR/BYM em dois estágios sob a mesma regra:
  rejeitado (queda do I de 0,0304; MAE por UF de 0,984, mas MAE regional pior)
  ([10](10-extensoes-rejeitadas.md)). A correlação não é removível com a informação
  disponível para uma UF nova. Não resta alternativa espacial na lista.
- **[resolvida] θ não é constante por região** (DP residual de 0,82 no Sudeste a 1,39 no
  Norte). Dispersão por região e log n: máximo |DP − 1| de 0,388 para 0,075
  ([07](07-dispersao-regional.md)). Resta o Norte (DP 1,08).
- **[resolvida] LRT da dispersão supõe municípios independentes** ([07](07-dispersao-regional.md),
  seção 3). O Wald robusto com Conley (e por UF) mantém `regiao` e `log n`, com p ≤ 2e-4
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
  72.630,6 contra 72.624,5): fraqueza da triagem. Correção possível: ampliar a triagem.
- **Stepwise de Hosmer-Lemeshow difere do livro em um ponto** (Paula, pp. 358-371): os
  limiares, a ordem das etapas, o passo 4 (Wald, alfa 0,05) e a tabela de π por iteração
  conferem. A etapa de interações testa só o `lag` e as 6 primeiras variáveis que
  sobreviveram ao Wald, por custo. Efeito: o stepwise é comparação, não decide o modelo (BIC
  decide). Correção possível: testar todos os pares, o que passou de 20 minutos sem terminar.
- **Envelope meio-normal fora da banda em todos os beta-binomiais**: `bb_reglag` 92,4%,
  `bb_selecao` 92,7%, `bb_slx` 77,8% dos pontos fora (corpo abaixo, caudas acima; curtose
  3,67 / 3,77 / 2,24). A dispersão por região melhora a curtose, mas o padrão persiste. Efeito:
  o envelope não valida a distribuição. Correção: modelo com caudas mais pesadas
  ([04](04-diagnostico.md), seção 8.3).
- **Influência no beta-binomial**: os maiores Cook de um passo são municípios pequenos
  (Jutaí/AM, São João das Missões/MG), com Brasília só em 3º no `bb_slx`. O reajuste sem os
  10 maiores move no máximo 0,69 EP, sem troca de sinal. Efeito: inferência estável; a sensibilidade do quase-binomial
  às capitais fica restrita a ele ([04](04-diagnostico.md), seções 4 e 8.7).
- **Ligação sinalizada só em `bb_slx`** (p Conley 0,0076; P95 de \|Δπ̂\| 0,25 p.p.,
  máximo 0,41); `bb_selecao` não é sinalizado (p Conley 0,098). Efeito: pequeno, abaixo de 0,5 p.p. Correção: ligação
  paramétrica ou η̂² na média, validado por validação cruzada
  ([04](04-diagnostico.md), seção 8.1).
- **Não linearidade de `pct_pretos_pardos`** (única material: P95 de \|Δπ̂\| 0,64 e
  0,63 p.p., máximo 1,35 e 1,52). Correção: spline nessa covariável, a validar por validação
  cruzada ([04](04-diagnostico.md), seção 8.2).
- **VIF > 10 nas defasagens SLX do `bb_slx`** (`w_pct_pretos_pardos` 28,5, `pct_pretos_pardos` 24,0,
  `w_cobertura_pbf` 17,4, `w_lag` 16,5, `w_pct_16_24` 13,7, `lag` 11,7, `w_log_dens` 10,1;
  número de condição 21,3, 32 colunas). Efeito: leitura de coeficientes individuais frágil, inclusive a tabela
  de coeficientes do README da raiz. Correção: ler os pares (x, Wx) por Wald de 2 gl
  ([04](04-diagnostico.md), seção 8.6).

## Validação

- **Cobertura dos intervalos (27/27 UFs, 5/5 regiões) é otimista**: σ_UF = 0,0570 foi
  ajustado nas mesmas UFs (a cobertura IC90 por UF caiu de 96,3 % para 92,6 % com a rodada atual);
  com σ = 0 a cobertura é 13/27 no modelo final com Conley
  ([09](09-modelo-final.md), seção 4). Correção: σ_UF por validação aninhada.
- **Erro regional LOSO favorece binomiais canônicas por construção** ([05](05-validacao-cruzada.md), seção 6);
  não serve para seleção.
- **13 modelos dentro de 1 EP**: os dados não os separam com clareza [deduced]. A escolha
  de `bb_selecao` repousa na regra de parcimônia, não numa vantagem demonstrada.
  O modelo final `bb_slx` (LOSO 1,073) fica acima de `bb_selecao` (1,005), dentro da
  tolerância de 1,2088.

- **[resolvida] LOSO vazava o 2º turno da UF retida** (comparecimento e n reais de 2026).
  Agora o modelo usa só covariáveis anteriores ao turno e o LOSO pesa a UF retida por n̂
  ([09](09-modelo-final.md), seção 1); as métricas das Partes 11 e 13 a 17 são ponderadas
  por n̂. O limite de aceitação passou a 1,1358 (LOSO sem vazamento, valor da época) e, com a seleção aninhada, a 1,2088.
- **σ_UF (0,0570) também é ajustado nas mesmas 27 UFs**; mesma limitação da cobertura
  otimista acima. Correção: validação aninhada.
- **[resolvida] Seleção de variáveis e fusão Nordeste + Norte usavam o 2º turno de todas as UFs**,
  o que tornava otimista o LOSO do `bb_selecao` (1,005). Refeitas dentro de cada dobra: 1,067;
  a escolha não muda e L passa a 1,2088 ([05](05-validacao-cruzada.md), seção 8).
- **Extensões com a seleção dos dados completos.** `bb_disp`, `bb_slx` e as rejeitadas herdam
  as variáveis do `bb_selecao` escolhidas com todas as UFs; contra um L aninhado o limite é
  mais folgado. Nenhuma decisão está perto dele (`bb_slx` 1,073 contra 1,2088). Correção:
  refazer a seleção dentro de cada dobra para as extensões.
- **[resolvida] Cenário de apuração parcial (ordem real das seções)**: Parte 18, em
  [11](11-apuracao-parcial.md).
- **[resolvida] σ_UF e corte de Conley de 2026 reusados na apuração parcial de 2022.** Cada
  (ano, corte) tem agora o seu, só do apurado ([11](11-apuracao-parcial.md), seção 9).
- **Piso de 1 km do corte de Conley nos cortes iniciais da apuração parcial** (2026 a 5 % e,
  na etapa R1, a 10 %): Conley com 1 km equivale a HC0 e ignora a correlação espacial, então
  o IC desses cortes tende a ser estreito demais quanto à dependência espacial.
  Correção: testar outro piso ou outro estimador para amostras apuradas pequenas
  ([11](11-apuracao-parcial.md), seção 8).

- **Krigagem: o corte do correlograma (CORTE) vem dos resíduos dentro da amostra com todos
  os dados** (Parte 15, c. 20) e é reusado como alcance da krigagem (c. 33). Efeito:
  vazamento leve que favorece aceitar a krigagem; ela foi rejeitada, então a decisão não
  muda ([10](10-extensoes-rejeitadas.md), seção 3). Correção: estimar o corte dentro de
  cada dobra.
- **[resolvida] ρ e o CAR usavam resíduos sem peso, embora a variância deles seja cerca de
  1/n**; o I de Moran fora da amostra também. Sensibilidade com resíduos de Pearson
  (beta-binomial), só informativa: nenhuma decisão muda. A queda do Moran fica abaixo de 0,05
  (krigagem −0,001, CAR 0,024) e o MAE por UF passa nas duas versões
  ([10](10-extensoes-rejeitadas.md), seção 6).
- **Moran sem peso superestima a correlação residual.** Padronizado, o I de Moran base cai de
  0,4394 para 0,3882: parte da dependência espacial vem de municípios pequenos. Com peso, o MAE
  regional piora (krigagem 0,630 contra 0,433 p.p.; CAR 0,620 contra 0,585).
- **[resolvida] Intervalos da apuração parcial tratavam n_u como fixo.** n_u passou a ser
  sorteado (choque de UF mais resíduo municipal, só com seções recebidas) e a cobertura é
  reportada por unidade e ano. Resultado: nenhuma cobertura mudou e as larguras variam menos de
  0,05 p.p. ([11](11-apuracao-parcial.md), seções 3 e 6).
- **Descasamento de θ na apuração parcial**: θ é avaliado em log(n_u), fora da faixa de
  log n do ajuste (n apurado); o erro de θ é ignorado (só o bloco β da covariância de
  Conley é sorteado); σ_UF vem do LOSO do desenho de cada corte
  ([11](11-apuracao-parcial.md), seções 8 e 9).

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
- **[resolvida] Bloco do prefeito e fusão de níveis.** A fusão região/bloco vale só para a
  linhagem `bb_selecao` (passo de seleção); `bloco_outro` deixou de existir nela. A Parte 6 mantém
  `BLOCOS_PADRAO` de propósito: compara famílias numa especificação fixa ([03](03-modelos-candidatos.md), [09](09-modelo-final.md)).
  Os p-valores antigos do passo de fusão (0,236; 0,474; 0,083) não aparecem em nenhuma saída da
  rodada atual e não foram reproduzidos; quem citar esses valores deve tratá-los como de rodada anterior.
- **Inconsistências no markdown do `02-MLG` (a corrigir no notebook).** A célula da Parte 6 diz que
  `bin_probit` é o melhor (0,998; 13 de 18), contra o CSV e a célula seguinte (`bb_selecao`, 1,005); outra
  cita 11 contínuas, e são 12 além do lag.
- **Texto dos notebooks com números antigos.** Parte do markdown do `02-MLG` (ex.: seção 8, com
  p = 0,921 e 0,466, e a seção 8.3, com 11 contínuas e BIC 72.844,9) ainda cita a rodada
  anterior; as tabelas e figuras geradas estão certas. Correção: revisar o markdown contra as
  saídas.
- **Agrupamento por UF não feito.** A seleção aninhada funde só níveis de região e de bloco de
  prefeito; UFs com comportamento parecido não são agrupadas. Correção: testar a fusão de UFs
  dentro de cada dobra, se o erro por UF justificar.
- **Binária `suplementar` (prefeito eleito em eleição suplementar).** Entra só se o LRT de adição
  der p <= 0,05 (`usa_suplementar`, `src/especificacoes.py`). Município sem o valor é tratado
  como 0 (não suplementar). Conferir quantos municípios estão sem o valor e se tratá-los como 0
  distorce o coeficiente.
