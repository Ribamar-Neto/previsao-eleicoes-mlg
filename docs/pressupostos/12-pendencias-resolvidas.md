# Pendências resolvidas

Histórico das pendências de [12](12-pendencias.md) resolvidas ou fechadas como limitação. Movidas em 2026-11-28 com
o texto original: os números são os da data em que cada item foi fechado, não os atuais (os atuais estão nos docs
citados). Nova pendência resolvida entra aqui, na seção de origem.

## Modelo

- **[resolvida] Triagem das covariáveis, funil registrado.** 24 numéricas na AED → 20 após o
  VIF ([02](02-aed.md), Parte 3) → 21 candidatas do modelo (`CONTINUAS` em
  `src/especificacoes.py`: as 20 mais `nota_dep`, que a AED descartou por medir o mesmo que
  `pct_dep_esq` e que a seleção volta a testar) → 13 selecionadas por BIC (`lag` e 12).
  As três candidatas que antes ficavam fora (`pct_ate_fund_incompleto_2026`,
  `pct_sem_religiao_2022`, `pct_indigenas_2022`) agora entram e as três são selecionadas.
  Nenhuma variável recomendada pela AED fica sem avaliação.
  *Contagem atualizada em 2026-11-28:* com `lag1t` e `outros_1t` são 22 contínuas candidatas além do `lag` e 17
  selecionadas, 18 com o `lag`; `lag1t` entra e `outros_1t` não; `pct_sem_religiao` e `pct_indigenas` saem
  ([03](03-modelos-candidatos.md), Parte 8).
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
- **[resolvida em 2026-11-28] Não linearidade de `pct_pretos_pardos`** (única material: P95 de \|Δπ̂\| 0,64 e
  0,63 p.p., máximo 1,35 e 1,52). Correção: spline nessa covariável, a validar por validação
  cruzada ([04](04-diagnostico.md), seção 8.2).
  *Fechamento:* com `lag1t` no modelo, `nao_linear` é falso em todas as covariáveis e o P95 de `pct_pretos_pardos`
  é 0,10 p.p. nos dois modelos, abaixo do corte de 0,5 (`results/tables/diag_final_linearidade.csv`).

## Validação

- **[resolvida] Cobertura dos intervalos (27/27 UFs, 5/5 regiões) com σ_UF ajustado nas mesmas UFs.**
  Com σ_UF aninhado (σ_{-u} das outras 26 UFs, de 0,0529 a 0,0581; σ_{-r} das UFs fora da região)
  a cobertura é a mesma: 27/27 a 95 %, 92,6 % a 90 %, 5/5 regiões; largura 5,41 p.p. por UF e 2,84
  por região (contra 5,42 e 2,87). Resta só o tamanho da amostra (5/5 é compatível com cobertura real
  a partir de ~55 %); com σ = 0 a cobertura é 13/27 ([09](09-modelo-final.md), seção 4).
- **[resolvida] LOSO vazava o 2º turno da UF retida** (comparecimento e n reais de 2026).
  Agora o modelo usa só covariáveis anteriores ao turno e o LOSO pesa a UF retida por n̂
  ([09](09-modelo-final.md), seção 1); as métricas das Partes 11 e 13 a 17 são ponderadas
  por n̂. O limite de aceitação passou a 1,1358 (LOSO sem vazamento, valor da época) e, com a seleção aninhada, a 1,2088.
- **[resolvida] σ_UF ajustado nas mesmas 27 UFs** (0,0570 na época, antes da covariável do 1º turno; hoje 0,0114,
  [09](09-modelo-final.md)). Conferido com σ aninhado, sem efeito na cobertura (item acima).
- **[resolvida] Seleção de variáveis e fusão Nordeste + Norte usavam o 2º turno de todas as UFs**,
  o que tornava otimista o LOSO do `bb_selecao` (1,005). Refeitas dentro de cada dobra: 1,067;
  a escolha não muda e L passa a 1,2088 ([05](05-validacao-cruzada.md), seção 8).
- **[resolvida] Cenário de apuração parcial (ordem real das seções)**: Parte 18, em
  [11](11-apuracao-parcial.md).
- **[resolvida] σ_UF e corte de Conley de 2026 reusados na apuração parcial de 2022.** Cada
  (ano, corte) tem agora o seu, só do apurado ([11](11-apuracao-parcial.md), seção 9).
- **[resolvida] Piso de 1 km do corte de Conley nos cortes iniciais da apuração parcial.** Conley com
  1 km equivale a HC0 e ignora a correlação espacial. Quando a regra da Parte 15 dá 0 km num desenho
  parcial, o corte passa a 200 km, o da regra aplicada à apuração completa de 2022 (conhecida antes
  do 2º turno de 2026, sem vazamento). Só 2026 a 5 % e 10 % usaram o piso
  (com o 1º turno do mesmo ano nenhum corte precisou; [11](11-apuracao-parcial.md), seção 8).
- **[resolvida] ρ e o CAR usavam resíduos sem peso, embora a variância deles seja cerca de
  1/n**; o I de Moran fora da amostra também. Sensibilidade com resíduos de Pearson
  (beta-binomial), só informativa: nenhuma decisão muda. A queda do Moran fica abaixo de 0,05
  (krigagem −0,001, CAR 0,024) e o MAE por UF passa nas duas versões
  ([10](10-extensoes-rejeitadas.md), seção 6).
- **[resolvida] Intervalos da apuração parcial tratavam n_u como fixo.** n_u passou a ser
  sorteado (choque de UF mais resíduo municipal, só com seções recebidas) e a cobertura é
  reportada por unidade e ano. Resultado: nenhuma cobertura mudou; as larguras variavam menos de
  0,05 p.p. antes do 1º turno do mesmo ano e até 0,15 p.p. (mediana) agora ([11](11-apuracao-parcial.md), seções 3 e 6).
- **[resolvida] Swing do 1º turno como linha de base do veredito.** Entrou como terceira linha de base (e como linha
  tracejada no gráfico de erro): 2022 passa a não útil (3 de 6), 2026 segue útil (6 de 6). A regra mudou depois do
  resultado, o que o [11](11-apuracao-parcial.md) (seção 1) declara.
- **[resolvida] Crista plana da dispersão do Centro-Oeste em 2026 a 10 %.** Teto de θ = 1e9 no limite binomial, com guarda
  LRT; o ajuste converge ([14](14-apuracao-diagnostico-1t.md), seção 6).
- **[resolvida] σ_UF e choque de UF.** O choque passou a σ̂·T_k, com k as UFs do desenho do corte ([11](11-apuracao-parcial.md),
  seção 6).
- **[resolvida] Fusão de região com a referência no 02.** O 02 passou a usar o mesmo agrupamento de `_fusoes` do 03.
- **[resolvida em 2026-11-28] Contagem do funil de covariáveis desatualizada.** O item "Triagem" acima conta 13
  selecionadas; com `lag1t` e `outros_1t` como candidatas são 18 contínuas selecionadas
  (`results/tables/selecao_variaveis.json`). *Fechamento:* o [03](03-modelos-candidatos.md) (Parte 8) já traz a
  contagem atual, e o item "Triagem" ganhou a nota de atualização.

## Backtest

Histórico: a Parte 12 foi removida em 2026-11-23 ([06](06-backtest.md)); os três itens perderam o objeto.

- **Sem perfil do eleitorado de 2018**: o ajuste usa o perfil de 2022.
- **Censo 2022 (publicado em 2023)** entra no ajuste de 2018: pequeno look-ahead.
- **Modelo entre turnos sem recalibração erra o nível nacional em −4,7 p.p.**; a causa não
  foi testada. Correção: testar efeito de turno e de abstenção diferencial.

## Código/testes

- **[resolvida] Teste de convergência fraco.** `tests/test_convergencia.py` usa dados reais
  (`data/processed/municipios.parquet`; pulado se ausente): a tolerância antiga (`gtol` =
  1e-4·N) para a ~0,3 do ótimo polido nos parâmetros, quase tudo em log θ (θ ~35 % fora); o padrão
  fica a 3e-7 de llf. A premissa do teste é essa distância, não a perda de llf, que depende do
  condicionamento: 398, depois 150 no conjunto anterior e 41 com `lag` e `lag1t` colineares.
  O teste sintético foi removido de `tests/test_modelos.py`.
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
- **[resolvida] `mypy` em `src/pressupostos.py`:** sem erros.
- **[resolvida, histórico] Checagem de tipos (`ty`).** Rodou em `src/`: 28 diagnósticos, depois 15. A
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
  Os p-valores dos passos de fusão (0,236; 0,474; 0,083) estão nas Tabelas 8.11 e 8.12 do `02-MLG`
  (`results/tables/fusao_regiao_passos.csv` e `fusao_bloco_passos.csv`) e reproduzem os valores antigos.
- **[resolvida] Inconsistências no markdown do `02-MLG`.** A Parte 6 cita o `bb_selecao` como o melhor, como o CSV;
  "11 contínuas" ficou só para o conjunto antigo.
- **[resolvida] Texto com números conferido.** Corrigidos a tabela de coeficientes do README, os resultados finais,
  a tabela regional e o MAE do [01](01-visao-geral.md), o σ_UF desta lista, o SP do [08](08-dependencia-espacial.md)
  (5,6 p.p. e 81 %, recalculados) e a regra de n̂ do [09](09-modelo-final.md). Bateram com `results/` as células 26,
  28, 33, 82 e 146 do `02-MLG`, θ, ρ e fusão do [03](03-modelos-candidatos.md), as seções 5 a 8 do
  [05](05-validacao-cruzada.md), as larguras de BA e SP do [09](09-modelo-final.md) e a tabela regional do README.
- **[resolvida em 2026-11-28] Binária `suplementar` (prefeito eleito em eleição suplementar).** Entra só se o LRT de
  adição der p <= 0,05 (`usa_suplementar`, `src/especificacoes.py`). Município sem o valor é tratado
  como 0 (não suplementar). Conferir quantos municípios estão sem o valor e se tratá-los como 0
  distorce o coeficiente.
  *Fechamento:* 16 de 5.570 municípios estão sem o valor (27 suplementares). A binária não entra: LRT p = 0,727,
  razão de chances 0,998 (Tabela 8.5 do `02-MLG`; `usa_suplementar` falso em `results/tables/final_decisao.json`).
  Sem o termo no modelo, a codificação dos 16 não afeta nenhuma estimativa; [deduced] com p = 0,727, recodificar
  16 casos não levaria o LRT a p ≤ 0,05.
