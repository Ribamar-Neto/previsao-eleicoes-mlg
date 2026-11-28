# Pendências

Todas são [fact] salvo marca em contrário. Conferidas em 2026-11-28 com o código, `results/` e os docs 03 a 15.
Cada item: efeito e correção possível. As resolvidas ficam em [12-pendencias-resolvidas.md](12-pendencias-resolvidas.md).

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
- **`pct_vab_agro` usa 2021** e **`pib_per_capita_2023` divide pela população de 2022**
  ([dicionário](../dados/dicionario.md)). Efeito: defasagem de cinco anos e denominador aproximado. Correção:
  atualizar quando o IBGE publicar VAB posterior e estimativas de população.
- **Classificação ideológica de UP e Missão sem citação** (não constam em Bolognesi,
  Ribeiro e Codato 2023). UP em esquerda e Missão em direita é julgamento do projeto, registrado em
  [ideologia-partidos](../dados/ideologia-partidos.md) (`CLASSE_MANUAL` em `src/coleta/tse_ideologia.py`), sem fonte
  acadêmica. Efeito: bloco de poucos votos com base fraca. Correção: citar fonte ou agrupar como "outros".

## Modelo

- **Regras adicionadas depois dos primeiros resultados da Parte 13** (Wald robusto, SLX e
  corte de 2022, em 2026-11-09): foram fixadas antes de cada resultado, mas não antes do
  notebook inteiro; o leitor deve pesar isso.
- **Fusão de Nordeste e Norte na média e rodada de seleção de variáveis** (BIC, AIC,
  Hosmer-Lemeshow e todos os subconjuntos) vieram da revisão da análise, depois dos
  resultados anteriores; o leitor deve pesar isso. A busca em todos os subconjuntos
  (triagem quase-binomial e reajuste dos 200 melhores) não achou o ótimo do stepwise (BIC
  62.571,7 com 16 variáveis contra 62.571,4, 0,3 ponto abaixo; [03](03-modelos-candidatos.md), Parte 8): fraqueza
  da triagem, que não muda a escolha. Correção possível: ampliar a triagem.
- **Stepwise de Hosmer-Lemeshow difere do livro em um ponto** (Paula, pp. 358-371): os
  limiares, a ordem das etapas, o passo 4 (Wald, alfa 0,05) e a tabela de π por iteração
  conferem. A etapa de interações testa só o `lag` e as 6 primeiras variáveis que
  sobreviveram ao Wald, por custo. Efeito: o stepwise é comparação, não decide o modelo (BIC
  decide). Correção possível: testar todos os pares, o que passou de 20 minutos sem terminar.
- **Envelope meio-normal fora da banda nos beta-binomiais**: `bb_selecao` 94,6 % e `bb_slx` 72,8 % dos
  pontos fora (corpo abaixo: 92,2 e 71,7 %; caudas acima: 2,44 e 1,06 %; curtose em excesso 2,27 e 2,25). A
  dispersão por região não muda a curtose e o padrão persiste. As contagens vêm de B = 49 e de uma só realização
  dos resíduos aleatorizados, então variam com a semente ([04](04-diagnostico.md), seção 7). Efeito: o envelope
  não valida a distribuição. Correção: modelo com caudas mais pesadas ([04](04-diagnostico.md), seção 8.3).
- **Influência no beta-binomial**: os maiores Cook de um passo são municípios pequenos; Belo Monte/AL e Santo
  Antônio de Lisboa/PI lideram nos dois modelos. O reajuste sem os 10 maiores move no máximo 1,37 EP
  (`bb_selecao`, `log_theta`) e 1,05 EP (`bb_slx`, `disp_reg_Nordeste`), sem troca de sinal. Efeito: inferência
  estável pela regra 7; a sensibilidade do quase-binomial às capitais fica restrita a ele
  ([04](04-diagnostico.md), seções 4 e 8.7).
- **Ligação sinalizada nos dois modelos** (p Conley 1,3e-4 no `bb_selecao` e 4,6e-3 no `bb_slx`; P95 de
  \|Δπ̂\| 0,13 e 0,09 p.p., máximo 0,25 e 0,19). Efeito: pequeno, abaixo de 0,5 p.p. Correção: ligação
  paramétrica ou η̂² na média, validado por validação cruzada ([04](04-diagnostico.md), seção 8.1).
- **VIF > 10 no `bb_slx`**: 11 de 38 fatores (`w_lag1t` 251,7, `w_lag` 228,7, `lag1t` 114,4, `lag` 110,5,
  `w_pct_pretos_pardos` 23,7 e outros seis entre 10 e 18,4); número de condição 82,4, 41 colunas. Efeito: leitura
  de coeficientes individuais frágil, inclusive a tabela de coeficientes do README da raiz. Correção: ler os pares
  (x, Wx) por Wald de 2 gl, ainda não feito ([04](04-diagnostico.md), seção 8.6).

## Validação

- **Erro regional LOSO favorece binomiais canônicas por construção** ([05](05-validacao-cruzada.md), seção 6);
  não serve para seleção.
- **4 modelos dentro de 1 EP no registro, 10 na regra aninhada** ([05](05-validacao-cruzada.md), seções 5 e
  8): os dados não os separam com clareza [deduced]. A escolha de `bb_selecao` repousa na regra de
  parcimônia, não numa vantagem demonstrada. O modelo final `bb_slx` (LOSO 0,1940) fica abaixo de `bb_selecao`
  (0,2134) e dentro de L = 0,2904 ([08](08-dependencia-espacial.md), seções 1 e 3).
- **Extensões com a seleção dos dados completos.** `bb_disp`, `bb_slx` e as rejeitadas herdam
  as variáveis do `bb_selecao` escolhidas com todas as UFs; contra um L aninhado o limite é
  mais folgado. Nenhuma decisão está perto dele (`bb_slx` 0,194 contra 0,290; [05](05-validacao-cruzada.md),
  seção 8). Correção: refazer a seleção dentro de cada dobra para as extensões.
- **L antigo no [01](01-visao-geral.md).** A regra 5 de seleção ainda cita L = 1,0669 + 0,1419 = 1,2088 p.p.
  (1,1472 sem aninhar); o L atual é 0,2497 + 0,0407 = 0,2904 (0,2541 sem aninhar; [05](05-validacao-cruzada.md)).
  Correção: atualizar o 01.
- **Krigagem: o corte do correlograma (CORTE) vem dos resíduos dentro da amostra com todos
  os dados** (Parte 15, c. 20) e é reusado como alcance da krigagem (c. 33). Efeito:
  vazamento leve que favorece aceitar a krigagem; ela foi rejeitada, então a decisão não
  muda ([10](10-extensoes-rejeitadas.md), seção 3). Correção: estimar o corte dentro de
  cada dobra.
- **Moran sem peso superestima a correlação residual.** Padronizado, o I de Moran base cai de
  0,4394 para 0,3882: parte da dependência espacial vem de municípios pequenos. Com peso, o MAE
  regional piora (krigagem 0,630 contra 0,433 p.p.; CAR 0,620 contra 0,585).
- **Descasamento de θ na apuração parcial**: θ é avaliado em log(n_u), fora da faixa de
  log n do ajuste (n apurado); o erro de θ é ignorado (só o bloco β da covariância de
  Conley é sorteado); σ_UF vem do LOSO do desenho de cada corte
  ([11](11-apuracao-parcial.md), seções 8 e 9). O ruído binomial/θ é de 20 a 57 % da variância do resíduo LOSO
  que sustenta σ_UF; subtraí-lo (H1) não fechou a lacuna de calibração ([15](15-apuracao-corte-precoce-calibracao.md),
  seções 3.2 e 3.3).
- **[resolvida em parte] IC largo na apuração parcial.** O diagnóstico D1 a D4 mostrou que o σ_UF inflado, por falta do
  1º turno do mesmo ano, era a causa principal; com a covariável o IC do Brasil caiu de 1,82 para 0,39 p.p. e a cobertura
  foi de 56 de 60 para 60 de 60 ([14](14-apuracao-diagnostico-1t.md)). Resta que as primeiras seções (capitais e grandes
  cidades) não são amostra aleatória da UF. O encolhimento por UF (BLUP) estreitou o IC e perdeu cobertura em 2026
  (Brasil 6/6 → 4/6, regiões 27/30 → 20/30) e foi rejeitado, mas no modelo de 2026-11-26, sem a covariável do 1º
  turno; não foi retestado depois dela ([13](13-apuracao-extensoes-rejeitadas.md)). Correção possível: encolhimento
  que pondere o apurado pela semelhança com o restante da UF, calibrado com critério que inclua a cobertura.
- **Cobertura do IC da apuração parcial não calibrada.** 72 de 72 itens dentro (z² médio 0,52, calibrado ≈ 1); não
  separa 95 % de valores maiores. Diagnosticada em 2026-11-27 ([15](15-apuracao-corte-precoce-calibracao.md), seções 3
  e 4): a regra R-cal não disparou e o IC não mudou. A condição (a) falha em todo componente e a (b) também; H1 a H3
  fecham pouco da lacuna. Explicação líder, não testada diretamente: o DP total modelado da parcela restante da UF fica
  superestimado quando pesado pelos votos restantes (ρ total 0,58 em 2022 e 0,70 em 2026; sem peso, 0,91 e 0,64).
  Viés de n_u em 2026: votos restantes ≈ 4 % acima da estimativa (razão em log de 0,040 a 0,049), não modelado. No
  corte de 2 %, fora do veredito, o final de 2026 fica ≈ 0,02 p.p. acima do IC do Brasil, e Sergipe não tem município
  apurado nos dois anos. Correção: σ_UF pesado pelos votos ou dependente do tamanho da UF, sob nova regra
  pré-registrada; de fundo, mais eleições ([11](11-apuracao-parcial.md), seção 6).
- **Bloco de resíduos negativos sem leitura.** A figura de influência do modelo final tem um bloco de resíduos
  negativos nos índices ~1850 a 2100. As linhas 1850 a 2100 de `data/processed/municipios.parquet` são todas da BA
  (251 de 251) e o ajuste mantém a ordem do parquet (`carregar_base` em `src/dados.py` não reordena); [deduced] o
  bloco é da BA. Nenhuma leitura o comenta. Correção: comentar na leitura da figura.

## Código/testes

- **Envelope da beta-binomial no 02-MLG consome a maior parte dos 7,5 min** (49 reajustes
  de ~4 s). Correção: partida a quente (warm start) e remover o passo de polimento.
- **Logit empírico sem citação em Paula.** A escolha do logit empírico (`log((y+½)/(n−y+½))`) como linha de base e para
  `lag1t` cita McCullagh e Nelder ([referências](referencias.md)); a seção de Paula que a sustenta não foi
  identificada. Também falta conferir se Paula 4.6.6 (citada no notebook 03 para a variância de β̂ e em `src/` para a
  seleção dentro do município) é a seção certa nos dois usos.
- **Agrupamento por UF não feito.** A seleção aninhada funde só níveis de região e de bloco de
  prefeito; UFs com comportamento parecido não são agrupadas. Correção: testar a fusão de UFs
  dentro de cada dobra, se o erro por UF justificar.
- **Higiene do repositório.** O lint do repositório (`make lint-all`, regras padrão do ruff) passa. As regras de nome
  do pep8-naming (`N`), fora da configuração, acusam 362 achados em `src/` e `tests/`, quase todos notação
  estatística (`X`, `N`, `L`, `PI`), e o `pyproject.toml` não registra se valem. O repositório não tem CI. O
  `02-MLG` não foi reexecutado depois do commit de estilo c97afed (só `src/` e `tests/`), então `results/` não foi
  conferido contra o código atual. Correção: registrar a exceção de nomes com motivo (`ignore` no `pyproject.toml`),
  CI com `make lint test` e reexecutar o 02.
- **Diagnóstico da calibração fora do repositório.** Os scripts e as saídas da R-cal (`rho_uf.csv`, `h1_ruido.csv`,
  `pit.csv` e outros) rodaram fora do repositório e não estão versionados
  ([15](15-apuracao-corte-precoce-calibracao.md), seção 5): os números das seções 3 e 4 do 15 não se refazem a partir
  do código. Correção: versionar o diagnóstico em `src/` e as tabelas em `results/tables/`.
