# Pressupostos da análise

Índice. Leia este arquivo primeiro e depois só o arquivo da seção necessária.

| Arquivo | Conteúdo |
|---|---|
| [01-visao-geral.md](01-visao-geral.md) | Problema, decisões de projeto e razões, organização dos notebooks (funções em `src/`, cada valor e cada explicação uma vez, texto curto, gráficos interativos com PNG em `results/figures`, uma leitura por gráfico), explicar × prever, regra de seleção, resultados finais |
| [02-aed.md](02-aed.md) | Análise exploratória dos dados, incluindo a leitura nível a nível das categóricas e binárias (seção 3.1), o Norte dividido entre UFs e a regressão à média de 2022 ← 2018 |
| [03-modelos-candidatos.md](03-modelos-candidatos.md) | Famílias, ligações, regra de fusão de níveis (seção 8.1, com os p-valores dos passos e o rótulo da referência acumulando os níveis fundidos nela; só na linhagem `bb_selecao`; a Parte 6 mantém `BLOCOS_PADRAO`) e seleção de variáveis (`bb_selecao`, 17 contínuas além do lag; Hosmer-Lemeshow com passo 4 de Wald) |
| [04-diagnostico.md](04-diagnostico.md) | Resíduos, influência (cortes de Cook 4/N e 1 no gráfico), envelope e autocorrelação espacial; pressupostos dos modelos escolhidos (`bb_selecao` e `bb_slx`: ligação sinalizada só no `bb_slx`, linearidade, variância, colinearidade com 11 termos acima de 10, influência) |
| [05-validacao-cruzada.md](05-validacao-cruzada.md) | Validação cruzada (5 dobras e LOSO sem vazamento), regra de 1 EP, seleção aninhada do `bb_selecao`, limite L e sensibilidade regional |
| [06-backtest.md](06-backtest.md) | Removido (Parte 12, 2026-11-23): razão da remoção |
| [07-dispersao-regional.md](07-dispersao-regional.md) | Dispersão do beta-binomial por região (`bb_disp`; porte testado e descartado), com Wald robusto |
| [08-dependencia-espacial.md](08-dependencia-espacial.md) | Filtro de autovetores (rejeitado), EPs de Conley, SLX (aceito) e corte de 2022 |
| [09-modelo-final.md](09-modelo-final.md) | Regra de informação (covariáveis pré-turno, n̂), estimativas e cobertura do `bb_slx` (choque de UF t de Student, σ_UF também aninhado: mesma cobertura) |
| [10-extensoes-rejeitadas.md](10-extensoes-rejeitadas.md) | Krigagem e CAR/BYM nos resíduos (rejeitadas, também com peso por n), decomposição do I de Moran |
| [11-apuracao-parcial.md](11-apuracao-parcial.md) | Apuração parcial na ordem real do TSE (2022 e 2026), com o desenho refeito em cada corte (vazamento corrigido), 1º turno do mesmo ano (seção 10), veredito com três linhas de base (bruta, swing, swing do 1º turno: 2022 não útil, 3 de 6; 2026 útil, 6 de 6), IC com choque de UF t de Student, teto de θ no limite binomial, IC do Brasil em 12 de 12 e regiões em 60 de 60, corte de Conley pela regra (200 km quando a regra dá 0), incerteza de n_u e cobertura por unidade |
| [12-pendencias.md](12-pendencias.md) | Pendências abertas, com efeito e correção possível |
| [13-apuracao-extensoes-rejeitadas.md](13-apuracao-extensoes-rejeitadas.md) | Encolhimento por UF (BLUP), parcela apurada do município e desenho fixo de 2026 na apuração parcial (rejeitados: o IC estreita e perde cobertura); estado datado de 26/11, anterior ao `swing_1t` e ao choque t |
| [14-apuracao-diagnostico-1t.md](14-apuracao-diagnostico-1t.md) | Diagnóstico do IC largo da apuração parcial (D1 a D4), covariável do 1º turno do mesmo ano, regra de aceitação e resultado (HEAD contra agora), swing do 1º turno como linha de base do veredito, Hessiana analítica e teto de θ (crista de 2026 a 10 % resolvida) |
| [referencias.md](referencias.md) | Referências únicas e conferidas (Paula, com as seções usadas; Pavía-Miralles; Gelman et al., seção 3.2, para o choque t) |

Dados e fontes: [../dados/consolidacao.md](../dados/consolidacao.md),
[../dados/dicionario.md](../dados/dicionario.md),
[../dados/ideologia-partidos.md](../dados/ideologia-partidos.md).

## Notebooks para docs

| Notebook | Partes | Doc |
|---|---|---|
| `01-AED` | 1 a 4 | [02-aed.md](02-aed.md) |
| `02-MLG` | 5 a 9 | [03-modelos-candidatos.md](03-modelos-candidatos.md) |
| `02-MLG` | 10 | [04-diagnostico.md](04-diagnostico.md) |
| `02-MLG` | 11 | [05-validacao-cruzada.md](05-validacao-cruzada.md) |
| `02-MLG` | 12 (removida) | [06-backtest.md](06-backtest.md) |
| `02-MLG` | 13 | [07-dispersao-regional.md](07-dispersao-regional.md) |
| `02-MLG` | 14 e 15 | [08-dependencia-espacial.md](08-dependencia-espacial.md) |
| `02-MLG` | 16 | [09-modelo-final.md](09-modelo-final.md) |
| `02-MLG` | 17 | [10-extensoes-rejeitadas.md](10-extensoes-rejeitadas.md) |
| `03-apuracao-parcial` | 18 | [11-apuracao-parcial.md](11-apuracao-parcial.md) |
| `03-apuracao-parcial` | 18 (extensões rejeitadas) | [13-apuracao-extensoes-rejeitadas.md](13-apuracao-extensoes-rejeitadas.md) |
| `03-apuracao-parcial` | 18 (diagnóstico D1 a D4) | [14-apuracao-diagnostico-1t.md](14-apuracao-diagnostico-1t.md) |
