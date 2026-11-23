# Pressupostos da análise

Índice. Leia este arquivo primeiro e depois só o arquivo da seção necessária.

| Arquivo | Conteúdo |
|---|---|
| [01-visao-geral.md](01-visao-geral.md) | Problema, decisões de projeto e razões, organização dos notebooks (funções em `src/`, cada valor calculado uma vez), regra de seleção, resultados finais |
| [02-aed.md](02-aed.md) | Análise exploratória dos dados, incluindo a leitura nível a nível das categóricas e binárias (seção 3.1) |
| [03-modelos-candidatos.md](03-modelos-candidatos.md) | Famílias, ligações, regra de fusão de níveis (seção 8.1; Nordeste+Norte) e seleção de variáveis (`bb_selecao`) |
| [04-diagnostico.md](04-diagnostico.md) | Resíduos, influência, envelope e autocorrelação espacial; pressupostos dos modelos escolhidos (`bb_selecao` e `bb_slx`: ligação, linearidade, variância, colinearidade, influência) |
| [05-validacao-cruzada.md](05-validacao-cruzada.md) | Validação cruzada (5 dobras e LOSO sem vazamento), regra de 1 EP, seleção aninhada do `bb_selecao`, limite L e sensibilidade regional |
| [06-backtest.md](06-backtest.md) | Ajuste em 2018 e previsão de 2026 a partir de 2022 |
| [07-dispersao-regional.md](07-dispersao-regional.md) | Dispersão do beta-binomial por região e porte (`bb_disp`), com Wald robusto |
| [08-dependencia-espacial.md](08-dependencia-espacial.md) | Filtro de autovetores (rejeitado), EPs de Conley, SLX (aceito) e corte de 2022 |
| [09-modelo-final.md](09-modelo-final.md) | Regra de informação (covariáveis pré-turno, n̂), estimativas e cobertura do `bb_slx` |
| [10-extensoes-rejeitadas.md](10-extensoes-rejeitadas.md) | Krigagem e CAR/BYM nos resíduos (rejeitadas, também com peso por n), decomposição do I de Moran |
| [11-apuracao-parcial.md](11-apuracao-parcial.md) | Apuração parcial na ordem real do TSE (2022 e 2026), incerteza de n_u e cobertura por unidade |
| [12-pendencias.md](12-pendencias.md) | Pendências abertas, com efeito e correção possível |
| [referencias.md](referencias.md) | Referências únicas e conferidas (Paula, com as seções usadas) |

Dados e fontes: [../dados/consolidacao.md](../dados/consolidacao.md),
[../dados/dicionario.md](../dados/dicionario.md),
[../dados/ideologia-partidos.md](../dados/ideologia-partidos.md).

## Notebooks para docs

| Notebook | Partes | Doc |
|---|---|---|
| `01-AED` | 1 a 4 | [02-aed.md](02-aed.md) |
| `02-MLG` | 5 a 9 | [03-modelos-candidatos.md](03-modelos-candidatos.md) |
| `02-MLG` | 10 | [04-diagnostico.md](04-diagnostico.md) |
| `03-validacao` | 11 | [05-validacao-cruzada.md](05-validacao-cruzada.md) |
| `04-backtest` | 12 | [06-backtest.md](06-backtest.md) |
| `05-modelo-final` | 13 | [07-dispersao-regional.md](07-dispersao-regional.md) |
| `05-modelo-final` | 14 e 15 | [08-dependencia-espacial.md](08-dependencia-espacial.md) |
| `05-modelo-final` | 16 | [09-modelo-final.md](09-modelo-final.md) |
| `05-modelo-final` | 17 | [10-extensoes-rejeitadas.md](10-extensoes-rejeitadas.md) |
| `06-apuracao-parcial` | 18 | [11-apuracao-parcial.md](11-apuracao-parcial.md) |
