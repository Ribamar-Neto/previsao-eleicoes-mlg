# Pressupostos da análise

Índice. Leia este arquivo primeiro e depois só o arquivo da seção necessária.

| Arquivo | Conteúdo |
|---|---|
| [01-visao-geral.md](01-visao-geral.md) | Problema, decisões de projeto e razões, regra de seleção, resultados finais |
| [02-aed.md](02-aed.md) | Análise exploratória dos dados |
| [03-modelos-candidatos.md](03-modelos-candidatos.md) | Famílias, ligações e especificações candidatas |
| [04-diagnostico.md](04-diagnostico.md) | Resíduos, influência, envelope e autocorrelação espacial |
| [05-validacao-cruzada.md](05-validacao-cruzada.md) | Validação cruzada (5 dobras e leave-one-state-out) e regra de seleção |
| [06-estimativas-regionais.md](06-estimativas-regionais.md) | Estimativas por região e intervalos |
| [07-pendencias.md](07-pendencias.md) | Pendências abertas, com efeito e correção possível |
| [08-backtest.md](08-backtest.md) | Ajuste em 2018 e previsão de 2026 a partir de 2022 |
| [09-dispersao-regional.md](09-dispersao-regional.md) | Dispersão do beta-binomial por região e porte (modelo `bb_disp`) |
| [10-dependencia-espacial.md](10-dependencia-espacial.md) | Filtro de autovetores (rejeitado) e EPs de Conley |
| [11-modelo-final.md](11-modelo-final.md) | Estimativas regionais e por UF do modelo final, cobertura |
| [referencias.md](referencias.md) | Referências únicas e conferidas (36, todas verificadas) |

Dados e fontes: [../dados/consolidacao.md](../dados/consolidacao.md),
[../dados/dicionario.md](../dados/dicionario.md),
[../dados/ideologia-partidos.md](../dados/ideologia-partidos.md).

## Notebooks para docs

| Notebook | Partes | Doc |
|---|---|---|
| `01-AED` | 1 a 4 | [02-aed.md](02-aed.md) |
| `02-MLG` | 5 a 9 (modelos) | [03-modelos-candidatos.md](03-modelos-candidatos.md) |
| `02-MLG` | 10 (diagnósticos) | [04-diagnostico.md](04-diagnostico.md) |
| `03-validacao` | 11 | [05-validacao-cruzada.md](05-validacao-cruzada.md) |
| `03-validacao` | 12 | [06-estimativas-regionais.md](06-estimativas-regionais.md) |
| `04-backtest` | 13 | [08-backtest.md](08-backtest.md) |
| `05-modelo-final` | 14 | [09-dispersao-regional.md](09-dispersao-regional.md) |
| `05-modelo-final` | 15 e 16 | [10-dependencia-espacial.md](10-dependencia-espacial.md) |
| `05-modelo-final` | 17 | [11-modelo-final.md](11-modelo-final.md) |
