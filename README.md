# Previsão das eleições 2026 por MLG

Trabalho acadêmico: ajusta MLGs da família binomial à votação por município no segundo
turno presidencial de 2026 (Lula × Flávio Bolsonaro) e agrega as previsões às 5 regiões.
Notebooks e documentação em português.

## Objetivo

Estimar a participação regional de cada candidato com um modelo escolhido por regra fixada
antes dos resultados ([docs/pressupostos/01-visao-geral.md](docs/pressupostos/01-visao-geral.md)).

## Fontes de dados

- TSE: votação presidencial 2018, 2022 e 2026 por município; perfil do eleitorado 2022 e
  2026; votos em deputado federal e partido do prefeito (ideologia).
- IBGE: Censo 2022, PIB dos Municípios, tabela de equivalência IBGE × TSE e malha municipal.
- Portal da Transparência (CGU): famílias do Auxílio Brasil (set/2022) e Bolsa Família
  (set/2026), pois a API do MDS está indisponível.
- Classificação ideológica dos partidos: Bolognesi, Ribeiro e Codato (2023).

Regras de merge e limitações: [docs/dados/consolidacao.md](docs/dados/consolidacao.md).

## Fórmula e unidades

- Unidade: município (5.570; censo, não amostra). Yᵢ = votos de Lula; nᵢ = votos de Lula +
  Flávio (só Brasil; exterior `ZZ` fora). Yᵢ ~ Bin(nᵢ, πᵢ), g(πᵢ) = xᵢᵀβ; π_Flávio = 1 − π_Lula.
- Participação regional: p̂_r = Σ π̂ᵢ nᵢ / Σ nᵢ (soma sobre os municípios da região).
- Participações são frações nos dados e p.p. nos relatórios. O modelo descreve municípios,
  não eleitores (falácia ecológica).

## Exemplo

Dois municípios na mesma região: n = 10.000 com π̂ = 0,60 e n = 30.000 com π̂ = 0,40.
p̂ = (6.000 + 12.000) / 40.000 = 45,0% Lula e 55,0% Flávio.

## Casos de borda

- Boa Esperança do Norte (MT) não está nas 5.570 linhas; seus votos são reportados à parte.
- Voto no exterior (UF `ZZ`) é descartado.
- Brancos e nulos ficam fora de nᵢ.
- Em ligação canônica com dummies de região, o total regional dentro da amostra é exato;
  por isso as estimativas regionais vêm de previsões fora da amostra (LOSO).

## Reprodução

```bash
uv sync
# Coleta: o crosswalk TSE↔IBGE vem primeiro, porque Bolsa Família e a consolidação dependem dele.
uv run --no-sync python -m src.coleta.ibge_municipios
uv run --no-sync python -m src.coleta.tse_presidente
uv run --no-sync python -m src.coleta.tse_ideologia
uv run --no-sync python -m src.coleta.tse_perfil
uv run --no-sync python -m src.coleta.ibge_censo_pib
uv run --no-sync python -m src.coleta.mds_bolsa_familia
uv run --no-sync python -m src.coleta.consolidar   # grava data/processed/municipios.parquet
uv run --no-sync pytest
uv run --no-sync jupyter lab notebooks/
```

Rode os notebooks na ordem 01, 02, 03, 04, 05.

## Resultados principais

- Superdispersão forte (φ de Pearson 21,7); a beta-binomial a captura (resíduos com DP 1,03,
  contra 3,43 na binomial).
- A regra de seleção escolhe `bb_backward` (20 parâmetros, erro por UF 1,000 p.p. no LOSO);
  12 modelos ficam dentro de 1 EP do melhor.
  O notebook 05 refina o escolhido (dispersão por região); DP residual regional:
  máximo |DP − 1| de 0,409 para 0,095.
- Modelo final (`bb_disp`: `bb_backward` com dispersão por região e log n; EPs de Conley a
  200 km; filtro espacial rejeitado): Lula, Brasil, estimado 51,84% [51,1; 52,6] contra
  51,78% real. Por região (estimado/real): Centro-Oeste 40,12/39,99; Nordeste 69,40/69,37;
  Norte 49,53/50,01; Sudeste 47,58/47,24; Sul 37,02/37,42
  ([docs/pressupostos/11-modelo-final.md](docs/pressupostos/11-modelo-final.md)).
- Backtest: com o 1º turno de 2026 e recalibração, erro de 0,58 p.p. por UF e 0,68 por região.
- Pendências: [docs/pressupostos/07-pendencias.md](docs/pressupostos/07-pendencias.md).

## Estrutura

```
src/        coleta (src/coleta/), especificações dos modelos, modelos próprios, backtest
tests/      testes unitários
notebooks/  01-AED, 02-MLG, 03-validacao, 04-backtest, 05-modelo-final
data/       raw (cache por fonte), interim (um parquet por fonte), processed, referencia
results/    cv, figures, tables
docs/       dados (fontes) e pressupostos (decisões e resultados)
```
