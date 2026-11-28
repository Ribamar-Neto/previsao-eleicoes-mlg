# Critérios de verificação

Uma seção por categoria de tarefa. Só regras de negócio.

## Comandos

- `make smoke` roda `make lint` e `make test`.
- `make lint` confere só os `.py` alterados desde `BASE` (padrão `HEAD`).
- `make lint-all` confere a árvore toda. Achados em arquivos que a tarefa não tocou não bloqueiam.
- O usuário roda os notebooks com `!`. O Claude não os roda.

## Refatoração sem mudança de resultado

- **Sucesso**: o Run-All do notebook afetado regrava `results/` byte a byte igual ao de antes da mudança.
- **Conferência**: grave o SHA-256 de cada arquivo de `results/` antes da mudança. Depois do Run-All, compare.
- **Casos de borda**: um arquivo novo ou apagado em `results/` também conta como diferença.

## Mudança de análise

Vale para mudança no modelo, num notebook ou nas tabelas e figuras de `results/`.

- **Sucesso**: cada número citado em `docs/pressupostos/` vem de um arquivo de `results/` do mesmo commit.
- **Documentação**: o mesmo commit atualiza o arquivo da parte e o índice `docs/pressupostos/README.md`.
- **Regras a conferir**: as de `CLAUDE.md`, seção Convenções. Exemplos: φ de Pearson próprio, padronização só na dobra de treino, totais regionais só da LOSO.
- **Casos de borda**: um número sem arquivo de origem em `results/` declara a origem no próprio doc.
