# CLAUDE.md

Projeto acadêmico: MLGs binomiais da votação por município no 2º turno de 2026 (Lula ×
Flávio), agregados em 5 regiões. Notebooks: `01-AED` (Partes 1 a 4), `02-MLG` (5 a 10),
`03-validacao` (11 e 12), `04-backtest` (13), `05-modelo-final` (14 a 17). Língua do projeto: português (notebooks e docs).

## Convenções

- O caminho do repositório tem espaço: coloque entre aspas.
- Rode com `uv run --no-sync`.
- Todo modelo é ajustado por `Candidato.ajustar` em `src/especificacoes.py`; os notebooks 03
  e 04 reusam as mesmas definições (`CANDIDATOS`).
- Quase-binomial: φ de Pearson próprio, porque `scale='X2'` do statsmodels erra com endog de
  2 colunas.
- A padronização das covariáveis é ajustada só na dobra de treino (sem vazamento na CV).
- Estimativas regionais vêm só de previsões fora da amostra (LOSO); dentro da amostra a
  binomial canônica com dummies de região reproduz os totais.
- Dados baixados ficam em cache em `data/raw/<fonte>/`; cada fonte grava um
  `data/interim/<fonte>.parquet`.

## Documentação da análise

`docs/pressupostos/` registra decisões, pressupostos e resultados de cada etapa.

- **Atualizar a cada commit.** Todo commit que muda a análise (notebook, modelo, testes,
  figuras ou tabelas em `results/`) atualiza no mesmo commit o arquivo correspondente e o
  índice `docs/pressupostos/README.md`.
- **Limite de tamanho.** Máximo ~250 linhas ou ~20 KB por arquivo; passou, divida por
  seção e atualize o índice.
- **Nova parte do notebook = novo arquivo** `NN-assunto.md` com o próximo número livre.
  Pendências em `07-pendencias.md`, referências em `referencias.md`.
- **Números de seção estáveis.** Não renumere; cite outra seção por link.
- **Só regras de negócio.** Documente regras, decisões e razões, nunca o que o código faz.
- **Leitura econômica.** Leia o índice e depois só o arquivo necessário.
