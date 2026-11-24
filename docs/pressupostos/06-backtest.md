# 06 - Backtest 2022 → 2026 (removido)

**Parte 12 removida em 2026-11-23.** O arquivo fica só para os links não quebrarem. O número da Parte 12 não foi reaproveitado.

## Razão da remoção

- **Sem consumidor.** Nenhuma decisão de seleção ou de modelo final dependia do backtest: a escolha vem da validação cruzada (Parte 11, [05](05-validacao-cruzada.md)) e as estimativas, do modelo final ([09](09-modelo-final.md)).
- **Números defasados.** O backtest não acompanhou as mudanças de regra de informação (covariáveis pré-turno, n̂) e de seleção aninhada; seus resultados já não correspondiam ao modelo final.
- **Previsão fora do tempo já testada.** A apuração parcial (Parte 18, [11](11-apuracao-parcial.md)) testa a previsão em dados que o modelo não viu, nas eleições de 2022 e 2026, na ordem real de apuração.

## O que permanece

`src/backtest.py` (módulo `BT`) e `tests/test_backtest.py` continuam: a validação cruzada, o modelo final e a apuração parcial importam `BT`. As tabelas `results/tables/backtest_*.csv` e as figuras do backtest foram apagadas.
