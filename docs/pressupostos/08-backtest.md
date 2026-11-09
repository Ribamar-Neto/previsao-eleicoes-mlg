# Pressupostos 08 — Backtest 2022 → 2026

Notebook: `notebooks/04-backtest.ipynb` (Parte 13). Código: `src/backtest.py`. Testes: `tests/test_backtest.py`.
Células contadas a partir de 0. Erros em pontos percentuais (p.p.), previsto menos real.

## 1. Objetivo e regra de negócio

Medir o erro que o modelo teria cometido em 2026 se tivesse sido ajustado em 2022 sem nenhuma informação posterior a 2 out 2022. O resultado real de 2026 (Lula 51,78% dos válidos) só entra na avaliação.

Regra central: o intercepto e os efeitos de região absorvem o swing nacional do período de ajuste (2018→2022, cerca de +6 p.p. para o PT). Aplicados a 2026, repetem esse swing. Por isso a análise separa o erro de nível nacional do erro geográfico.

## 2. Dados e mapas (células 1 a 3)

- Função: `MAPA_FIT_2022` e `MAPA_PREV_2026` (mesmos papéis de `MAPA_2026`, colunas diferentes).
- Ajuste: 2º turno de 2022 ~ `logit_pt_2t_2018`, voto em deputado 2018, prefeito 2020, comparecimento do 2º turno de 2018, perfil do eleitorado 2022, Censo 2022, `cobertura_pbf_2022`, PIB 2021, VAB agro 2021.
- Previsão: tudo deslocado 4 anos (lag 2022, deputados 2022, prefeitos 2024, perfil 2026, `cobertura_pbf_2026`, PIB 2023).
- Decisão: comparecimento do próprio turno e `delta_comp` ficam fora, pois não são conhecidos antes da eleição. O comparecimento defasado os substitui.
- Limitação aceita: Censo 2022 e as colunas de religião, cor e agropecuária têm o mesmo valor nos dois mapas. O Censo só foi publicado em 2023, então o ajuste de 2022 "sabe" um pouco mais do que um analista de 2022 saberia.
- Coluna ausente: não há perfil do eleitorado de 2018, por isso o ajuste usa o de 2022 (publicado antes do 2º turno).
- Boa Esperança do Norte (MT) não existe no parquet nem nos CSV reais. Os totais reais recomputados das 5.570 linhas coincidem com os CSV (célula 20).
- Conclusão: as covariáveis são razoáveis para um backtest, com o viés de informação do Censo declarado.

## 3. Ajuste e previsão (células 4 e 5)

- Função: `ajustar_e_prever(nome, df, mapa_fit, mapa_prev)`, que chama `Candidato.ajustar` e `Candidato.prever`. Cinco modelos: `quasi_logit`, `quasi_reglag`, `bb_logit`, `bb_reglag`, `glmm_uf`.
- Pressuposto: a relação estimada em 2022 vale em 2026 (estacionariedade). A padronização usa média e desvio do ajuste, de modo que mudanças de nível nas covariáveis aparecem na previsão.
- Resultado: coeficiente do lag 2018→2022 de 0,80 a 0,87 (escala bruta, com covariáveis), contra 0,804 da AED sem covariáveis e 0,962 para 2022→2026. Persistência menor que 1 comprime os municípios para a média, e isso erra em 2026, onde a persistência foi maior.

## 4. Número de votos previstos (célula 1)

- Função: `n_previsto`: eleitorado 2026 × (n 2º turno 2022 / aptos 2022).
- Pressuposto: o comparecimento condicional ao PT+Bolsonaro de 2022 se repete.
- Resultado: o total previsto fica 3% abaixo do real. Trocar `n_prev` por `n_real` altera o erro nacional em cerca de 0,15 p.p. (células 7 e 9), portanto o erro vem do share, não do comparecimento.

## 5. Variantes A e B e referências (células 6 a 13)

- A (bruto): previsão direta.
- B (recalibrado): `recalibrar` desloca o logit por uma constante, de modo que a média ponderada por n̂ seja o share nacional de 2022. Pressuposto: swing nacional nulo, informação disponível antes de 2026.
- Referências: `base_2022` (repete 2022) e `base_2022_swing_real` (oráculo de nível: usa o resultado de 2026, serve como piso do erro geográfico).
- Métricas: `resumo_erros`: erro nacional, média absoluta simples por região e por UF, e erro municipal ponderado por votos.
- Resultado: A erra o nível nacional em +4 a +5 p.p.; B zera o erro de nível, mas a média por UF fica em 3,8 a 4,3 p.p., pior que A em UF (3,3 a 3,7) e muito pior que repetir 2022 (1,65). O padrão regional de 2018→2022 (Sudeste +2,5, Norte −6,5 no melhor modelo) não se sustenta em 2026.
- Conclusão: com o desenho ex-ante sem 1º turno, nenhum modelo supera repetir o share de 2022.

## 6. Variante C: entre turnos (células 14 a 17)

- Função: `mapas_entre_turnos` e `preparar`. O 2º turno de 2022 é explicado pelo 1º turno de 2022 (logit PT/(PT+Bolsonaro), parcela dos demais, comparecimento do 1º turno, deputados do mesmo ciclo). A previsão usa o 1º turno de 2026.
- Pressuposto: a relação 1º → 2º turno é estável entre eleições. Coeficiente do logit do 1º turno: 0,94 a 0,97.
- Resultado: C direto erra o nível em −4,7 p.p. (a relação mudou de nível), mas com o nível recalibrado (C_recal) o erro por UF cai a 0,6 a 0,7 p.p., menos da metade do oráculo de nível sobre 2022 (1,6 p.p.).
- Conclusão: o 1º turno contém a informação geográfica de 2026; o nível nacional precisa de uma âncora externa. A hipótese sobre a causa da mudança de nível (transferência dos demais candidatos) não foi testada.

## 7. Quadro final (células 19 e 20)

Tabelas: `results/tables/backtest_resumo.csv`, `backtest_regiao.csv`, `backtest_uf.csv`, `backtest_coef_lag.csv`. Figuras: `results/figures/backtest_{A_bruto,B_recal,C_entre_turnos,C_recal}.html` (barras por região e UF, um modelo por cor).

Limite: um único par de eleições. O erro de nível nacional é uma observação, não uma distribuição; as diferenças entre famílias de modelos (quasi, beta-binomial, GLMM) são pequenas diante do efeito do swing e da escolha do preditor.

## Referências

- McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models*, 2ª ed. Chapman & Hall.
- Wedderburn, R. W. M. (1974). Quasi-likelihood functions, generalized linear models, and the Gauss-Newton method. *Biometrika*.
- Breslow, N. E.; Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *JASA*.
- Hyndman, R. J.; Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*.
- Gelman, A.; Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
