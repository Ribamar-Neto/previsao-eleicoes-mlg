# Modelo final: estimativas regionais e por UF (Parte 17 do notebook 05)

Células citadas contam a partir de 0 em `notebooks/05-modelo-final.ipynb` (22 a 25). Tabelas em `results/tables/final_estimativas_regionais.csv`, `final_estimativas_uf.csv`, `final_cobertura.csv` e `final_comparacao_regional.csv`; figura `results/figures/final_regional.html`. Decisões anteriores: dispersão em [09](09-dispersao-regional.md), filtro espacial e Conley em [10](10-dependencia-espacial.md).

**Modelo final = `bb_slx`** (desde 2026-11-09): `bb_disp` (os 12 termos de média de `bb_backward` com dispersão por região e log n, [doc 09](09-dispersao-regional.md)) mais SLX, isto é, W·x das 12 variáveis dos vizinhos (37 parâmetros; MAE por UF no LOSO 1,0224 p.p., limite 1,1308). O filtro espacial foi rejeitado e o SLX aceito ([doc 10](10-dependencia-espacial.md)). Os EPs usados são os de Conley a 200 km, corte conferido com 2022. O SLX quase não reduz o I de Moran dos resíduos (0,339 para 0,333).

## 1. Receita (células 22 a 25)

Espelha a do [notebook 03](06-estimativas-regionais.md): para cada UF deixada de fora, ajusta-se nas 26 restantes; o bloco de β da covariância de Conley (200 km) é calculado nesse ajuste de treino; θᵢ é previsto por município; B = 1000 sorteios, semente 7. Pressupostos (δ independente entre UFs, sorteios de β independentes entre as reestimações, σ constante) são os do doc 06.

- **σ_UF = 0,052** (RMS dos resíduos LOSO das UFs no logit), contra 0,0536 de `bb_backward` ([doc 06](06-estimativas-regionais.md)): o choque de UF praticamente não mudou.
- **Ressalva.** σ_UF é estimado nas mesmas UFs em que a cobertura é conferida; a cobertura com σ_UF é **otimista**.

## 2. Estimativas regionais

Lula %, intervalo de 95% (Brasil: 90%). A coluna `bb_disp` repete o modelo final anterior (HEAD, antes do SLX); só o erro é mostrado:

| Região | Real | `bb_backward` | Final (`bb_slx`) | erro final (p.p.) | erro `bb_disp` (p.p.) |
|---|---|---|---|---|---|
| Centro-Oeste | 39,99 | 40,32 [38,61; 41,97] | 40,04 [38,64; 41,51] | 0,05 | 0,12 |
| Nordeste | 69,37 | 69,42 [68,50; 70,27] | 69,41 [68,54; 70,22] | 0,04 | 0,03 |
| Norte | 50,01 | 49,50 [48,10; 51,03] | 49,52 [48,08; 50,90] | -0,48 | -0,48 |
| Sudeste | 47,24 | 47,71 [46,11; 49,47] | 47,86 [46,30; 49,40] | 0,62 | 0,34 |
| Sul | 37,42 | 37,00 [35,51; 38,46] | 37,01 [35,77; 38,51] | -0,41 | -0,40 |
| Brasil | 51,78 | 51,91 [51,14; 52,63] | 51,95 [51,30; 52,58] (90%) | 0,17 | 0,06 |

- **Frente ao `bb_disp`:** o SLX mexe pouco (Brasil 51,84 para 51,95; erro 0,06 para 0,17 p.p.; Sudeste 0,34 para 0,62 p.p.). Erro regional absoluto médio: 0,357 p.p. (`bb_backward`), 0,321 (final). São só 5 regiões, e o [doc 06](06-estimativas-regionais.md), seção 5, mostra que o erro regional LOSO não escolhe modelos; é descrição, não evidência de superioridade. O erro nacional e o do Sudeste pioram um pouco: coerente com o MAE por UF um pouco pior do SLX ([doc 10](10-dependencia-espacial.md), seção 5).
- Os intervalos finais são um pouco mais estreitos que os de `bb_backward` (por exemplo, Centro-Oeste 2,87 contra 3,36 p.p.), apesar da covariância de β maior; provavelmente porque a dispersão por região ajusta melhor θ [deduced].

## 3. Cobertura

| Covariância + σ | UF 95% | UF 90% | Região 95% | Região 90% |
|---|---|---|---|---|
| Conley + σ_UF | 100% (27/27) | 96,3% | 5/5 | 5/5 |
| Conley + σ = 0 | 59,3% (16/27) | 48,1% | 3/5 | 60% |
| Modelo + σ_UF | 100% (27/27) | 96,3% | 5/5 | 5/5 |
| Modelo + σ = 0 | 44,4% (12/27) | 29,6% | 2/5 | 40% |

- **O choque de UF é o que sustenta a cobertura.** Com σ = 0 a cobertura por UF cai a 59% (Conley) ou 44% (modelo): ruído binomial e de θ sozinhos não cobrem o erro de extrapolar para uma UF nova.
- **Conley ajuda, mas pouco.** A largura média do intervalo regional com σ = 0 sobe só de 0,68 para 0,88 p.p. (modelo contra Conley): o ruído binomial e o de θ dominam.
- **GO fica fora a 90% nas duas variantes com σ_UF.** O real (40,34) fica abaixo do limite inferior do intervalo de 90%, tanto com Conley (40,47) quanto com o modelo (40,68); a 95% as duas cobrem. Com `bb_disp`, só Conley errava GO, na borda.
- **Otimismo.** A cobertura com σ_UF vale como descrição, não como validação: σ_UF entra ajustado nas mesmas 27 UFs. Pendência: σ_UF por validação aninhada ([07](07-pendencias.md)).

## 4. Decisão

Os resultados do projeto vêm do modelo final (`bb_slx` com EPs de Conley). `bb_backward` segue como o modelo escolhido pela regra de seleção do [doc 01](01-visao-geral.md); o modelo final é o refinamento dele (dispersão por região e log n, mais SLX das 12 variáveis).

## Referências

- Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
