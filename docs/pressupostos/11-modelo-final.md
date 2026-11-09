# Modelo final: estimativas regionais e por UF (Parte 17 do notebook 05)

Células citadas contam a partir de 0 em `notebooks/05-modelo-final.ipynb` (15 a 18). Tabelas em `results/tables/final_estimativas_regionais.csv`, `final_estimativas_uf.csv`, `final_cobertura.csv` e `final_comparacao_regional.csv`; figura `results/figures/final_regional.html`. Decisões anteriores: dispersão em [09](09-dispersao-regional.md), filtro espacial e Conley em [10](10-dependencia-espacial.md).

**Modelo final = `bb_disp`**: os 12 termos de média de `bb_backward` com dispersão que varia por região e log n (aceita no doc 09). O filtro espacial foi rejeitado (doc 10). Os EPs usados são os de Conley a 200 km.

## 1. Receita (células 15 a 17)

Espelha a do [notebook 03](06-estimativas-regionais.md): para cada UF deixada de fora, ajusta-se nas 26 restantes; o bloco de β da covariância de Conley (200 km) é calculado nesse ajuste de treino; θᵢ é previsto por município; B = 1000 sorteios, semente 7. Pressupostos (δ independente entre UFs, sorteios de β independentes entre as reestimações, σ constante) são os do doc 06.

- **σ_UF = 0,0535** (RMS dos resíduos LOSO das UFs no logit), contra 0,0536 de `bb_backward` ([doc 06](06-estimativas-regionais.md)): o choque de UF praticamente não mudou.
- **Ressalva.** σ_UF é estimado nas mesmas UFs em que a cobertura é conferida; a cobertura com σ_UF é **otimista**.

## 2. Estimativas regionais

Lula %, intervalo de 95% (Brasil: 90%):

| Região | Real | `bb_backward` | Final |
|---|---|---|---|
| Centro-Oeste | 39,99 | 40,32 [38,61; 41,97] | 40,12 [38,68; 41,64] |
| Nordeste | 69,37 | 69,42 [68,50; 70,27] | 69,40 [68,48; 70,22] |
| Norte | 50,01 | 49,50 [48,10; 51,03] | 49,53 [48,13; 50,89] |
| Sudeste | 47,24 | 47,71 [46,11; 49,47] | 47,58 [45,99; 49,24] |
| Sul | 37,42 | 37,00 [35,51; 38,46] | 37,02 [35,53; 38,44] |
| Brasil | 51,78 | 51,91 [51,14; 52,63] | 51,84 [51,21; 52,49] (90%) |

- **Erro absoluto médio regional:** 0,357 p.p. (`bb_backward`) contra 0,276 (final). São só 5 regiões, e o [doc 06](06-estimativas-regionais.md), seção 5, mostra que o erro regional LOSO não escolhe modelos; é descrição, não evidência de superioridade.
- Os intervalos finais são um pouco mais estreitos que os de `bb_backward` (por exemplo, Centro-Oeste 2,96 contra 3,36 p.p.), apesar da covariância de β maior; provavelmente porque a dispersão por região ajusta melhor θ [deduced].

## 3. Cobertura

| Covariância + σ | UF 95% | UF 90% | Região 95% | Região 90% |
|---|---|---|---|---|
| Conley + σ_UF | 100% (27/27) | 96,3% | 5/5 | 5/5 |
| Conley + σ = 0 | 59,3% (16/27) | 44,4% | 4/5 | 60% |
| Modelo + σ_UF | 100% (27/27) | 100% | 5/5 | 5/5 |
| Modelo + σ = 0 | 44,4% (12/27) | 29,6% | 3/5 | 40% |

- **O choque de UF é o que sustenta a cobertura.** Com σ = 0 a cobertura por UF cai a 59% (Conley) ou 44% (modelo): ruído binomial e de θ sozinhos não cobrem o erro de extrapolar para uma UF nova.
- **Conley ajuda, mas pouco.** A variância de β é cerca de 5 vezes maior, mas a largura média do intervalo regional com σ = 0 sobe só de 0,68 para 0,81 p.p.: o ruído binomial e o de θ dominam.
- **GO no nível de 90%.** Conley + σ_UF erra GO e o modelo + σ_UF cobre: o real (40,34) fica na borda inferior dos dois intervalos (40,40 contra 40,25). Com B = 1000, as diferenças de largura por UF entre as duas variantes têm DP de 0,155 p.p.; é ruído de Monte Carlo na borda [deduced], não diferença de método.
- **Otimismo.** A cobertura com σ_UF vale como descrição, não como validação: σ_UF entra ajustado nas mesmas 27 UFs. Pendência: σ_UF por validação aninhada ([07](07-pendencias.md)).

## 4. Decisão

Os resultados do projeto passam a vir do modelo final (`bb_disp` com EPs de Conley). `bb_backward` segue como o modelo escolhido pela regra de seleção do [doc 01](01-visao-geral.md); o modelo final é o refinamento dele (dispersão), com a mesma média.

## Referências

- Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
