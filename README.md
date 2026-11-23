# Previsão das Eleições 2026 por MLG — 2º turno presidencial (Lula × Flávio Bolsonaro)

## 📌 Sobre o projeto

Este repositório contém o desenvolvimento de um trabalho acadêmico sobre **Modelos Lineares Generalizados (MLG) da família binomial**, aplicados à votação por município no **segundo turno presidencial de 2026** (Lula × Flávio Bolsonaro), com agregação das previsões às **5 regiões** do Brasil.

O objetivo é comparar especificações de MLG com um critério de escolha fixado **antes** dos resultados e estimar a participação regional de cada candidato com o modelo escolhido.

> **Fontes principais:** TSE, IBGE, Portal da Transparência (CGU) e Bolognesi, Ribeiro e Codato (2023). Detalhes na seção [Sobre o Dataset](#-sobre-o-dataset).

A base final tem **5.570 municípios** (um censo, não uma amostra) e 160 colunas. Notebooks e documentação em português.

> **Ressalva (falácia ecológica).** O modelo descreve **municípios**, não eleitores: os coeficientes são associações entre municípios e não permitem inferir o comportamento individual (Robinson, 1950).

---

## 🎯 Objetivos

### Objetivo geral

Estimar a participação regional de Lula e de Flávio Bolsonaro no 2º turno de 2026 com um MLG binomial escolhido por regra fixada antes dos resultados ([`docs/pressupostos/01-visao-geral.md`](docs/pressupostos/01-visao-geral.md)).

### Objetivos específicos

* Coletar e consolidar as fontes de dados (TSE, IBGE, CGU) em uma base por município;
* Realizar uma análise exploratória da votação e das covariáveis;
* Comparar famílias e ligações (binomial, quase-binomial, beta-binomial, GLMM, regressão beta);
* Diagnosticar superdispersão, resíduos, influência e autocorrelação espacial;
* Escolher o modelo por validação cruzada leave-one-state-out (LOSO) e regra do 1 erro-padrão;
* Estimar as participações regionais apenas com previsões fora da amostra, com intervalos de incerteza;
* Testar a capacidade de previsão em um backtest (ajuste em 2018, previsão de 2026 a partir de 2022);
* Modelar a dispersão por região e a dependência espacial (SLX, filtro de autovetores, EPs de Conley);
* Comparar as estimativas finais com o resultado real.

---

## 📊 Sobre o Dataset

Fontes (regras de merge e limitações em [`docs/dados/consolidacao.md`](docs/dados/consolidacao.md)):

* **TSE:** votação presidencial 2018, 2022 e 2026 por município; perfil do eleitorado 2022 e 2026; votos em deputado federal e partido do prefeito (ideologia).
* **IBGE:** Censo 2022, PIB dos Municípios, tabela de equivalência IBGE × TSE e malha municipal.
* **Portal da Transparência (CGU):** famílias do Auxílio Brasil (set/2022) e Bolsa Família (set/2026), porque a API do MDS está indisponível.
* **Ideologia dos partidos:** classificação de Bolognesi, Ribeiro e Codato (2023).


| Variável | Papel | Unidade | Mín | Mediana | Máx | Caxingó (PI) | Fonte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `y` (votos de Lula no 2º turno de 2026) | Resposta | contagem | 197 | 3.851 | 3.864.354 | 3.800 | TSE |
| `n` (votos em Lula + Flávio, só Brasil) | Total (Lula + Flávio) | contagem | 848 | 7.201 | 6.962.277 | 4.628 | TSE |
| `lag`: Logit da participação de Lula no 2º turno de 2022 | Defasagem | logit | -2,090 | 0,145 | 2,727 | 1,757 | TSE |
| `pct_dep_esq`: Votos de deputado federal em partidos de esquerda | Ideologia | fração | 0,005 | 0,111 | 0,802 | 0,453 | TSE + Bolognesi et al. (2023) |
| `comparecimento`: Comparecimento no 1º turno de 2026 | Mobilização | fração | 0,599 | 0,794 | 0,927 | 0,862 | TSE |
| `log_dens`: Log da densidade demográfica | Território | log(hab/km²) | -1,872 | 3,191 | 9,541 | 2,416 | IBGE (Censo 2022) |
| `pct_mulheres`: Mulheres no eleitorado | Demografia | fração | 0,434 | 0,509 | 0,557 | 0,490 | TSE (perfil) |
| `pct_16_24`: Eleitorado de 16 a 24 anos | Demografia | fração | 0,058 | 0,123 | 0,304 | 0,173 | TSE (perfil) |
| `pct_superior`: Eleitorado com ensino superior completo | Escolaridade | fração | 0,015 | 0,066 | 0,366 | 0,035 | TSE (perfil) |
| `pct_analfabeto`: Eleitorado analfabeto | Escolaridade | fração | 0,001 | 0,046 | 0,276 | 0,208 | TSE (perfil) |
| `cobertura_pbf`: Famílias do Bolsa Família sobre a população (pode passar de 1) | Renda e proteção social | razão | 0,003 | 0,267 | 1,464 | 0,731 | CGU + IBGE |
| `pct_vab_agro`: Valor adicionado bruto da agropecuária no total | Economia | fração | -0,419 | 0,190 | 0,935 | 0,127 | IBGE (PIB dos Municípios) |
| `pct_evangelicos`: Evangélicos na população | Religião | fração | 0,013 | 0,217 | 0,887 | 0,082 | IBGE (Censo 2022) |
| `pct_pretos_pardos`: Pretos e pardos na população | Raça/cor | fração | 0,025 | 0,608 | 0,972 | 0,808 | IBGE (Censo 2022) |
| `bloco_pref`: ideologia do partido do prefeito eleito em 2024 | Ideologia local | categórica | direita (4770), esquerda (735), centro (49), sem prefeito (16) | | | direita | TSE + Bolognesi et al. (2023) |
| `regiao`: região do IBGE | Região | categórica | Nordeste (1794), Sudeste (1668), Sul (1191), Centro-Oeste (467), Norte (450) | | | Nordeste | IBGE |


Todos os percentuais são frações em [0, 1]; `pct_vab_agro` e `cobertura_pbf` fogem da faixa (valores da fonte). Nenhuma covariável do 2º turno de 2026 entra no modelo (ela é desconhecida enquanto a apuração não chega); o comparecimento é o do 1º turno. As demais covariáveis candidatas (`nota_dep`, `delta_comp`, `log_pop`, `pct_60_mais`, `pib_pc`, `pct_urbana`) entram nos modelos candidatos e saem na seleção por BIC; `bloco_pref` (`bloco_prefeito_2024`) **permanece** no modelo ([`docs/pressupostos/03-modelos-candidatos.md`](docs/pressupostos/03-modelos-candidatos.md)).

---

## 🎯 Variável resposta

A resposta é o número de votos de Lula no município, dado o total de votos nos dois candidatos:

* $Y_i$ = votos de Lula no município $i$;
* $n_i$ = votos de Lula + votos de Flávio (só Brasil; o exterior, UF `ZZ`, fica fora).

$$
Y_i \sim \mathrm{Bin}(n_i, \pi_i), \qquad g(\pi_i) = x_i^\top \beta, \qquad \pi_{\text{Flávio}} = 1 - \pi_{\text{Lula}}
$$

Participação regional (soma sobre os municípios da região $r$):

$$
\hat p_r = \frac{\sum_i \hat\pi_i\, n_i}{\sum_i n_i}
$$

onde:

* $g$ é a função de ligação (logit no modelo final);
* $x_i$ são as covariáveis do município;
* participações são frações nos dados e pontos percentuais (p.p.) nos relatórios.

**Exemplo.** Dois municípios na mesma região: $n = 10.000$ com $\hat\pi = 0{,}60$ e $n = 30.000$ com $\hat\pi = 0{,}40$. Então $\hat p = (6.000 + 12.000)/40.000 = 45{,}0\%$ para Lula e $55{,}0\%$ para Flávio.

**Casos de borda.**

* Boa Esperança do Norte (MT) não está nas 5.570 linhas (instalado depois do Censo 2022); seus votos (PT 945, Bolsonaro 2.353) são reportados à parte e mudam o Centro-Oeste em −0,004 p.p.;
* o voto no exterior (UF `ZZ`) é descartado;
* brancos e nulos ficam fora de $n_i$;
* com ligação canônica e dummies de região, o total regional **dentro da amostra** é exato por construção; por isso as estimativas regionais vêm só de previsões fora da amostra (LOSO).

---

## 🔎 Variáveis explicativas

O modelo final usa a defasagem (`lag`) e 11 variáveis contínuas escolhidas por seleção passo a passo por BIC (`bb_selecao`), mais as mesmas 12 defasadas espacialmente (SLX), o bloco do prefeito e dummies de região (Nordeste e Norte fundidos na média; LRT p = 0,921), num total de 36 parâmetros (30 da média e 6 da dispersão).

```text
logit_pt_2t_2022 (lag)   pct_dep_esq      comparecimento    log_dens
pct_mulheres             pct_16_24        pct_superior      pct_analfabeto
cobertura_pbf            pct_vab_agro     pct_evangelicos   pct_pretos_pardos
W·x de cada uma das 12 (W = vizinhança Queen, padronizada por linha)
bloco_pref (esquerda, outro; referência: direita)   região (Centro-Oeste, Sul, Nordeste+Norte; referência: Sudeste)
```

Seleção: o passo a passo por BIC (backward, forward e both) chega ao mesmo conjunto, com BIC 72.844,87; a busca em todos os subconjuntos não achou esse ótimo (melhor BIC 72.848,78). As covariáveis contínuas são padronizadas (escore z) só na dobra de treino, sem vazamento na validação cruzada. Dispersão: $\log\theta = Z\gamma$, com dummies de região e $\log n$.

---

## 🧪 Metodologia

O desenvolvimento seguiu seis notebooks. Cada parte tem seu documento em [`docs/pressupostos/`](docs/pressupostos/README.md).

### 1. Análise exploratória (`01-AED`, Partes 1 a 4)

* Dados, resposta e covariáveis; mapas coropléticos;
* A resposta de 2026 contra 2022 tem inclinação 0,96 (OLS), o que justifica estimar o coeficiente da defasagem em vez de fixá-lo em 1;
* Colinearidade (VIF) das covariáveis.

### 2. Modelos candidatos e diagnóstico (`02-MLG`, Partes 5 a 10)

* Binomial com cinco ligações, quase-binomial (φ de Pearson próprio), beta-binomial, GLMM com PQL próprio, regressão beta e linhas de base (OLS e WLS sobre o logit empírico);
* Defasagem livre, offset, spline, defasagem por região e efeito de UF;
* Diagnóstico: resíduos quantílicos, envelope meio-normal, alavanca e distância de Cook, I de Moran; nos modelos escolhidos (`bb_selecao` e `bb_slx`), também teste de ligação, linearidade por covariável, VIF/GVIF da matriz final e Cook de um passo;
* Parte 8: categóricas por nível (bloco do prefeito, regiões fundidas quando o teste não as separa) e seleção de variáveis por BIC.

### 3. Validação cruzada e regra de seleção (`03-validacao`, Parte 11)

Regra fixada antes dos resultados:

1. menor erro absoluto médio da participação por UF no LOSO;
2. regra do 1 erro-padrão: entre os modelos dentro de 1 EP do melhor, o de menos parâmetros;
3. o MAE ponderado por votos do 5-fold é critério secundário.

Sem vazamento: a padronização é ajustada só na dobra de treino. O erro regional LOSO **não** é critério, porque favorece por construção as binomiais canônicas. (Uma versão anterior vazava informação da UF de teste; foi corrigida e as seções afetadas foram removidas.)

### 4. Backtest (`04-backtest`, Parte 12)

Ajuste em 2022 (com 2018 como defasagem) e previsão de 2026 a partir de 2022, com e sem os dados do 1º turno de 2026 e com recalibração do intercepto.

### 5. Modelo final (`05-modelo-final`, Partes 13 a 17)

* **Parte 13, dispersão por região e log n** (`bb_disp`), confirmada por teste de Wald robusto;
* **Parte 14, ESF e SLX:** filtro de autovetores (rejeitado) e SLX (aceito);
* **Parte 15, erros-padrão de Conley** a 200 km;
* **Parte 16, estimativas regionais e por UF** do modelo final e cobertura dos intervalos;
* **Parte 17, extensões rejeitadas:** krigagem LOSO dos resíduos e CAR/BYM.

### 6. Apuração parcial (`06-apuracao-parcial`, Parte 18)

Estimativa do resultado com parte das urnas apuradas (cortes de 5% a 90%), comparada à extrapolação bruta e ao swing.

---


## 🛠️ Tecnologias utilizadas

* **Python** (≥ 3.12) e **uv**
* **Jupyter Notebook**
* **Pandas**, **NumPy** e **PyArrow**
* **Plotly** (gráficos e mapas interativos)
* **SciPy**
* **Statsmodels** (GLM, beta-binomial)
* **Scikit-learn**
* **GeoPandas** e **geobr** (malha municipal)
* **PySAL** (`libpysal`, `esda`)
* **pytest**

---

## 📁 Estrutura do projeto

```text
.
├── src/          # coleta (src/coleta/), especificações dos modelos, modelos próprios, backtest
├── tests/        # testes unitários
├── notebooks/    # 01-AED a 06-apuracao-parcial
├── data/         # raw (cache por fonte), interim (um parquet por fonte), processed, referencia
├── results/      # cv, figures, tables
├── docs/         # dados (fontes) e pressupostos (decisões e resultados)
└── README.md
```

### `src/`

Os notebooks não definem funções: toda função auxiliar fica em `src/`, e a célula mostra só o passo estatístico.

```text
src/
├── coleta/          # um módulo por fonte e o consolidador
├── especificacoes.py # candidatos; todo modelo é ajustado por `Candidato.ajustar` (reusado pelos notebooks 03 e 04)
├── modelos.py       # modelos próprios (quase-binomial, beta-binomial, GLMM por PQL)
├── selecao.py       # seleção de variáveis (AIC/BIC, Hosmer-Lemeshow, todos os subconjuntos)
├── pressupostos.py  # diagnóstico dos pressupostos de um modelo já ajustado
├── validacao.py     # validação cruzada (5 dobras e LOSO) e erros por UF e região
├── backtest.py      # backtest 2018 -> 2022 -> 2026
├── espacial.py      # autovetores de Moran, Conley, SLX, krigagem e CAR
├── apuracao.py      # apuração parcial e simulação das participações
├── dados.py         # carga comum dos notebooks 05 e 06
├── graficos.py      # gráficos Plotly
└── saidas.py        # gravação padronizada de tabelas, figuras e JSON
```

### `data/`

Dados baixados ficam em cache em `data/raw/<fonte>/`; cada fonte grava um `data/interim/<fonte>.parquet`; a consolidação grava `data/processed/municipios.parquet` (5.570 linhas).

### `notebooks/`

Cada valor é calculado em um só notebook; os seguintes o leem de `results/` e citam a origem. Rode nesta ordem (o `02` roda duas vezes: o Passo 6 dele lê os termos do `bb_slx` em `results/tables/final_decisao.json`, escrito pelo `05`, e sem o json só `bb_selecao` entra no passo; o `03` lê `loso_bb_selecao.parquet`, escrito pelo `02`; o `05` lê `cv_loso_aninhado.csv` e `sensibilidade_regional.csv`, escritos pelo `03`, e completa `diag_final_moran.csv`, do `02`, com a linha do `bb_slx`):

1. `01-AED.ipynb`: análise exploratória (Partes 1 a 4);
2. `02-MLG.ipynb`: modelos candidatos e diagnóstico (Partes 5 a 10);
3. `03-validacao.ipynb`: validação cruzada (Parte 11);
4. `04-backtest.ipynb`: backtest (Parte 12);
5. `05-modelo-final.ipynb`: dispersão, dependência espacial, Conley, estimativas e extensões rejeitadas (Partes 13 a 17);
6. `06-apuracao-parcial.ipynb`: apuração parcial (Parte 18);
7. `02-MLG.ipynb` de novo, para o Passo 6 (pressupostos de `bb_selecao` e `bb_slx`).

### `results/`

Previsões fora da amostra (`cv/`), gráficos interativos em HTML (`figures/`) e tabelas em CSV (`tables/`).

### `docs/`

* `docs/dados/`: fontes, regras de consolidação e dicionário;
* `docs/pressupostos/`: decisões, pressupostos e resultados de cada etapa; comece pelo [índice](docs/pressupostos/README.md).

### Reprodução

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

O caminho do repositório tem espaço: coloque-o entre aspas.

---

## 📈 Resultados

Os resultados abaixo são do modelo final `bb_slx` (Partes 13 a 17, notebook `05-modelo-final.ipynb`) e da apuração parcial (Parte 18). Detalhes em [`docs/pressupostos/09-modelo-final.md`](docs/pressupostos/09-modelo-final.md). Todos os números vêm de `results/tables/`.

### Modelo ajustado

$$
Y_i \sim \mathrm{BetaBin}(n_i, \pi_i, \theta_i), \quad
\mathrm{logit}(\pi_i) = \beta_0 + \sum_{k=1}^{12} \beta_k z_{ik} + \sum_{k=1}^{12} \delta_k (Wz)_{ik} + \text{bloco}_i + \text{região}_i, \quad
\log \theta_i = Z_i \gamma
$$

* beta-binomial com ligação logit; 36 parâmetros;
* $z$ são as covariáveis padronizadas; $W$ é a vizinhança Queen padronizada por linha, e $(Wz)_{ik}$ é a média da variável $k$ nos vizinhos (SLX);
* $Z_i$ contém dummies de região e $\log n_i$;
* os erros-padrão são os de Conley a 200 km, de 1,0 a 3,5 vez os do modelo (mediana 1,7).

### Coeficientes (escala z)

Cada covariável contínua entra padronizada (média 0, desvio-padrão 1), então $e^\beta$ é a razão de chances do voto em Lula para **+1 desvio-padrão (DP)** da covariável, mantidos os demais termos. A leitura é sobre a **participação municipal** de Lula nos votos válidos, não sobre o comportamento de um eleitor (ver [Limitações](#limitações)).

| Termo | β (z) | EP Conley | e^β (OR por 1 DP) | Signif. 5% | Leitura |
| --- | --- | --- | --- | --- | --- |
| intercepto | 0,1945 | 0,0070 | n/a | sim | logit de Lula do município de referência (Sudeste, prefeito de direita, covariáveis na média): 54,8% |
| defasagem (logit de 2022) | 0,7749 | 0,0051 | 2,170 | sim | o voto de 2022 domina: o município com +1 DP de Lula em 2022 tem chance de Lula 2,17x maior |
| % de esquerda na Câmara | 0,0097 | 0,0020 | 1,010 | sim | municípios mais à esquerda no voto para deputado têm Lula mais forte |
| comparecimento no 1º turno | −0,0175 | 0,0022 | 0,983 | sim | comparecimento maior no 1º turno se associa a Lula um pouco menor |
| log da densidade | 0,0149 | 0,0035 | 1,015 | sim | municípios mais densos têm Lula mais forte |
| % de mulheres | 0,0294 | 0,0024 | 1,030 | sim | eleitorado mais feminino se associa a Lula mais forte |
| % de 16 a 24 anos | −0,0099 | 0,0034 | 0,990 | sim | eleitorado mais jovem se associa a Lula menor |
| % com ensino superior | 0,0092 | 0,0017 | 1,009 | sim | mais escolaridade superior se associa a Lula um pouco maior (mantido o resto) |
| % de analfabetos | −0,0025 | 0,0028 | 0,998 | não | sem associação distinguível de zero |
| cobertura do Bolsa Família | 0,0375 | 0,0046 | 1,038 | sim | maior cobertura do Bolsa Família, o maior efeito entre as não defasadas |
| % do VAB agropecuário | −0,0110 | 0,0023 | 0,989 | sim | economia mais agropecuária se associa a Lula menor |
| % de evangélicos | 0,0080 | 0,0025 | 1,008 | sim | efeito condicional pequeno e positivo, que pode diferir da associação simples entre municípios |
| % de pretos e pardos | 0,0196 | 0,0060 | 1,020 | sim | mais pretos e pardos se associa a Lula maior |
| `W·`defasagem (logit de 2022) | −0,0277 | 0,0082 | 0,973 | sim | vizinhos com Lula mais forte em 2022 reduzem a chance, dado o próprio município (reversão à média) |
| `W·`% de esquerda na Câmara | 0,0064 | 0,0031 | 1,006 | sim | vizinhança mais à esquerda soma um efeito pequeno |
| `W·`comparecimento no 1º turno | −0,0013 | 0,0041 | 0,999 | não | sem efeito distinguível de zero |
| `W·`log da densidade | 0,0122 | 0,0056 | 1,012 | sim | vizinhança mais densa soma um efeito positivo |
| `W·`% de mulheres | 0,0110 | 0,0042 | 1,011 | sim | vizinhança com mais mulheres soma um efeito positivo |
| `W·`% de 16 a 24 anos | −0,0038 | 0,0072 | 0,996 | não | sem efeito distinguível de zero |
| `W·`% com ensino superior | −0,0032 | 0,0039 | 0,997 | não | sem efeito distinguível de zero |
| `W·`% de analfabetos | −0,0130 | 0,0056 | 0,987 | sim | vizinhança com mais analfabetos reduz a chance de Lula |
| `W·`cobertura do Bolsa Família | 0,0525 | 0,0089 | 1,054 | sim | vizinhança com mais Bolsa Família soma um efeito positivo (maior do bloco espacial) |
| `W·`% do VAB agropecuário | −0,0053 | 0,0040 | 0,995 | não | sem efeito distinguível de zero |
| `W·`% de evangélicos | −0,0088 | 0,0039 | 0,991 | sim | vizinhança com mais evangélicos reduz a chance de Lula |
| `W·`% de pretos e pardos | −0,0004 | 0,0093 | 1,000 | não | sem efeito distinguível de zero |
| prefeito de esquerda | −0,0094 | 0,0050 | 0,991 | não | prefeito de esquerda (contra direita): sem efeito distinguível de zero |
| prefeito de centro/sem prefeito | 0,0214 | 0,0083 | 1,022 | sim | prefeito de centro ou sem prefeito (contra direita): chance maior |
| Centro-Oeste | −0,0191 | 0,0152 | 0,981 | não | Centro-Oeste contra Sudeste: sem diferença distinguível |
| Sul | −0,0544 | 0,0153 | 0,947 | sim | Sul contra Sudeste: chance menor |
| Nordeste+Norte | 0,0133 | 0,0141 | 1,013 | não | Nordeste e Norte (fundidos) contra Sudeste: sem diferença distinguível |

**Ressalva de colinearidade.** No `bb_slx`, seis termos têm VIF > 10: `W·pct_pretos_pardos` (21,5), `W·defasagem` (18,4), `W·cobertura_pbf` (17,2), `pct_pretos_pardos` (17,0), `W·pct_16_24` (12,5) e a defasagem (12,0). Cada covariável compartilha informação com a própria defasagem espacial, então os coeficientes desses termos, lidos um a um, são frágeis; a previsão não é afetada ([diagnóstico](docs/pressupostos/04-diagnostico.md), seção 8.6).

Leitura geral: a defasagem de 2022 explica quase tudo (OR 2,17 por DP); as demais covariáveis ajustam o nível em poucos pontos percentuais, e a vizinhança (termos `W·`) tem efeitos menores e menos estáveis, o que é coerente com o SLX ter reduzido pouco o I de Moran.

### Superdispersão

```text
φ de Pearson (binomial logit) = 21,8
beta-binomial: θ ≈ 624, ρ ≈ 0,0016
DP dos resíduos quantílicos: 1,03 (beta-binomial) contra 3,45 (binomial)
```

Com $n_i$ de milhares, a binomial pura subestima a variância por um fator da ordem de 20; por isso os EPs do binomial puro não valem.

### Seleção de variáveis

O passo a passo por BIC (backward, forward e both) dá o mesmo modelo: 11 contínuas além da defasagem, BIC 72.844,87, LOSO 1,035. A busca em todos os subconjuntos ficou em BIC 72.848,78, sem alcançar esse ótimo.

### Seleção por validação cruzada (LOSO, MAE por UF em p.p.)

```text
melhor              bin_probit = 0,998  (EP 0,135)
limite de 1 EP      1,133   (13 de 18 modelos dentro)
escolhido           bb_selecao = 1,035  (k = 19, menor número de parâmetros dentro do limite)
limite de aceitação L = 1,1168 (aninhado) + 0,1347 = 1,2515
bb_disp = 1,035 (aceito)   bb_slx = 1,078 (aceito, final)   bb_esf = 1,484 (rejeitado)
```

O `bb_slx` tem MAE por UF de 0,846 no 5-fold e 1,078 no LOSO.

### Dispersão regional

```text
Wald robusto (Conley, 200 km): dummies de região p = 1,0e-11 -> mantidas; log n p = 8,3e-16 -> mantido
θ do Sudeste ≈ 1.101; Norte ≈ 320 (a mais dispersa); efeito de log n: 0,277 em log θ
```

Leitura: a dispersão varia por região (o Norte tem θ menor, isto é, mais variância extra-binomial) e cresce com $n$.

### Dependência espacial

```text
I de Moran dos resíduos: 0,342 (sem termo espacial) -> 0,336 (SLX)
filtro de autovetores (k = 50): LOSO 1,484 > 1,2515 -> rejeitado
krigagem LOSO dos resíduos e CAR/BYM: rejeitadas
```

O SLX é aceito pela regra de validação, mas quase não reduz o I de Moran: a autocorrelação espacial residual persiste e os EPs de Conley corrigem a inferência, não as estimativas pontuais ([`docs/pressupostos/08-dependencia-espacial.md`](docs/pressupostos/08-dependencia-espacial.md), [`docs/pressupostos/10-extensoes-rejeitadas.md`](docs/pressupostos/10-extensoes-rejeitadas.md)).

### Estimativas regionais contra o real

Participação de Lula (%) na estimativa fora da amostra (LOSO) com Conley e $\sigma_{UF}$ = 0,0564; intervalo de 95%:

| Região | Estimado | Intervalo 95% | Real | Erro (p.p.) | Dentro |
| --- | --- | --- | --- | --- | --- |
| Centro-Oeste | 40,09 | [38,58; 41,62] | 39,99 | 0,10 | sim |
| Nordeste | 69,44 | [68,49; 70,29] | 69,37 | 0,07 | sim |
| Norte | 49,18 | [47,68; 50,72] | 50,01 | −0,82 | sim |
| Sudeste | 47,89 | [46,33; 49,56] | 47,24 | 0,65 | sim |
| Sul | 36,93 | [35,45; 38,46] | 37,42 | −0,49 | sim |
| Brasil | 51,76 | [51,02; 52,55] | 51,78 | −0,02 | sim |

Erro regional absoluto médio de 0,426 p.p. nas 5 regiões. As estimativas ficam dentro de 1 p.p. do real em todas as regiões e o Brasil dentro de 0,02 p.p.; o maior erro é o do Norte (−0,82). São só 5 regiões, e o erro regional LOSO não escolhe modelos: é descrição, não evidência de superioridade.

### Cobertura dos intervalos

| Covariância + σ | UF 95% | Região 95% |
| --- | --- | --- |
| Conley + σ_UF | 100% (27/27) | 5/5 |
| Conley + σ = 0 | 51,9% (14/27) | 2/5 |

O choque de UF é o que sustenta a cobertura. Como $\sigma_{UF}$ é ajustado nas mesmas UFs em que a cobertura é conferida, ela é **otimista**.

### Backtest

Melhor variante: `C_recal` (participações do 1º turno de 2026, intercepto ancorado na participação nacional de 2022), com erro de 0,58 p.p. por UF e 0,68 p.p. por região. A linha de base ingênua (repetir 2022) dá 1,65 (UF) e 0,64 (região). Sem o 1º turno, os modelos ficam piores que a linha de base (UF de 3,3 a 4,3) ([`docs/pressupostos/06-backtest.md`](docs/pressupostos/06-backtest.md)).

### Apuração parcial

Regra: o modelo é "útil" se vence a extrapolação bruta e o swing em pelo menos 4 de 6 cortes de urnas apuradas (5% a 90%).

* 2022: 4/6 cortes, "modelo útil"; 2026: 5/6, "modelo útil". O veredito é o mesmo com as duas definições de $n_u$ (votos restantes);
* o Brasil fica dentro do intervalo de 95% nos 12 de 12 cortes (2022 e 2026);
* o ganho aparece cedo, mas não em 5% das urnas: em 2026 com 5% apurado, o erro regional médio é 1,62 p.p. no modelo contra 1,49 no swing; com 25% cai para 0,25 contra 0,46 ([`docs/pressupostos/11-apuracao-parcial.md`](docs/pressupostos/11-apuracao-parcial.md)).

### Limitações

> [!WARNING]
> **Falácia ecológica (Robinson, 1950).** O modelo descreve **municípios**, não eleitores. "Municípios com mais evangélicos votam menos em Lula" **não implica** que evangélicos votem menos em Lula: a associação agregada pode existir sem que ela valha para o indivíduo (ou com sinal oposto). Os coeficientes acima são associações entre municípios, condicionais às demais variáveis, e não efeitos causais nem comportamento individual.

Demais limitações:

* **Autocorrelação espacial residual** (I de Moran 0,336): Conley corrige a inferência, não as estimativas pontuais;
* **Cobertura otimista:** $\sigma_{UF}$ vem das mesmas UFs; sem ele, só 14 das 27 UFs ficam dentro do intervalo; falta validação aninhada;
* **Seleção:** 13 dos 18 modelos ficam dentro de 1 EP, e a escolha repousa na parcimônia, não numa vantagem demonstrada;
* **Pouca base regional:** só 5 regiões e 27 UFs para avaliar o erro; o erro regional não escolhe modelo;
* **Regras tardias:** o Wald robusto, o SLX e o corte de Conley foram fixados antes de cada resultado, mas não antes do notebook 05 inteiro; uma versão anterior da validação vazava informação da UF de teste (corrigida);
* **Covariáveis defasadas:** o comparecimento é o do 1º turno e nenhuma covariável do 2º turno entra; `pct_vab_agro` usa 2021;
* **Dados:** Bolsa Família 2022 com 20,07 milhões de famílias, 2,81% abaixo dos 20,65 milhões da folha de setembro de 2022 (Ministério da Cidadania, nota de 16/09/2022, republicada pelo TNH1; o original no gov.br não foi encontrado; a base é família na folha, não NIS distinto pago); a API do MDS está indisponível e a fonte é a CGU;
* **Apuração parcial:** o desempenho em 5% das urnas não supera o swing.

Pendências abertas, com efeito e correção possível: [`docs/pressupostos/12-pendencias.md`](docs/pressupostos/12-pendencias.md).

---


## 📚 Referências

* Bolognesi, B.; Ribeiro, E.; Codato, A. (2023). Uma nova classificação ideológica dos partidos políticos brasileiros. *Dados*, 66(2). https://doi.org/10.1590/dados.2023.66.2.303
* Paula, G. A. (2013). *Modelos de Regressão com Apoio Computacional*. IME-USP. https://www.ime.usp.br/~giapaula/texto_2013.pdf
* McCullagh, P.; Nelder, J. A. (1989). *Generalized Linear Models* (2ª ed.). Chapman & Hall.
* Williams, D. A. (1975). The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity. *Biometrics*, 31(4), 949-952 (beta-binomial).
* Smyth, G. K. (1989). Generalized linear models with varying dispersion. *Journal of the Royal Statistical Society: Series B*, 51(1), 47-60.
* Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
* Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
* Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer (regra de 1 erro-padrão, cap. 7).
* Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review*, 15(3), 351-357.
* Dunn, P. K.; Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244.
* Tiefelsdorf, M.; Griffith, D. A. (2007). Semiparametric filtering of spatial autocorrelation: the eigenvector approach. *Environment and Planning A*, 39(5), 1193-1221.
* Breslow, N. E.; Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *Journal of the American Statistical Association*, 88(421), 9-25.
* Rey, S. J.; Anselin, L. (2010). PySAL: a Python library of spatial analytical methods. In *Handbook of Applied Spatial Analysis*. Springer, 175-193.
* TSE. Dados abertos (resultados, perfil do eleitorado e candidaturas). https://dadosabertos.tse.jus.br
* IBGE. Censo Demográfico 2022 e PIB dos Municípios. https://www.ibge.gov.br
* CGU. Portal da Transparência (Auxílio Brasil e Bolsa Família). https://portaldatransparencia.gov.br

Lista completa e conferida: [`docs/pressupostos/referencias.md`](docs/pressupostos/referencias.md).


---

## 👥 Autores

**José de Ribamar Ferreira Neto**

**Matheus Finger Ramos de Oliveira**

---

## 🎓 Contexto acadêmico

Projeto desenvolvido como parte das atividades da disciplina de **Modelagem Estatística**, com foco na aplicação de **Modelos Lineares Generalizados (MLGs)**, inferência estatística e dependência espacial.

