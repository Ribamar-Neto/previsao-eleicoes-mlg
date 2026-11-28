# 15 - Apuração parcial: corte precoce de 2 % e calibração do IC (Parte 18)

Notebook `notebooks/03-apuracao-parcial.ipynb` (Tabela 10). Tabela `results/tables/apuracao_precoce.csv`. O desenho e o
veredito estão em [11](11-apuracao-parcial.md); o diagnóstico anterior do IC, em [14](14-apuracao-diagnostico-1t.md).
O [11](11-apuracao-parcial.md) está no limite de tamanho, por isso este arquivo.

## 1. Corte precoce de 2 %: decisão

Decisão do usuário (2026-11-27): o corte de 2 % entra separado, com tabela própria
(`results/tables/apuracao_precoce.csv`, Tabela 10 do notebook 03).

- **Fora do veredito.** Não entra no veredito, na regra de aceitação (mínimo de 4 vitórias), na Tabela 5 nem nos
  agregados dos "12 cortes". Essas regras foram fixadas antes do corte de 2 %; incluí-lo mudaria o que o veredito mede.
- **Mesmo método.** Mesmo estimador de IC e mesmo desenho por corte dos demais cortes.
- **"Não estimável".** Significa que o desenho ou o ajuste falhou (matriz singular, região sem município apurado). A linha
  guarda o motivo e o resultado real do Brasil; nada é contado como vitória.

## 2. Resultado do corte de 2 %

Fonte dos números: `results/tables/apuracao_precoce.csv`.

| | 2022 | 2026 |
|---|---|---|
| Apuração efetiva | 2,005 % | 2,004 % |
| Municípios apurados | 657 | 960 |
| UFs elegíveis (k, com pelo menos 5 municípios apurados) | 20 | 24 |

Só Sergipe não tem município apurado, nos dois anos. Nenhuma região ficou vazia.

Brasil (parcela de Lula, %):

| Ano | Modelo | IC95 | Final | Bruta | Swing 1º turno | Final no IC? |
|---|---|---|---|---|---|---|
| 2022 | 50,91 | [50,50; 51,33] | 50,90 | 42,99 | 51,53 | dentro |
| 2026 | 51,42 | [51,09; 51,76] | 51,78 | 50,95 | 51,48 | fora |

Em 2026 o final fica cerca de 0,02 p.p. acima do limite superior. As 10 regiões-ano ficam dentro do IC95.

| Ano | Região | Modelo (%) | IC95 (%) | Final (%) |
|---|---|---|---|---|
| 2022 | Centro-Oeste | 39,89 | [39,27; 40,59] | 39,79 |
| 2022 | Nordeste | 69,40 | [68,91; 69,89] | 69,34 |
| 2022 | Norte | 49,39 | [48,74; 50,12] | 48,97 |
| 2022 | Sudeste | 45,40 | [44,59; 46,24] | 45,74 |
| 2022 | Sul | 38,47 | [37,67; 39,19] | 38,16 |
| 2026 | Centro-Oeste | 39,78 | [39,27; 40,27] | 39,99 |
| 2026 | Nordeste | 69,48 | [69,13; 69,85] | 69,37 |
| 2026 | Norte | 49,96 | [49,45; 50,45] | 50,01 |
| 2026 | Sudeste | 46,55 | [45,88; 47,26] | 47,24 |
| 2026 | Sul | 37,04 | [36,49; 37,59] | 37,42 |

## 3. Diagnóstico da largura do IC

Evidência de partida (`results/tables/apuracao_ic.csv`, 72 itens): z² médio = 0,52, com z = 1,96·erro/meia-largura
(calibrado ≈ 1). Brasil 2022: 1,14; Brasil 2026: 0,44; regiões 2022: 0,52; regiões 2026: 0,42. A cobertura equivalente
normal em 50, 80 e 90 % é 67, 94 e 100 %. O erro do Brasil tem um só sinal por ano (2022 todo positivo, 2026 todo
negativo).

### 3.1 Regra R-cal, fixada antes dos resultados

Um componente é "claramente superestimado" se, e somente se:

- (a) ρ = RMS realizado na parte restante / DP modelado tem mediana < 0,75 nos 6 cortes em CADA ano. Para o choque de UF,
  o DP modelado é σ̂_UF·√(k/(k−2)) no logit; para n_u, é o DP em log n.
- (b) um estimador estrutural, definido sem o desfecho, fecha pelo menos metade da lacuna 1 − ρ. H1: subtrair o ruído já
  presente no resíduo LOSO. H2: alvo da parte restante por duas metades dos municípios apurados. H3: separar parte comum e
  parte de UF.

Só então esse estimador substitui o atual. Nenhum fator de escala é ajustado nos desfechos. Depois de uma mudança, a
aceitação do [14](14-apuracao-diagnostico-1t.md) (seção 4) tem de valer nos 12 cortes.

### 3.2 Resultado

Mediana de ρ (2022 / 2026):

| Componente | 2022 | 2026 |
|---|---|---|
| Choque de UF, sem peso | 1,15 | 0,77 |
| Choque de UF, pesado pelos votos restantes | 0,61 | 0,83 |
| n_u, nível de UF | 1,07 | 1,02 |
| n_u, municipal | 0,99 | 1,05 |

A condição (a) falha em todo componente. Lacuna fechada (mediana, 2022 / 2026):

| Estimador | Sem peso | Pesado pelos votos |
|---|---|---|
| H1 | −0,52 / +0,24 | +0,30 / +0,29 |
| H1, só β | −0,05 / +0,03 | +0,03 / +0,04 |
| H2 | +0,06 / −0,18 | −0,03 / −0,19 |
| H3 | −0,04 / −0,01 | −0,14 / −0,03 |

A condição (b) também falha. Decisão: a R-cal não disparou; o IC fica como está.

### 3.3 Outros achados

- O ruído binomial/θ é de 20 a 57 % da variância do resíduo LOSO que sustenta σ_UF; o ruído de β é ≈ 0.
- Fator t √(k/(k−2)) = 1,041 a 1,044.
- Parcela da variância: θ 0,02 a 0,23; choque de UF 0,35 a 0,73.
- Deslocamento comum (resíduo médio das UFs): −0,008 a −0,012 em 2022 e ≈ ±0,003 em 2026.
- PIT: Brasil 2022 entre 0,10 e 0,32 em todo corte; Brasil 2026 entre 0,45 e 0,86; regiões, 0 de 60 fora de
  [0,025; 0,975]; |PIT − 0,5| médio = 0,20 nos dois anos (0,25 se uniforme).

## 4. Explicação mais provável e limites

Explicação líder, não testada diretamente: o DP total modelado da parcela restante de uma UF (σ_UF·t + β + binomial/θ)
fica superestimado quando pesado pelos votos restantes. O ρ total é 0,58 em 2022 e 0,70 em 2026 (sem peso: 0,91 e
0,64). UFs grandes têm resíduos menores que o σ_UF sem peso, e as parcelas do Brasil e das regiões pesam as UFs pelos
votos.

Achado lateral: em 2026 os votos restantes reais superam o n_u estimado em ≈ 4 % no nível Brasil (razão em log de 0,040
a 0,049), um viés que o choque de n_u não modela.

Limites: são duas eleições, com erros correlacionados dentro do ano; ρ e z² são evidência fraca. O erro de 2 % em 2026
mostra que um IC mais estreito custaria cobertura no início da apuração.

## 5. Origem dos números

Diagnóstico rodado fora do repositório em 2026-11-27, com scripts temporários (não versionados), lendo só os 12
desenhos em cache (`results/cv/desenho_parcial_*.json`, VERSAO 5). A recomputação LOSO reproduziu todo σ_UF em cache e
o z² guardado. Saídas: `rho_uf.csv`, `rho_uf_resumo.csv`, `rho_nu.csv`, `h1_ruido.csv`, `h1_dobras.csv`, `pit.csv`,
`h5_t_theta.csv`, `rcal.csv`. Esses arquivos não estão versionados; só a decisão e os números acima são o registro.
