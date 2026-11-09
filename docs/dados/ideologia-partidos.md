# Ideologia dos partidos e dos prefeitos

Script: `src/coleta/tse_ideologia.py` -> `data/interim/tse_ideologia.parquet`
(uma linha por `id_municipio_tse`, 5.571 municípios).
Tabela de referência: `data/referencia/ideologia_partidos.csv`.

## Fonte da classificação

Bolognesi, Ribeiro e Codato (2023), "Uma nova classificação ideológica dos partidos
políticos brasileiros", *Dados* 66(2), doi 10.1590/dados.2023.66.2.303 (survey com
especialistas; nota média de 0 = esquerda a 10 = direita).

A Tabela 1 do artigo traz a nota média por partido. Os rótulos de categoria foram
atribuídos pelos cortes descritos no texto: 0-1,5 extrema-esquerda; 1,51-3 esquerda;
3,01-4,49 centro-esquerda; 4,5-5,5 centro; 5,51-7 centro-direita; 7,01-8,5 direita;
8,51-10 extrema-direita. O PL não consta na tabela: usa-se o PR (mesmo partido, renomeado).
## Colapso em 3 blocos

| Bloco    | Categorias do artigo                                |
|----------|-----------------------------------------------------|
| esquerda | extrema-esquerda, esquerda, centro-esquerda         |
| centro   | centro                                              |
| direita  | centro-direita, direita, extrema-direita           |

## Partidos fundidos, renomeados e novos

A coluna `regra` do CSV registra a origem de cada linha.

- Renomeações usam a nota do predecessor: Cidadania = PPS; Republicanos = PRB;
  PL = PR; Agir = PTC; Mobiliza = PMN; Democrata (2026) = PMB (número 35);
  Avante = Avante/PTdoB; Solidariedade = SDD; Progressistas = PP; Podemos = PODE.
- Incorporações: Podemos absorveu PSC e Solidariedade absorveu PROS. Os votos de 2018
  são classificados pela sigla vigente na época (PSC, PROS), os de 2022 em diante
  pela sigla atual; o bloco é o mesmo (direita).
- Fusões: União Brasil = DEM + PSL (notas 8,57 e 8,11, média 8,34); PRD = PTB + Patriota
  (notas 6,10 e 8,55, média 7,33). Em ambos os casos os dois predecessores caem no mesmo
  bloco (direita), então não houve conflito.
- Sem classificação no artigo (`regra = sem_classificacao_BRC: ...`), classificados pelo
  programa do partido, sem nota:
  - UP (Unidade Popular): esquerda, partido socialista de esquerda radical.
  - Missão (criado a partir do MBL): direita, programa liberal-conservador.
  Julgamento do projeto, não do artigo.

## Medida contínua e 7 classes

Cortes do artigo: "0-1,5 extrema-esquerda; 1,51-3 esquerda; 3,01-4,49 centro-esquerda;
4,5-5,5 centro; 5,51-7 centro-direita; 7,01-8,5 direita; 8,51-10 extrema-direita".
O bloco "centro" tem só 1,7% a 4,1% dos votos e "direita" junta MDB/PSD com PL; por isso
há também `nota_media_dep_{ano}` (média das notas ponderada pelos votos, só partidos com
nota; UP e Missão ficam fora) e `pct_dep_{classe}_{ano}` para as 7 classes
(`ext_esq, esq, centro_esq, centro, centro_dir, dir, ext_dir`), sobre os votos válidos.
UP entra em `esq` e Missão em `dir` (julgamento do projeto); União e PRD usam os cortes
sobre a média dos predecessores (ambos `dir`).

## Federações

O TSE registra os votos por partido-membro (`SG_PARTIDO`), e a federação aparece em
coluna separada. Os votos são então atribuídos pelo partido, não pela federação.
Em 2022 e 2026, a federação PT/PCdoB/PV soma esquerda (PT, PCdoB) e centro (PV).

## Votos em deputado federal

- Fonte: `votacao_partido_munzona_{2018,2022,2026}` (arquivo `_BRASIL`), 1º turno,
  cargo Deputado Federal.
- Votos do partido = nominais válidos + de legenda válidos.
- Exclui o exterior (`ZZ`) e `BR`/`VT`.
- `pct_dep_*` = votos do bloco / votos válidos de partidos classificados no município.
  As três frações somam 1.
- Boa Esperança do Norte (MT, 73709) só existe a partir de 2024: fica com votos de
  2018 e 2022 vazios (NaN).

## Prefeito

- Fonte: `consulta_cand_{2020,2024}`, cargo Prefeito, situação `ELEITO*`; em municípios
  com 2º turno vale o eleito do 2º turno.
- Se a eleição ordinária foi anulada e houve suplementar, vale o eleito da suplementar
  (prefeito em exercício). A coluna `prefeito_suplementar_{ano}` marca esses casos.
- Sem prefeito eleito no arquivo: `bloco_prefeito = 'sem_prefeito'` (Brasília e Fernando
  de Noronha por natureza; os demais por eleição sem resultado definido no arquivo do TSE
  na data da coleta, sobretudo em 2024).
