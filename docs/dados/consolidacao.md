# Consolidação das fontes

Script: `src/coleta/consolidar.py`. Saídas: `data/processed/municipios.parquet`
(5.570 linhas, 160 colunas), `results/tables/resultado_regional.csv`,
`results/tables/resultado_uf.csv`. Colunas: ver [dicionario.md](dicionario.md).
Classificação ideológica: ver [ideologia-partidos.md](ideologia-partidos.md).

## Regras do merge

- Unidade: município com Censo 2022. A base e a tabela de equivalência
  IBGE x TSE é `ibge_municipios`; as demais fontes entram por junção 1 para 1 e todas
  devem casar sem chave faltante.
- Chaves: IBGE (`id_municipio`) para Censo e Bolsa Família; TSE (`id_municipio_tse`)
  para votos, ideologia e perfil. As duas são inteiros.
- Boa Esperança do Norte (MT, IBGE 5101837, TSE 73709) foi instalado depois do Censo
  2022: não tem defasagem 2018/2022 nem dados do Censo, e fica fora da tabela. Seus
  votos de 2026 no segundo turno (PT 945, Bolsonaro 2.353) são reportados, não modelados.
- Verificação de totais: a soma do segundo turno de 2026 nos 5.570 municípios é igual ao
  total nacional do TSE no país (PT 64.056.078; Bolsonaro 59.654.544) menos os votos
  desse município.
- Resultados regional e por UF: soma dos votos dos 5.570 municípios; `prop_pt` =
  votos_pt / (votos_pt + votos_bolsonaro), ou seja, sobre votos nos dois candidatos.
  Não incluem exterior nem Boa Esperança do Norte.

## Fonte `tse_bweb` (apuração por seção)

Script: `src/coleta/tse_bweb.py`; saída `data/interim/tse_bweb.parquet` (471.010 seções em
2022 e 497.897 em 2026), fora do `municipios.parquet`. Usada só na
[Parte 18](../pressupostos/11-apuracao-parcial.md).

- O que é: boletim de urna (BU) do 2º turno, cargo Presidente, uma linha por seção, com o
  instante em que o BU foi recebido na totalização. Origem:
  `https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes<ano>/buweb/bweb_2t_<UF>_<carimbo>.zip`,
  um zip por UF; cache em `data/raw/tse_bweb/`.
- Fuso: `dt_bu_recebido` é horário de Brasília (`America/Sao_Paulo`) em todas as UFs. Conferido:
  o primeiro recebimento de cada UF vem depois do fechamento das 17h de Brasília, mesmo onde
  o fuso local é 2 h atrás. `DT_ENCERRAMENTO` da fonte é local e não é usado.
- Leiautes: 2022 traz data `dd/mm/aaaa hh:mm:ss` e `DS_AGREGADAS`; 2026 traz ISO
  `aaaa-mm-dd hh:mm:ss` e `DS_SECOES_AGREGADAS`. Os dois entram no mesmo esquema.
- Seções agregadas: a fonte traz só a seção principal; `agregadas` guarda a lista da fonte e
  não há linha para as secundárias.
- Só as 27 UFs (sem exterior). `validos` = Lula (13) + adversário (22).
- Conferência: os totais por município fecham exatamente com `votos_pt_2t_<ano>` e
  `n_2t_<ano>`, salvo Boa Esperança do Norte (TSE 73709), ausente do `municipios.parquet`:
  em 2026 são 945 votos de Lula em 3.298 válidos, que entram na Parte 18.
- Validação do esquema em Python puro (`validar_secoes`), porque Pandera não está nas
  dependências. A virada de 2022 cai às 18:44:11, com 68,1 % recebido (TSE: 18h44).

## Escalas

- O IBGE publica percentuais em 0-100; aqui todos viram fração em [0, 1], para ficarem na
  mesma escala dos percentuais do TSE. Convertidos: pretos e pardos, indígenas,
  alfabetização, evangélicos, católicos, sem religião, urbana, VAB agropecuário.
- A proporção do PT usa só os dois candidatos do segundo turno. Brancos e nulos ficam
  fora de `n_2t_*`, que é o número de tentativas do GLM binomial.

## Limitações conhecidas

- Religião: o denominador são pessoas de 10 anos ou mais (tabela 9537 do Censo), não a
  população total. Evangélicos, católicos e sem religião não somam 1 (há outras religiões).
- O IBGE não publica VAB por setor depois de 2021. O modelo de 2026 usa, portanto,
  `pct_vab_agro_2021`; a defasagem é de cinco anos.
- `pib_per_capita_2023` divide o PIB de 2023 pela população do Censo 2022, não por uma
  estimativa de 2023.
- `pct_vab_agro_2021` tem um valor negativo (município 5204250), igual ao do IBGE
; mantido.
- Bolsa Família conta famílias no município de pagamento, não de residência. Por isso
  `cobertura_pbf` passa de 1 em 83 municípios (2022) e 36 (2026); não foi truncada.
- Auxílio Brasil de 2022: contagem de NIS distintos no arquivo do Portal da Transparência,
  o único publicado para setembro de 2022. Não é igual ao número oficial de famílias.
- Os perfis de eleitorado têm denominador sem "não informado"; o de 2026 foi gerado em
  14/07/2026 e não reflete mudanças de cadastro posteriores.
- Prefeitos: 6 municípios sem partido em 2020 e 16 em 2024 (`bloco_prefeito` = sem_prefeito).
  São os únicos valores ausentes da tabela.
