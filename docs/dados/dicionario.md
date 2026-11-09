# Dicionário de dados: `data/processed/municipios.parquet`

Uma linha por município com Censo 2022 (5.570). Gerado por `src/coleta/consolidar.py`.
Todas as proporções ("pct_", "prop_", "taxa_", "cobertura_") são frações em [0, 1],
exceto `cobertura_pbf_*` (pode passar de 1) e `pct_vab_agro_2021` (um valor negativo do IBGE).
Fontes: IBGE-MUN = `ibge_municipios`; TSE-PRES = resultados de presidente (TSE);
TSE-DEP = deputados e prefeitos (TSE) classificados por
[ideologia-partidos.md](ideologia-partidos.md); TSE-PERFIL = perfil do eleitorado (TSE);
IBGE-CENSO = Censo 2022 e PIB dos Municípios (IBGE); MDS = Bolsa Família (Portal da Transparência).

## Identificação e geografia (IBGE-MUN; período: malha vigente no Censo 2022)

| Coluna | Descrição | Unidade |
|---|---|---|
| id_municipio | Código IBGE de 7 dígitos (chave) | inteiro |
| id_municipio_tse | Código do município no TSE | inteiro |
| nome_municipio | Nome do município | texto |
| sigla_uf, id_uf, nome_uf | UF (sigla é categórica) | texto / inteiro |
| regiao | Grande região (categórica, 5 níveis) | texto |
| area_km2 | Área territorial | km² |
| lat, lon | Coordenadas da sede | graus |

## Votos presidenciais (TSE-PRES)

Sufixo `_{t}t_{ano}`: turno t (1, 2) e ano 2018, 2022, 2026. "pt" = Haddad (2018) ou Lula
(2022, 2026); "bolsonaro" = Jair (2018, 2022) ou Flávio (2026). Votos no Brasil
(exterior excluído).

| Coluna | Descrição | Unidade | Período |
|---|---|---|---|
| votos_pt_{t}t_{ano} | Votos no candidato do PT | votos | eleição de {ano} |
| votos_bolsonaro_{t}t_{ano} | Votos no candidato bolsonarista | votos | idem |
| validos_{t}t_{ano} | Votos válidos (candidatos + legenda, sem brancos e nulos) | votos | idem |
| aptos_{t}t_{ano} | Eleitores aptos | pessoas | idem |
| comparecimento_{t}t_{ano} | Eleitores que compareceram | pessoas | idem |
| abstencoes_{t}t_{ano} | Abstenções | pessoas | idem |
| brancos_{t}t_{ano} | Votos em branco | votos | idem |
| nulos_{t}t_{ano} | Votos nulos | votos | idem |

## Variáveis derivadas do voto (calculadas aqui; ano em 2018, 2022, 2026)

| Coluna | Descrição | Unidade |
|---|---|---|
| n_2t_{ano} | votos_pt_2t + votos_bolsonaro_2t (denominador do GLM binomial) | votos |
| prop_pt_2t_{ano} | votos_pt_2t / n_2t | fração |
| logit_pt_2t_{ano} | log(p / (1 - p)) de prop_pt_2t | log-odds |
| prop_pt_1t_{ano} | votos_pt_1t / validos_1t | fração |
| prop_bolsonaro_1t_{ano} | votos_bolsonaro_1t / validos_1t | fração |
| taxa_comparecimento_{t}t_{ano} | comparecimento / aptos | fração |
| delta_comparecimento_2t_2022 | taxa_comparecimento_2t 2022 menos 2018 | pontos (fração) |
| delta_comparecimento_2t_2026 | taxa_comparecimento_2t 2026 menos 2022 | pontos (fração) |
| swing_logit_2022 | logit_pt_2t_2022 menos logit_pt_2t_2018 | log-odds |
| swing_logit_2026 | logit_pt_2t_2026 menos logit_pt_2t_2022 | log-odds |

## Voto em deputados e prefeitos (TSE-DEP)

Votos em deputado por bloco ideológico do partido;
ano de deputado: 2018, 2022, 2026. Prefeitos: eleições municipais 2020 e 2024.

| Coluna | Descrição | Unidade |
|---|---|---|
| votos_dep_esq_{ano}, votos_dep_centro_{ano}, votos_dep_dir_{ano} | Votos em partidos de cada bloco | votos |
| votos_dep_validos_{ano} | Votos válidos para deputado | votos |
| pct_dep_esq_{ano}, pct_dep_centro_{ano}, pct_dep_dir_{ano} | Parcela de cada bloco nos válidos | fração |
| pct_dep_ext_esq_, pct_dep_esq_, pct_dep_centro_esq_, pct_dep_centro_, pct_dep_centro_dir_, pct_dep_dir_, pct_dep_ext_dir_ + {ano} | Parcela das 7 categorias do artigo | fração |
| nota_media_dep_{ano} | Nota ideológica média ponderada por votos (0 esquerda a 10 direita) | escala 0-10 |
| partido_prefeito_{2020,2024} | Partido do prefeito eleito | texto |
| bloco_prefeito_{2020,2024} | esquerda, centro, direita ou sem_prefeito | categoria |
| prefeito_suplementar_{2020,2024} | Eleição suplementar substituiu o prefeito | booleano |

## Perfil do eleitorado (TSE-PERFIL; ano 2022 e 2026)

Denominadores excluem "não informado". O perfil de 2026 foi gerado em 14/07/2026.

| Coluna | Descrição | Unidade |
|---|---|---|
| eleitorado_{ano} | Eleitores aptos no perfil | pessoas |
| pct_mulheres_{ano} | Mulheres | fração |
| pct_16_24_{ano} | Idade 16 a 24 anos | fração |
| pct_60_mais_{ano} | Idade 60 anos ou mais | fração |
| pct_analfabeto_{ano} | Analfabetos | fração |
| pct_ate_fund_incompleto_{ano} | Até fundamental incompleto | fração |
| pct_superior_completo_{ano} | Superior completo | fração |

## Censo e PIB (IBGE-CENSO)

| Coluna | Descrição | Unidade | Período |
|---|---|---|---|
| populacao_2022 | População residente | pessoas | Censo 2022 |
| domicilios_2022 | Domicílios | unidades | Censo 2022 |
| pct_pretos_pardos_2022 | Pretos e pardos | fração | Censo 2022 |
| pct_indigenas_2022 | Indígenas | fração | Censo 2022 |
| taxa_alfabetizacao_15mais_2022 | Alfabetizados entre 15 anos ou mais | fração | Censo 2022 |
| pct_evangelicos_2022, pct_catolicos_2022, pct_sem_religiao_2022 | Religião; denominador: pessoas de 10 anos ou mais (tabela 9537) | fração | Censo 2022 |
| pct_urbana_2022 | População em situação urbana | fração | Censo 2022 |
| pib_per_capita_2021 | PIB municipal / população | R$ correntes | 2021 |
| pib_per_capita_2023 | PIB municipal 2023 / população do Censo 2022 | R$ correntes | 2023 |
| pct_vab_agro_2021 | Agropecuária no valor adicionado bruto | fração | 2021 |

## Bolsa Família (MDS)

| Coluna | Descrição | Unidade | Período |
|---|---|---|---|
| familias_pbf_2022 | Auxílio Brasil: NIS distintos pagos, por município de pagamento | famílias | setembro 2022 |
| familias_pbf_2026 | Novo Bolsa Família, por município de pagamento | famílias | setembro 2026 |

## Demais derivadas

| Coluna | Descrição | Unidade |
|---|---|---|
| log_populacao_2022 | ln(populacao_2022) | ln(pessoas) |
| log_eleitorado_2022, log_eleitorado_2026 | ln(eleitorado) | ln(pessoas) |
| densidade_2022 | populacao_2022 / area_km2 | hab/km² |
| log_densidade_2022 | ln(densidade_2022) | ln(hab/km²) |
| cobertura_pbf_2022, cobertura_pbf_2026 | familias_pbf / domicilios_2022; sem truncar | razão |
