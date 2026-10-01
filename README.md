# CP2-SERS — APIs de energia renovável e Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedrosampaiom2007-gif/CP2-SERS/blob/main/Aula_APIs_Energia_Renovavel_ML.ipynb)

Trabalho que consulta duas APIs públicas de energia/clima, organiza os dados e resolve duas tarefas de aprendizado de máquina, comparando **três algoritmos em cada uma**.

| Tarefa | Tipo | Pergunta |
|---|---|---|
| 1 — ANEEL | Classificação | Dá para identificar a fonte (Solar, Eólica ou Hidráulica) de um empreendimento a partir de potência e localização? |
| 2 — Open-Meteo | Regressão | Qual a radiação solar horária (W/m²) em Petrolina (PE) dadas as condições do tempo e a hora? |

## Arquivos

- `Aula_APIs_Energia_Renovavel_ML.ipynb` — notebook completo (consulta às APIs, análise, modelos e conclusões).
- `aneel_classificacao_orange.csv` e `meteo_regressao_orange.csv` — gerados pelas células de exportação do notebook (ainda não versionados neste repositório; rode o notebook para criá-los). O notebook de classificação lê o CSV da ANEEL direto do repositório do professor.

## Fontes e período dos dados

- **ANEEL — SIGA** ([link](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel)), API CKAN/DataStore. Siglas usadas: `UFV` (Solar), `EOL` (Eólica) e `UHE`/`PCH`/`CGH` (agrupadas como Hidráulica).
- **Open-Meteo — histórico** ([link](https://open-meteo.com/en/docs/historical-weather-api)), Petrolina (lat -9,39 / lon -40,50), **01/04/2025 a 30/06/2025**, fuso `America/Recife`, apenas horas de 7h a 17h.

Nenhuma das duas consultas exige login, token ou chave.

## Como executar

1. Abra o notebook no Colab (botão acima) ou localmente com Jupyter.
2. Dependências: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
3. Execute as células em ordem (precisa de internet para as APIs).

## O que foi feito

### Tarefa 1 — Classificação (ANEEL)

1. **Coleta:** 3.876 empreendimentos (1.200 UFV, 1.200 EOL, 221 UHE, 537 PCH, 718 CGH), convertidos para o CSV com `potencia_kw`, `latitude`, `longitude` e `fonte`.
2. **Análise:** sem valores ausentes; classes com 1.476 Hidráulica, 1.200 Solar e 1.200 Eólica (pouco desbalanceadas). Foi observado que alguns registros têm coordenadas `0,0`.
3. **Modelagem:** `X` = potência + latitude + longitude; `y` = fonte. Nenhum campo que "entrega" a classe (sigla, nome, CEG) foi usado como entrada. Divisão estratificada com `random_state=42`.
4. **Modelos comparados:** Regressão Logística, KNN e Random Forest.

| Modelo | Acurácia |
|---|---|
| Regressão Logística | 0,713 |
| KNN | 0,806 |
| **Random Forest** | **0,955** |

Também foram geradas `classification_report` (precision, recall, F1) e matrizes de confusão dos três modelos.

**Conclusão:** o Random Forest é o melhor; a classe mais confundida é Hidráulica (principalmente com Solar e Eólica). Potência e localização não bastam para uma aplicação real, pois não capturam microclima, tecnologia, restrições regulatórias/infraestrutura nem geração efetiva (a potência outorgada é nominal).

> **Ressalva:** na divisão treino/teste do notebook, os nomes das variáveis estão trocados (`x_test, x_train, ... = train_test_split(..., test_size=0.2)`), de modo que o modelo foi treinado com 20% dos dados e testado com 80% (3.100 linhas de teste). Os resultados são válidos, mas a divisão pretendida (80% treino / 20% teste) deve ser corrigida invertendo os nomes. Além disso, o `StandardScaler` não foi usado nessa tarefa, o que prejudica a Regressão Logística (aviso de não convergência) e o KNN.

### Tarefa 2 — Regressão (Open-Meteo)

1. **Coleta:** 2.184 horas retornadas; após filtrar 7h–17h e remover nulos, **1.001 registros** com `temperatura_c`, `umidade_pct`, `nuvens_pct`, `vento_kmh`, `hora` e o alvo `radiacao_w_m2`.
2. **Análise:** sem valores ausentes; gráficos da radiação média por hora e da radiação × nuvens.
3. **Divisão temporal (sem embaralhar):** primeiras 800 horas para treino (01/04 → 12/06) e 201 finais para teste (12/06 → 30/06). `StandardScaler` ajustado só no treino e usado na Regressão Linear.
4. **Modelos comparados:** Regressão Linear, Árvore de Decisão (`max_depth=6`) e Random Forest (200 árvores, `max_depth=8`).

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---|---|---|
| Regressão Linear | 145,21 | 30.034 | 0,360 |
| Árvore de Decisão | 90,94 | 15.127 | 0,678 |
| **Random Forest** | **67,09** | **7.394** | **0,842** |

Também há gráficos de real × previsto, série temporal do teste, erro médio por hora e importância das variáveis.

**Conclusão:** a relação é não linear (curva do sol ao longo do dia), então os modelos de árvore superam a regressão linear. A `hora` é a variável mais importante e as nuvens explicam a variação entre dias. A radiação estimada **não é** a energia gerada por um sistema fotovoltaico: faltam área/eficiência dos painéis, temperatura do módulo, inclinação, sombreamento, perdas do sistema e a integração no tempo (W/m² → Wh).

## Resumo geral

Em ambas as tarefas o **Random Forest** foi o melhor algoritmo, mostrando que os dados têm relações não lineares que modelos simples (logística/linear) capturam mal.
