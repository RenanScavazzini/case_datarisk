# Credit Risk Case

## Objetivo

Desenvolver uma solução completa de modelagem de risco de crédito para estimar a probabilidade de inadimplência de novos contratos presentes na base `base_submissao.parquet`.

O projeto contempla todas as etapas do ciclo analítico, desde a construção da população de modelagem até a definição de uma política de crédito baseada nas probabilidades estimadas pelo modelo final.

Ao final do processo é gerado um arquivo de submissão contendo a probabilidade estimada de inadimplência para cada cliente da base de score.

---

# Metodologia

A solução foi construída seguindo uma pipeline estruturada de modelagem de risco:

```text
População
↓
Target
↓
EDA
↓
Feature Engineering
↓
Feature Selection
↓
Modelagem
↓
Política de Crédito
```

---

# Estrutura do Projeto

```text
case_datarisk/

│
├── data/
│   ├── raw/
│   │   ├── base_cadastral.parquet
│   │   ├── base_submissao.parquet
│   │   ├── historico_emprestimos.parquet
│   │   └── historico_parcelas.parquet
│   │
│   └── processed/
│       ├── population_active.parquet
│       ├── population_score.parquet
│       ├── population_target.parquet
│       ├── train_base.parquet
│       ├── train_model.parquet
│       ├── train_rus_model.parquet
│       ├── oos_model.parquet
│       ├── oot_base.parquet
│       ├── oot_model.parquet
│       └── model_dataset.parquet
│
├── notebooks/
│   ├── 01_population.ipynb
│   ├── 02_target_definition.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_feature_selection.ipynb
│   ├── 06_modeling.ipynb
│   └── 07_credit_policy.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── population.py
│   ├── target.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── modeling.py
│   ├── policy.py
│   └── utils.py
│
├── outputs/
│   ├── dicts/
│   │   ├── tipos_var.pkl
│   │   ├── dicionario_imputacao.pkl
│   │   ├── dicionario_dominio.pkl
│   │   ├── woe_dictionary.pkl
│   │   ├── normalization_dictionary.pkl
│   │   ├── features_corr.pkl
│   │   └── best_model_features.pkl
│   │
│   ├── models/
│   │   └── best_model.pkl
│   │
│   └── submissions/
│       └── submissao_case.csv
│
├── docs/
│   └── Case Técnico DS.pdf
│
├── requirements.txt
├── requirements-dev.txt
├── runtime.txt
├── .gitignore
└── README.md
```

---

# Dados Utilizados

O projeto utiliza quatro fontes de dados:

| Base                  | Descrição                              |
| --------------------- | -------------------------------------- |
| base_cadastral        | Informações cadastrais dos clientes    |
| base_submissao        | Solicitações de crédito para escoragem |
| historico_emprestimos | Histórico de contratos de crédito      |
| historico_parcelas    | Histórico de pagamentos das parcelas   |

---

# Notebooks

## 01_population.ipynb

### Objetivos

* Construção da população ativa de modelagem;
* Construção da população de score;
* Integração entre dados cadastrais e históricos;
* Criação de atributos históricos de crédito;
* Definição da unidade de modelagem cliente + safra.

### Saídas

```text
population_active.parquet
population_score.parquet
```

---

## 02_target_definition.ipynb

### Objetivos

* Construção da variável alvo;
* Cálculo dos indicadores de atraso;
* Avaliação da distribuição temporal da inadimplência;
* Definição do target final.

### Definição Utilizada

```text
target = ever_45
```

Onde:

```text
0 = Adimplente
1 = Inadimplente
```

### Saída

```text
population_target.parquet
```

---

## 03_eda.ipynb

### Objetivos

* Análise exploratória dos dados;
* Tratamento e padronização dos tipos;
* Criação da variável idade;
* Tratamento de valores ausentes;
* Controle de domínio das variáveis;
* Weight of Evidence (WOE);
* Information Value (IV);
* Normalização Min-Max;
* Avaliação de correlação.

### Artefatos Gerados

```text
tipos_var.pkl
dicionario_imputacao.pkl
dicionario_dominio.pkl
woe_dictionary.pkl
normalization_dictionary.pkl
```

---

## 04_feature_engineering.ipynb

### Objetivos

Aplicar automaticamente todas as transformações definidas durante a etapa de EDA.

### Transformações

* Conversões de variáveis binárias;
* Criação da variável idade;
* Imputação de valores ausentes;
* Controle de domínio;
* Aplicação de WOE;
* Normalização.

### Divisão das Bases

```text
Train
OOS (Out of Sample)
OOT (Out of Time)
```

### Balanceamento

Aplicado apenas na base de treinamento utilizando:

```text
Random Under Sampling (RUS)
```

### Saídas

```text
train_model.parquet
train_rus_model.parquet
oos_model.parquet
oot_model.parquet
```

---

## 05_feature_selection.ipynb

### Objetivos

Reduzir redundâncias entre variáveis e minimizar problemas de multicolinearidade.

### Técnicas Utilizadas

* Variance Inflation Factor (VIF);
* Correlação Linear.

### Observação

O Information Value (IV) foi utilizado apenas como análise complementar, não sendo empregado como critério de exclusão de variáveis.

### Artefato Gerado

```text
features_corr.pkl
```

---

## 06_modeling.ipynb

### Objetivos

Treinar e comparar diferentes algoritmos de classificação para previsão de inadimplência.

### Modelos Avaliados

* Logistic Regression;
* Random Forest;
* LightGBM.

### Métricas Utilizadas

* AUC (Area Under the Curve);
* KS (Kolmogorov-Smirnov);
* Gini;
* PSI (Population Stability Index).

### Estratégia de Validação

* Validação Cruzada Estratificada;
* OOS (Out of Sample);
* OOT (Out of Time).

### Modelo Selecionado

```text
LightGBM
```

Critério principal de seleção:

```text
Maior KS na base OOT
```

### Artefatos Gerados

```text
best_model.pkl
best_model_features.pkl
```

---

## 07_credit_policy.ipynb

### Objetivos

Transformar as probabilidades estimadas pelo modelo em uma política de crédito operacional.

### Etapas

* Escoragem da base de treino;
* Escoragem da base OOT;
* Escoragem da base de submissão;
* Construção dos ratings de risco;
* Definição das regras de negócio;
* Geração do arquivo final de submissão.

### Política de Crédito Proposta

| Rating | Ação                 |
| ------ | -------------------- |
| A      | Aprovação Automática |
| B      | Aprovação Automática |
| C      | Análise Simplificada |
| D      | Análise Manual       |
| E      | Reprovação           |

### Saída Final

```text
submissao_case.csv
```

Formato:

| Coluna                      |
| --------------------------- |
| id_cliente                  |
| probabilidade_inadimplencia |

---

# Modelo Final

Após a comparação dos algoritmos Logistic Regression, Random Forest e LightGBM, o modelo selecionado foi o **LightGBM**, apresentando o melhor desempenho na métrica KS durante a validação temporal (OOT).

O modelo demonstrou maior capacidade de separação entre clientes adimplentes e inadimplentes, mantendo estabilidade entre as bases de desenvolvimento e validação.

---

# Política de Crédito

A partir das probabilidades estimadas pelo modelo foi construída uma política de crédito baseada em ratings de risco.

Os clientes foram segmentados em cinco faixas de risco (A até E), definidas a partir da distribuição dos scores observados na base de desenvolvimento.

| Rating | Ação Recomendada     |
| ------ | -------------------- |
| A      | Aprovação Automática |
| B      | Aprovação Automática |
| C      | Análise Simplificada |
| D      | Análise Manual       |
| E      | Reprovação           |

A análise da inadimplência observada demonstrou crescimento monotônico entre os ratings, indicando que o modelo foi capaz de ordenar corretamente os clientes conforme seu nível de risco.

---

# Tecnologias Utilizadas

* Python 3.13
* Pandas
* NumPy
* SciPy
* Scikit-Learn
* LightGBM
* StatsModels
* Imbalanced-Learn
* Matplotlib
* Seaborn
* PyArrow

---

# Resultado Final

O projeto entrega:

* Pipeline completa de modelagem de risco de crédito;
* Construção automatizada da população de modelagem;
* Definição e validação da variável alvo;
* Processo padronizado de Feature Engineering;
* Seleção de variáveis baseada em VIF e correlação;
* Modelo preditivo de inadimplência utilizando LightGBM;
* Política de crédito baseada em ratings de risco;
* Processo automatizado de escoragem para novas solicitações;
* Arquivo final de submissão contendo as probabilidades estimadas de inadimplência.

### Arquivo Final

```text
outputs/submissions/submissao_case.csv
```

### Colunas

```text
id_cliente
probabilidade_inadimplencia
```

---

# Autor

| Autor                             | GitHub           | LinkedIn         | Email                                                         |
| --------------------------------- | ---------------- | ---------------- | ------------------------------------------------------------- |
| Renan Douglas Floriano Scavazzini | @RenanScavazzini | renan-scavazzini | [renanscavazzini@gmail.com](mailto:renanscavazzini@gmail.com) |

---

Desenvolvido como solução para o desafio técnico de Ciência de Dados, contemplando todo o ciclo de modelagem de risco de crédito, desde a construção da população até a definição de uma política de crédito baseada em Machine Learning.