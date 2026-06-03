# Credit Risk Case

## Objetivo

Desenvolver um modelo preditivo para estimar a probabilidade de inadimplência de novos contratos de crédito presentes na base `base_submissao.parquet`, utilizando informações históricas de empréstimos, pagamentos e dados cadastrais dos clientes.

Além da modelagem, será proposta uma política de crédito baseada nas probabilidades estimadas.

---

# Estrutura do Projeto

```text
case/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_population.ipynb
│   ├── 02_target_definition.ipynb
│   ├── 03_feature_store.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modeling.ipynb
│   └── 07_credit_policy.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── population.py
│   ├── target.py
│   ├── feature_store.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── policy.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── submissions/
│
├── requirements.txt
│
└── README.md
```

---

# Notebooks

## 01_population.ipynb

Objetivos:

* Leitura das bases
* Análise de qualidade
* Integridade referencial
* Relacionamentos entre tabelas
* Definição da população de modelagem
* Definição da população de score

Saída:

```text
population_train.parquet
population_score.parquet
```

---

## 02_target_definition.ipynb

Objetivos:

* Construção de possíveis definições de inadimplência
* Comparação entre targets
* Escolha da target final

Targets avaliadas:

* FPD
* EVER30MOB03
* OVER60MOB06

Saída:

```text
population_target.parquet
```

---

## 03_feature_store.ipynb

Objetivos:

Construção de atributos históricos reutilizáveis.

Blocos:

* Perfil cadastral
* Histórico de crédito
* Histórico de pagamentos
* Recência
* Frequência
* Monetárias

Saída:

```text
feature_store.parquet
```

---

## 04_eda.ipynb

Objetivos:

* Análise exploratória
* Missing values
* Distribuições
* Bad Rate
* Correlações

---

## 05_feature_engineering.ipynb

Objetivos:

* Transformações
* Encoding
* Tratamento de outliers
* Criação de variáveis derivadas

Saída:

```text
model_dataset.parquet
```

---

## 06_modeling.ipynb

Objetivos:

* Logistic Regression
* LightGBM
* Avaliação

Métricas:

* ROC AUC
* KS
* Gini
* Precision Recall

Saída:

```text
final_model.pkl
```

---

## 07_credit_policy.ipynb

Objetivos:

* Construção da política de crédito
* Segmentação por risco
* Definição de faixas de aprovação

Saída:

```text
submissao_case.csv
```
