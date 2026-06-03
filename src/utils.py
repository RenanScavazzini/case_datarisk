"""
Descrição:
    Funções utilitárias para métricas de avaliação e operações numéricas seguras.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score


def safe_ratio(num, den):
    """
    Descrição:
        Calcula a razão `num / den` de forma segura, tratando divisões
        por zero e valores infinitos, retornando NaN nesses casos.

    Parâmetros:
        num: Série ou array numérico numérico (numerador).
        den: Série ou array numérico (denominador).

    Retorno:
        pd.Series: Resultado da divisão com tratamento de valores inválidos.

    Referências:
        ---
    """
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return (num / den).replace([np.inf, -np.inf], np.nan)


def gini_score(y_true, y_score):
    """
    Descrição:
        Calcula o Gini a partir do AUC (Gini = 2*AUC - 1).

    Parâmetros:
        y_true: Série de rótulos verdadeiros (0/1).
        y_score: Pontuações previstas (probabilidades).

    Retorno:
        float: Valor do Gini.

    Referências:
        ---
    """
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def ks_statistic(y_true, y_score):
    """
    Descrição:
        Calcula a estatística KS (Kolmogorov-Smirnov) entre as distribuições
        acumuladas de bons e maus ordenadas pela pontuação prevista.

    Parâmetros:
        y_true: Série de rótulos verdadeiros (0/1).
        y_score: Pontuações previstas (probabilidades).

    Retorno:
        float: Valor da estatística KS.

    Referências:
        ---
    """
    data = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    data = data.sort_values("y_score", ascending=False)
    data["cum_good"] = (~data["y_true"]).cumsum() / (~data["y_true"]).sum()
    data["cum_bad"] = data["y_true"].cumsum() / data["y_true"].sum()
    return float((data["cum_bad"] - data["cum_good"]).abs().max())


def evaluate_classification(y_true, y_score):
    """
    Descrição:
        Retorna um dicionário com métricas básicas de classificação
        (ROC AUC, Gini, KS e acurácia) dado `y_true` e `y_score`.

    Parâmetros:
        y_true: Série de rótulos verdadeiros (0/1).
        y_score: Pontuações previstas (probabilidades).

    Retorno:
        dict: Dicionário com chaves `roc_auc`, `gini`, `ks` e `accuracy`.

    Referências:
        ---
    """
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "gini": gini_score(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
    }
