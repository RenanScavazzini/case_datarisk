"""
Descrição:
    Módulo com funções de avaliação, validação cruzada, estabilidade e
    visualização de desempenho de modelos de classificação.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 08/06/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "target"


def calculate_gini(auc):
    """
    Descrição:
        Calcula o coeficiente de Gini a partir do valor de AUC.

    Parâmetros:
        auc (float): Área sob a curva ROC.

    Retorno:
        float: Valor do coeficiente de Gini.

    Referências:
        ---
    """

    return (
        2 * auc
    ) - 1


def calculate_ks(
    y_true,
    y_score
):
    """
    Descrição:
        Calcula a estatística KS a partir da separação entre as distribuições
        de score das classes good e bad.

    Parâmetros:
        y_true (array-like): Vetor com os rótulos verdadeiros (0/1).
        y_score (array-like): Vetor com os scores previstos.

    Retorno:
        float: Valor da estatística KS.

    Referências:
        scipy.stats.ks_2samp documentation
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    good = y_score[y_true == 0]

    bad = y_score[y_true == 1]

    ks = ks_2samp(
        bad,
        good
    ).statistic

    return ks


def calculate_psi(
    expected,
    actual,
    bins=10
):
    """
    Descrição:
        Calcula o Population Stability Index (PSI) entre uma distribuição
        esperada e uma distribuição observada.

    Parâmetros:
        expected (array-like): Distribuição de referência.
        actual (array-like): Distribuição observada para comparação.
        bins (int): Quantidade de faixas utilizadas no cálculo.

    Retorno:
        float: Valor do PSI.

    Referências:
        ---
    """

    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 2:
        return 0.0

    expected_bins = pd.cut(
        expected,
        bins=breakpoints,
        include_lowest=True,
        duplicates="drop"
    )

    actual_bins = pd.cut(
        actual,
        bins=breakpoints,
        include_lowest=True,
        duplicates="drop"
    )

    expected_counts = (
        pd.Series(expected_bins)
        .value_counts()
        .sort_index()
    )

    actual_counts = (
        pd.Series(actual_bins)
        .value_counts()
        .sort_index()
        .reindex(
            expected_counts.index,
            fill_value=0
        )
    )

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct = actual_pct.replace(0, 0.0001)

    psi = np.sum(
        (actual_pct - expected_pct)
        *
        np.log(
            actual_pct / expected_pct
        )
    )

    return float(psi)


def evaluate_predictions(
    y_true,
    y_score
):
    """
    Descrição:
        Calcula métricas de avaliação de predição, incluindo AUC, KS e Gini.

    Parâmetros:
        y_true (array-like): Vetor com os rótulos verdadeiros (0/1).
        y_score (array-like): Vetor com os scores previstos.

    Retorno:
        dict: Dicionário com as métricas `AUC`, `KS` e `Gini`.

    Referências:
        ---
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    auc = roc_auc_score(
        y_true,
        y_score
    )

    ks = calculate_ks(
        y_true,
        y_score
    )

    gini = calculate_gini(
        auc
    )

    return {
        "AUC": auc,
        "KS": ks,
        "Gini": gini
    }


def cross_validation_scores(
    model,
    X,
    y
):
    """
    Descrição:
        Executa validação cruzada estratificada e retorna métricas calculadas
        sobre os scores out-of-fold.

    Parâmetros:
        model: Estimador compatível com scikit-learn.
        X: Matriz de variáveis explicativas.
        y: Vetor da variável alvo.

    Retorno:
        tuple: Métricas de validação cruzada e scores out-of-fold.

    Referências:
        StratifiedKFold e cross_val_predict - scikit-learn documentation
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_score = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    metrics = evaluate_predictions(
        y,
        oof_score
    )

    return metrics, oof_score


def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_oos,
    y_oos,
    X_oot,
    y_oot
):
    """
    Descrição:
        Treina o modelo, avalia seu desempenho em treino via validação cruzada
        e calcula métricas e estabilidade nas bases OOS e OOT.

    Parâmetros:
        model: Estimador compatível com scikit-learn.
        X_train: Matriz de treino.
        y_train: Vetor alvo de treino.
        X_oos: Matriz out-of-sample.
        y_oos: Vetor alvo out-of-sample.
        X_oot: Matriz out-of-time.
        y_oot: Vetor alvo out-of-time.

    Retorno:
        tuple: Modelo treinado, dicionário de resultados e dicionário de scores.

    Referências:
        ---
    """

    cv_metrics, oof_score = (
        cross_validation_scores(
            model,
            X_train,
            y_train
        )
    )

    model.fit(
        X_train,
        y_train
    )

    oos_score = model.predict_proba(
        X_oos
    )[:, 1]

    oot_score = model.predict_proba(
        X_oot
    )[:, 1]

    oos_metrics = evaluate_predictions(
        y_oos,
        oos_score
    )

    oot_metrics = evaluate_predictions(
        y_oot,
        oot_score
    )

    psi_oos = calculate_psi(
        oof_score,
        oos_score
    )

    psi_oot = calculate_psi(
        oof_score,
        oot_score
    )

    results = {

        "cv": cv_metrics,

        "oos": {
            **oos_metrics,
            "PSI": psi_oos
        },

        "oot": {
            **oot_metrics,
            "PSI": psi_oot
        }

    }

    scores = {
        "oof_score": oof_score,
        "oos_score": oos_score,
        "oot_score": oot_score
    }

    return model, results, scores


def plot_roc_comparison(
    y_true,
    scores_dict
):
    """
    Descrição:
        Plota a curva ROC comparando múltiplos modelos a partir de seus scores.

    Parâmetros:
        y_true (array-like): Vetor com os rótulos verdadeiros (0/1).
        scores_dict (dict): Dicionário com nome do modelo e vetor de scores.

    Retorno:
        ---

    Referências:
        matplotlib documentation
    """

    y_true = np.asarray(y_true)

    plt.figure(figsize=(6, 5))

    for model_name, y_score in scores_dict.items():

        y_score = np.asarray(y_score)

        auc = roc_auc_score(
            y_true,
            y_score
        )

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} ({auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "Comparação ROC"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


def plot_score_distribution(
    y_true,
    y_score,
    model_name,
    ax,
    bins=np.arange(0, 1.1, 0.1)
):
    """
    Descrição:
        Plota a distribuição percentual dos scores para as classes good e bad
        em faixas definidas de probabilidade.

    Parâmetros:
        y_true (array-like): Vetor com os rótulos verdadeiros (0/1).
        y_score (array-like): Vetor com os scores previstos.
        model_name (str): Nome do modelo exibido no gráfico.
        ax: Eixo matplotlib utilizado para desenhar o gráfico.
        bins (array-like): Faixas de score utilizadas na distribuição.

    Retorno:
        ---

    Referências:
        matplotlib documentation
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    good_scores = y_score[
        y_true == 0
    ]

    bad_scores = y_score[
        y_true == 1
    ]

    good_counts, _ = np.histogram(
        good_scores,
        bins=bins
    )

    bad_counts, _ = np.histogram(
        bad_scores,
        bins=bins
    )

    if good_counts.sum() > 0:
        good_pct = (
            good_counts / good_counts.sum()
        ) * 100
    else:
        good_pct = np.zeros(
            len(bins) - 1
        )

    if bad_counts.sum() > 0:
        bad_pct = (
            bad_counts / bad_counts.sum()
        ) * 100
    else:
        bad_pct = np.zeros(
            len(bins) - 1
        )

    labels = [
        f"{i:.1f}-{j:.1f}"
        for i, j in zip(
            bins[:-1],
            bins[1:]
        )
    ]

    x = np.arange(len(labels))

    width = 0.4

    ax.bar(
        x - width/2,
        good_pct,
        width=width,
        color="#1f77b4",
        label="Good (0)"
    )

    ax.bar(
        x + width/2,
        bad_pct,
        width=width,
        color="#d62728",
        label="Bad (1)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        fontsize=8
    )

    ax.set_ylim(0, 50)

    ax.set_title(
        model_name,
        fontsize=10
    )

    ax.set_ylabel("%")

    ax.grid(
        axis="y",
        alpha=0.3
    )
