import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    roc_auc_score
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

    return (
        2 * auc
    ) - 1


def calculate_ks(
    y_true,
    y_score
):

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

    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, bins + 1)
    )

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

    plt.figure(figsize=(6, 5))

    for model_name, y_score in scores_dict.items():

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

    df_plot = pd.DataFrame({
        "target": y_true,
        "score": y_score
    })

    good = df_plot[
        df_plot["target"] == 0
    ]

    bad = df_plot[
        df_plot["target"] == 1
    ]

    good_pct = (
        pd.cut(
            good["score"],
            bins=bins,
            include_lowest=True
        )
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    bad_pct = (
        pd.cut(
            bad["score"],
            bins=bins,
            include_lowest=True
        )
        .value_counts(normalize=True)
        .sort_index()
        * 100
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
        good_pct.values,
        width=width,
        color="#1f77b4",
        label="Good (0)"
    )

    ax.bar(
        x + width/2,
        bad_pct.values,
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