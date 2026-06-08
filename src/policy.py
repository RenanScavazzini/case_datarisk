import numpy as np
import pandas as pd


RATING_ACTION = {

    "A": "Aprovação Automática",

    "B": "Aprovação Automática",

    "C": "Análise Simplificada",

    "D": "Análise Manual",

    "E": "Reprovação"

}


def build_rating_policy(
    train_scored,
    score_col="probabilidade_inadimplencia"
):

    rating_bins = (
        train_scored[score_col]
        .quantile(
            [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        .values
    )

    rating_bins = np.unique(
        rating_bins
    )

    rating_labels = [
        "A",
        "B",
        "C",
        "D",
        "E"
    ]

    return (
        rating_bins,
        rating_labels
    )


def apply_rating_policy(
    df,
    rating_bins,
    rating_labels,
    score_col="probabilidade_inadimplencia"
):

    df = df.copy()

    df["rating"] = pd.cut(
        df[score_col],
        bins=rating_bins,
        labels=rating_labels,
        include_lowest=True
    )

    df["acao"] = (
        df["rating"]
        .astype(str)
        .map(RATING_ACTION)
    )

    return df


def policy_report(
    df,
    target,
    score_col="probabilidade_inadimplencia"
):

    report = (
        df
        .groupby(
            ["rating", "acao"],
            observed=True
        )
        .agg(
            clientes=(
                target,
                "count"
            ),
            bads=(
                target,
                "sum"
            ),
            inadimplencia=(
                target,
                "mean"
            ),
            score_min=(
                score_col,
                "min"
            ),
            score_max=(
                score_col,
                "max"
            )
        )
        .reset_index()
    )

    report["inadimplencia"] *= 100

    return report


def compare_policy(
    train_report,
    oot_report
):

    comparison = (
        train_report[
            [
                "rating",
                "inadimplencia"
            ]
        ]
        .merge(
            oot_report[
                [
                    "rating",
                    "inadimplencia"
                ]
            ],
            on="rating",
            suffixes=(
                "_train",
                "_oot"
            )
        )
    )

    return comparison