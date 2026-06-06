"""
Descrição:
    Funções auxiliares para Análise Exploratória de Dados (EDA).

Autor:
    Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def dataset_overview(df: pd.DataFrame):
    """
    Visão geral do dataset.
    """

    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "% missing": (df.isna().mean() * 100).round(2),
        "unique": df.nunique()
    })

    return overview.sort_values("% missing", ascending=False)


def target_distribution(df: pd.DataFrame):
    """
    Distribuição do target.
    """

    total = len(df)
    bad = df["target"].sum()
    good = total - bad

    return pd.DataFrame({
        "total": [total],
        "good": [good],
        "bad": [bad],
        "% bad": [round((bad / total) * 100, 2)]
    })


def target_by_safra(df: pd.DataFrame):

    result = (
        df.groupby("safra_mes")["target"]
        .agg(["count", "sum"])
        .reset_index()
    )

    result.columns = ["safra", "total", "bad"]

    result["good"] = result["total"] - result["bad"]

    result["% bad"] = (
        result["bad"] /
        result["total"] *
        100
    ).round(2)

    return result


def missing_report(df: pd.DataFrame):

    result = pd.DataFrame({
        "variable": df.columns,
        "missing": df.isna().sum(),
        "% missing": (df.isna().mean() * 100).round(2)
    })

    return (
        result
        .sort_values("% missing", ascending=False)
        .reset_index(drop=True)
    )


def numeric_summary(df: pd.DataFrame):

    return (
        df.describe()
        .T
        .sort_index()
    )


def categorical_summary(df: pd.DataFrame):

    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    result = []

    for col in cat_cols:

        result.append({
            "variable": col,
            "unique": df[col].nunique(),
            "missing": df[col].isna().sum()
        })

    return pd.DataFrame(result)


def target_rate_by_category(df, col):

    result = (
        df.groupby(col)["target"]
        .agg(["count", "mean"])
        .reset_index()
    )

    result.columns = [
        col,
        "total",
        "bad_rate"
    ]

    result["bad_rate"] = (
        result["bad_rate"] * 100
    ).round(2)

    return result.sort_values(
        "bad_rate",
        ascending=False
    )


def correlation_matrix(df):

    numeric = df.select_dtypes(
        include=["number"]
    )

    return numeric.corr()


def plot_target_by_safra(df):

    summary = target_by_safra(df)

    plt.figure(figsize=(12, 5))

    plt.plot(
        summary["safra"],
        summary["% bad"],
        marker="o"
    )

    plt.xticks(rotation=45)

    plt.title(
        "Taxa de Bad por Safra"
    )

    plt.tight_layout()

    plt.show()


def plot_missing(df):

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 6))

    missing.plot.bar()

    plt.title(
        "Top 20 Variáveis com Missing"
    )

    plt.tight_layout()

    plt.show()