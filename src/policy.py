"""
Descrição:
    Módulo responsável pela definição da política de rating e geração
    de relatórios comparativos para apoio à tomada de decisão de crédito.

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
    """
    Descrição:
        Define os intervalos de rating com base nos quantis do score de
        inadimplência da base de treino.

    Parâmetros:
        train_scored (pd.DataFrame): Base de treino contendo a coluna de score.
        score_col (str): Nome da coluna com a probabilidade de inadimplência.

    Retorno:
        tuple: Tupla contendo os bins de rating e os respectivos rótulos.

    Referências:
        ---
    """

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
    """
    Descrição:
        Aplica a política de rating em uma base pontuada e atribui a ação
        correspondente para cada faixa de score.

    Parâmetros:
        df (pd.DataFrame): Base contendo a coluna de score.
        rating_bins (array-like): Limites utilizados para classificação em rating.
        rating_labels (list): Rótulos das faixas de rating.
        score_col (str): Nome da coluna com a probabilidade de inadimplência.

    Retorno:
        pd.DataFrame: Base com as colunas `rating` e `acao` atribuídas.

    Referências:
        ---
    """

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
    """
    Descrição:
        Consolida um relatório por faixa de rating com volume de clientes,
        quantidade de maus, inadimplência e faixa de score observada.

    Parâmetros:
        df (pd.DataFrame): Base com rating, ação, target e score calculados.
        target (str): Nome da coluna alvo utilizada no cálculo do relatório.
        score_col (str): Nome da coluna com a probabilidade de inadimplência.

    Retorno:
        pd.DataFrame: Relatório agregado da política por rating e ação.

    Referências:
        ---
    """

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
    """
    Descrição:
        Compara a inadimplência observada por rating entre os relatórios de
        treino e out-of-time.

    Parâmetros:
        train_report (pd.DataFrame): Relatório de política calculado na base de treino.
        oot_report (pd.DataFrame): Relatório de política calculado na base out-of-time.

    Retorno:
        pd.DataFrame: Comparativo de inadimplência por rating entre as bases.

    Referências:
        ---
    """

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