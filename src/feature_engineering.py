"""
Descrição:
    Módulo com funções de preparação e engenharia de variáveis para
    o dataset de modelagem (scaling, encoding e tratamento de missing).

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler


def feature_engineering(
    df: pd.DataFrame,
    dicionario_imputacao: dict,
    dicionario_dominio: dict,
    woe_dictionary: dict,
    normalization_dictionary: dict
):
    """
    Replica todas as transformações
    definidas no EDA.
    """

    df = df.copy()

    # ==================================================
    # BINÁRIAS
    # ==================================================

    df["sexo_masculino"] = (
        df["sexo"]
        .map({"M": 1, "F": 0})
    )

    df["possui_carro"] = (
        df["possui_carro"]
        .map({"Y": 1, "N": 0})
    )

    df["possui_imovel"] = (
        df["possui_imovel"]
        .map({"Y": 1, "N": 0})
    )

    df.drop(
        columns=["sexo"],
        inplace=True
    )

    # ==================================================
    # DATAS
    # ==================================================

    df["data_solicitacao"] = pd.to_datetime(
        df["data_solicitacao"]
    )

    df["data_nascimento"] = pd.to_datetime(
        df["data_nascimento"]
    )

    # ==================================================
    # IDADE
    # ==================================================

    df["idade"] = (
        (
            df["data_solicitacao"]
            - df["data_nascimento"]
        ).dt.days
        / 365
    ).round()

    df.drop(
        columns=[
            "data_solicitacao",
            "data_nascimento"
        ],
        inplace=True
    )

    # ==================================================
    # REMOVE VARIÁVEIS
    # ==================================================

    vars_remover = [
        "max_delay",
        "ever_30",
        "ever_45",
        "ever_60",
        "ever_90",
        "ocupacao"
    ]

    df.drop(
        columns=vars_remover,
        errors="ignore",
        inplace=True
    )

    # ==================================================
    # IMPUTAÇÃO
    # ==================================================

    df = df.fillna(
        dicionario_imputacao
    )

    # ==================================================
    # DOMÍNIO NUMÉRICO
    # ==================================================

    for col, limites in (
        dicionario_dominio[
            "numerico"
        ].items()
    ):

        if col in df.columns:

            df[col] = df[col].clip(
                lower=limites["inferior"],
                upper=limites["superior"]
            )

    # ==================================================
    # DOMÍNIO CATEGÓRICO
    # ==================================================

    for col, categorias in (
        dicionario_dominio[
            "categorico"
        ].items()
    ):

        if col not in df.columns:
            continue

        df.loc[
            ~df[col].isin(categorias),
            col
        ] = pd.NA

        df[col] = df[col].fillna(
            dicionario_imputacao[col]
        )

    # ==================================================
    # WOE
    # ==================================================

    for col, mapping in (
        woe_dictionary.items()
    ):

        new_col = f"{col}_woe"

        df[new_col] = (
            df[col]
            .map(mapping)
        )

        df.drop(
            columns=[col],
            inplace=True
        )

    # ==================================================
    # NORMALIZAÇÃO
    # ==================================================

    for col, params in (
        normalization_dictionary.items()
    ):

        if col not in df.columns:
            continue

        minimo = params["min"]
        maximo = params["max"]

        if minimo == maximo:

            df[col] = 0

        else:

            df[col] = (
                (
                    df[col]
                    - minimo
                )
                /
                (
                    maximo
                    - minimo
                )
            )

    return df


def apply_rus(
    df: pd.DataFrame,
    target_col: str = "target",
    sampling_strategy: float = 1.0,
    random_state: int = 42,
    verbose: bool = True
):
    """
    Aplica Random Under Sampling (RUS)
    na base de treino.

    Parameters
    ----------
    df : pd.DataFrame
        Base contendo target.

    target_col : str
        Nome da variável alvo.

    sampling_strategy : float
        Proporção desejada entre minoritária
        e majoritária após o balanceamento.

    random_state : int
        Seed de reprodutibilidade.

    verbose : bool
        Exibe distribuição antes e depois.

    Returns
    -------
    pd.DataFrame
        Base balanceada.
    """
    def target_distribution(df):

        total = len(df)

        bad = df["target"].sum()

        good = total - bad

        return pd.DataFrame({
            "total": [total],
            "good": [good],
            "bad": [bad],
            "% bad": [
                round(
                    bad / total * 100,
                    2
                )
            ]
        })

    X = df.drop(
        columns=[target_col]
    )

    y = df[target_col]

    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    X_rus, y_rus = rus.fit_resample(
        X,
        y
    )

    df_rus = pd.concat(
        [
            X_rus,
            y_rus
        ],
        axis=1
    )

    if verbose:

        print("Antes do RUS")

        display(
            target_distribution(
                df
            )
        )

        print("Depois do RUS")

        display(
            target_distribution(
                df_rus
            )
        )

    return df_rus