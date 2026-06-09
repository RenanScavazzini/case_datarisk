"""
Descrição:
    Módulo com funções para seleção de variáveis com base em multicolinearidade
    e correlação entre atributos.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 08/06/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter

def calculate_vif(df):
    """
    Descrição:
        Calcula o Variance Inflation Factor (VIF) para cada variável do dataset.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo apenas variáveis numéricas.

    Retorno:
        pd.DataFrame: Tabela com as variáveis e seus respectivos valores de VIF.

    Referências:
        statsmodels variance_inflation_factor documentation
    """

    result = pd.DataFrame({
        "variavel": df.columns,
        "vif": [
            variance_inflation_factor(
                df.values,
                i
            )
            for i in range(df.shape[1])
        ]
    })

    return result.sort_values(
        "vif",
        ascending=False
    ).reset_index(drop=True)


def vif_selection(
    df,
    threshold=10
):
    """
    Descrição:
        Realiza seleção iterativa de variáveis removendo o atributo com maior
        VIF até que todos os valores fiquem abaixo do limite definido.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as variáveis candidatas.
        threshold (float): Limite máximo aceitável para o VIF.

    Retorno:
        tuple: Lista final de variáveis selecionadas e tabela de VIF da última iteração.

    Referências:
        ---
    """

    features = list(df.columns)

    iteration = 1

    while True:

        vif_df = calculate_vif(
            df[features]
        )

        print("\n" + "=" * 80)
        print(f"Iteração {iteration}")
        print("=" * 80)

        display(vif_df)

        max_vif = vif_df["vif"].max()

        if max_vif <= threshold:

            print(
                f"\nTodos os VIFs estão abaixo de {threshold}."
            )

            break

        remove_var = vif_df.iloc[0]["variavel"]

        print(
            f"\nRemovendo: {remove_var} "
            f"(VIF={max_vif:.2f})"
        )

        features.remove(remove_var)

        iteration += 1

    return features, vif_df


def correlation_selection(
    df,
    threshold=0.80
):
    """
    Descrição:
        Remove iterativamente variáveis com alta correlação absoluta até que
        todos os pares remanescentes fiquem abaixo do limite definido.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as variáveis candidatas.
        threshold (float): Limite máximo de correlação absoluta entre pares.

    Retorno:
        list: Lista final de variáveis selecionadas.

    Referências:
        ---
    """

    features = list(df.columns)

    iteration = 1

    while True:

        corr_matrix = (
            df[features]
            .corr()
            .abs()
        )

        pairs = []

        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):

                corr = corr_matrix.iloc[i, j]

                if corr >= threshold:

                    pairs.append({
                        "var1": cols[i],
                        "var2": cols[j],
                        "corr": corr
                    })

        if len(pairs) == 0:

            print(
                f"\nTodas as correlações estão abaixo de {threshold}"
            )

            break

        pair_df = (
            pd.DataFrame(pairs)
            .sort_values("corr", ascending=False)
        )

        counts = Counter()

        for _, row in pair_df.iterrows():

            counts[row["var1"]] += 1
            counts[row["var2"]] += 1

        remove_var = counts.most_common(1)[0][0]

        print("\n" + "=" * 80)
        print(f"Iteração {iteration}")

        print("\nPares encontrados:")

        display(
            pair_df.head(10)
        )

        print("\nFrequência:")

        display(
            pd.DataFrame(
                counts.items(),
                columns=["variavel", "frequencia"]
            ).sort_values(
                "frequencia",
                ascending=False
            )
        )

        print(f"\nRemovendo: {remove_var}")

        features.remove(remove_var)

        iteration += 1

    return features