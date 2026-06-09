"""
Descrição:
    Módulo com funções auxiliares para Análise Exploratória de Dados (EDA),
    preparação de bases, tratamento de domínio, WOE, normalização e cálculo
    de métricas de informação de variáveis.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 08/06/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def dataset_overview(df: pd.DataFrame):
    """
    Descrição:
        Gera uma visão geral do dataset com tipo, volume de missing e
        cardinalidade por coluna.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser resumido.

    Retorno:
        pd.DataFrame: Visão consolidada das colunas do dataset.

    Referências:
        ---
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
    Descrição:
        Calcula a distribuição geral do target em termos de total, good, bad
        e percentual de maus.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna `target`.

    Retorno:
        pd.DataFrame: Tabela resumo da distribuição do target.

    Referências:
        ---
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
    """
    Descrição:
        Calcula a distribuição do target por safra mensal.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas `safra_mes` e `target`.

    Retorno:
        pd.DataFrame: Tabela com total, good, bad e `% bad` por safra.

    Referências:
        ---
    """

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
    """
    Descrição:
        Gera um relatório de valores ausentes por variável.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser analisado.

    Retorno:
        pd.DataFrame: Relatório com quantidade e percentual de missing por variável.

    Referências:
        ---
    """

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


def numeric_summary(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Calcula um resumo estatístico das variáveis numéricas, inteiras e
        binárias presentes na base.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        pd.DataFrame: Estatísticas descritivas das variáveis selecionadas.

    Referências:
        ---
    """

    numeric_cols = (
        tipos_var["numeric"]
        + tipos_var["integer"]
        + tipos_var["binary"]
    )

    numeric_cols = [
        col
        for col in numeric_cols
        if col in df.columns
    ]

    if len(numeric_cols) == 0:
        return pd.DataFrame()

    result = (
        df[numeric_cols]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    return result.sort_values(
        "variable"
    ).reset_index(drop=True)


def categorical_summary(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Resume as variáveis categóricas em termos de cardinalidade, missing e moda.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        pd.DataFrame: Resumo das variáveis categóricas disponíveis.

    Referências:
        ---
    """

    result = []

    for col in tipos_var["categorical"]:

        if col not in df.columns:
            continue

        moda = df[col].mode(dropna=True)

        moda = (
            moda.iloc[0]
            if len(moda) > 0
            else pd.NA
        )

        result.append({
            "variable": col,
            "unique": df[col].nunique(dropna=True),
            "missing": df[col].isna().sum(),
            "% missing": round(
                df[col].isna().mean() * 100,
                2
            ),
            "mode": moda
        })

    return (
        pd.DataFrame(result)
        .sort_values("variable")
        .reset_index(drop=True)
    )


def target_rate_by_category(df, col):
    """
    Descrição:
        Calcula a taxa de target por categoria de uma variável.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna `target`.
        col (str): Nome da variável categórica a ser analisada.

    Retorno:
        pd.DataFrame: Tabela com total e taxa de bad por categoria.

    Referências:
        ---
    """

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
    """
    Descrição:
        Calcula a matriz de correlação entre variáveis numéricas do dataset.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser analisado.

    Retorno:
        pd.DataFrame: Matriz de correlação entre colunas numéricas.

    Referências:
        ---
    """

    numeric = df.select_dtypes(
        include=["number"]
    )

    return numeric.corr()


def plot_target_by_safra(df):
    """
    Descrição:
        Plota a evolução da taxa de bad por safra.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas `safra_mes` e `target`.

    Retorno:
        ---

    Referências:
        matplotlib documentation
    """

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
    """
    Descrição:
        Plota as 20 variáveis com maior percentual de valores ausentes.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser analisado.

    Retorno:
        ---

    Referências:
        matplotlib documentation
    """

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


def prepare_dataset(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Padroniza tipos de dados e aplica tratamentos iniciais nas variáveis
        binárias, categóricas, numéricas, inteiras e target.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        tuple: DataFrame tratado e dicionário de tipos atualizado.

    Referências:
        ---
    """

    df = df.copy()

    # binárias textuais
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

    if "sexo" in tipos_var["binary"]:
        tipos_var["binary"].remove("sexo")

    if "sexo_masculino" not in tipos_var["binary"]:
        tipos_var["binary"].append("sexo_masculino")

    # datas
    for col in [
        "data_solicitacao",
        "data_nascimento"
    ]:

        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col]
            ).astype("datetime64[us]")

    # string
    for col in (
        tipos_var["key"]
        + tipos_var["categorical"]
    ):
        if col in df.columns:
            df[col] = df[col].astype("string")

    # binary
    for col in tipos_var["binary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("Int64")

    # integer
    for col in tipos_var["integer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("Int64")

    # numeric
    for col in tipos_var["numeric"]:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    # target
    for col in tipos_var["target"]:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    return df, tipos_var


def create_age_feature(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Cria a variável `idade` a partir das datas de solicitação e nascimento.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo colunas de data.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        tuple: DataFrame com a variável `idade` e dicionário de tipos atualizado.

    Referências:
        ---
    """

    df = df.copy()

    df["idade"] = (
        (
            df["data_solicitacao"]
            - df["data_nascimento"]
        ).dt.days
        / 365
    ).round(0)

    df["idade"] = (
        df["idade"]
        .astype("Int64")
    )

    df.drop(
        columns=[
            "data_solicitacao",
            "data_nascimento"
        ],
        inplace=True
    )

    tipos_var["integer"].append(
        "idade"
    )
    tipos_var["datetime"] = []

    return df, tipos_var


def remove_variables(
    df: pd.DataFrame,
    tipos_var: dict,
    variables: list
):
    """
    Descrição:
        Remove variáveis do dataset e atualiza o dicionário de tipos.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.
        variables (list): Lista de variáveis a serem removidas.

    Retorno:
        tuple: DataFrame sem as variáveis removidas e dicionário de tipos atualizado.

    Referências:
        ---
    """

    df = df.copy()

    df.drop(
        columns=variables,
        errors="ignore",
        inplace=True
    )

    for var in variables:

        for tipo in tipos_var:

            if var in tipos_var[tipo]:
                tipos_var[tipo].remove(var)

    return df, tipos_var


def build_imputation_dictionary(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Constrói o relatório de imputação e o dicionário de valores de
        preenchimento a partir da tipologia das variáveis.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        tuple: Relatório de imputação e dicionário de imputação por variável.

    Referências:
        ---
    """

    imputation_report = []
    dicionario_imputacao = {}

    # -------------------------------------------------------------------------
    # Variáveis numéricas
    # -------------------------------------------------------------------------

    for col in tipos_var["numeric"]:

        media = df[col].mean()
        mediana = df[col].median()

        moda = (
            df[col]
            .mode(dropna=True)
        )

        moda = (
            moda.iloc[0]
            if len(moda) > 0
            else pd.NA
        )

        imputation_report.append({
            "variavel": col,
            "tipo": "numeric",
            "media": round(media, 2),
            "mediana": round(mediana, 2),
            "moda": moda,
            "valor_imputacao": round(media, 2)
        })

        dicionario_imputacao[col] = media

    # -------------------------------------------------------------------------
    # Variáveis inteiras
    # -------------------------------------------------------------------------

    for col in tipos_var["integer"]:

        media = round(df[col].mean())
        mediana = round(df[col].median())

        moda = (
            df[col]
            .mode(dropna=True)
        )

        moda = (
            moda.iloc[0]
            if len(moda) > 0
            else pd.NA
        )

        imputation_report.append({
            "variavel": col,
            "tipo": "integer",
            "media": media,
            "mediana": mediana,
            "moda": moda,
            "valor_imputacao": media
        })

        dicionario_imputacao[col] = int(media)

    # -------------------------------------------------------------------------
    # Variáveis binárias
    # -------------------------------------------------------------------------

    for col in tipos_var["binary"]:

        moda = (
            df[col]
            .mode(dropna=True)
        )

        moda = (
            moda.iloc[0]
            if len(moda) > 0
            else pd.NA
        )

        imputation_report.append({
            "variavel": col,
            "tipo": "binary",
            "media": pd.NA,
            "mediana": pd.NA,
            "moda": moda,
            "valor_imputacao": moda
        })

        dicionario_imputacao[col] = moda

    # -------------------------------------------------------------------------
    # Variáveis categóricas
    # -------------------------------------------------------------------------

    for col in tipos_var["categorical"]:

        moda = (
            df[col]
            .mode(dropna=True)
        )

        moda = (
            moda.iloc[0]
            if len(moda) > 0
            else pd.NA
        )

        imputation_report.append({
            "variavel": col,
            "tipo": "categorical",
            "media": pd.NA,
            "mediana": pd.NA,
            "moda": moda,
            "valor_imputacao": moda
        })

        dicionario_imputacao[col] = moda

    # -------------------------------------------------------------------------
    # Relatório final
    # -------------------------------------------------------------------------

    imputation_report = (
        pd.DataFrame(imputation_report)
        .sort_values(
            ["tipo", "variavel"]
        )
        .reset_index(drop=True)
    )

    return (
        imputation_report,
        dicionario_imputacao
    )


def imputation_dictionary_to_df(
    dicionario_imputacao: dict
):
    """
    Descrição:
        Converte o dicionário de imputação em DataFrame para consulta tabular.

    Parâmetros:
        dicionario_imputacao (dict): Dicionário com valores de imputação por variável.

    Retorno:
        pd.DataFrame: Tabela com variáveis e respectivos valores de imputação.

    Referências:
        ---
    """
    return (
        pd.DataFrame({
            "variavel": dicionario_imputacao.keys(),
            "valor_imputacao": dicionario_imputacao.values()
        })
        .sort_values("variavel")
        .reset_index(drop=True)
    )


def apply_imputation(
    df: pd.DataFrame,
    dicionario_imputacao: dict
):
    """
    Descrição:
        Aplica imputação de valores ausentes conforme o dicionário informado.

    Parâmetros:
        df (pd.DataFrame): DataFrame com valores ausentes.
        dicionario_imputacao (dict): Dicionário com valores de imputação por variável.

    Retorno:
        pd.DataFrame: DataFrame após imputação dos valores ausentes.

    Referências:
        ---
    """

    before = df.isna().sum().sum()

    df = df.fillna(
        dicionario_imputacao
    )

    after = df.isna().sum().sum()

    print(
        f"Missings antes: {before:,}"
    )

    print(
        f"Missings depois: {after:,}"
    )

    return df


def target_rate_report(
    df,
    tipos_var
):
    """
    Descrição:
        Exibe relatórios de taxa de target para variáveis binárias e categóricas.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna `target`.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        ---

    Referências:
        ---
    """

    for col in (
        tipos_var["binary"]
        + tipos_var["categorical"]
    ):

        print(f"\n{'='*80}")
        print(col)

        display(
            target_rate_by_category(
                df,
                col
            )
        )


def build_numeric_domain(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Constrói o domínio das variáveis numéricas e inteiras utilizando
        percentis para definição de limites operacionais.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        tuple: Dicionário de domínio numérico e relatório dos limites calculados.

    Referências:
        ---
    """

    dominio_numerico = {}
    relatorio_numerico = []

    variaveis_numericas = (
        tipos_var["numeric"]
        + tipos_var["integer"]
    )

    total = len(df)

    for col in variaveis_numericas:

        if col not in df.columns:
            continue

        if col == "idade":

            limite_inferior = 0
            limite_superior = 120

        else:

            limite_inferior = df[col].quantile(0.01)
            limite_superior = df[col].quantile(0.99)

        qtd_abaixo = (
            df[col] < limite_inferior
        ).sum()

        qtd_acima = (
            df[col] > limite_superior
        ).sum()

        relatorio_numerico.append({
            "variavel": col,
            "limite_inferior": round(limite_inferior, 2),
            "limite_superior": round(limite_superior, 2),
            "qtd_abaixo": qtd_abaixo,
            "%_abaixo": round(
                qtd_abaixo / total * 100,
                2
            ),
            "qtd_acima": qtd_acima,
            "%_acima": round(
                qtd_acima / total * 100,
                2
            )
        })

        dominio_numerico[col] = {
            "inferior": limite_inferior,
            "superior": limite_superior
        }

    relatorio_numerico = (
        pd.DataFrame(relatorio_numerico)
        .sort_values("variavel")
        .reset_index(drop=True)
    )

    return (
        dominio_numerico,
        relatorio_numerico
    )


def build_categorical_domain(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Constrói o domínio permitido das variáveis categóricas com base nas
        categorias observadas na base de referência.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        tuple: Dicionário de domínio categórico e relatório de categorias por variável.

    Referências:
        ---
    """

    dominio_categorico = {}
    relatorio_categorico = []

    for col in tipos_var["categorical"]:

        if col not in df.columns:
            continue

        categorias = sorted(
            df[col]
            .dropna()
            .unique()
            .tolist()
        )

        dominio_categorico[col] = categorias

        relatorio_categorico.append({
            "variavel": col,
            "qtd_categorias": len(categorias)
        })

    relatorio_categorico = (
        pd.DataFrame(relatorio_categorico)
        .sort_values("variavel")
        .reset_index(drop=True)
    )

    return (
        dominio_categorico,
        relatorio_categorico
    )


def apply_domain(
    df: pd.DataFrame,
    dominio_numerico: dict,
    dominio_categorico: dict,
    dicionario_imputacao: dict
):
    """
    Descrição:
        Aplica o tratamento de domínio em variáveis numéricas e categóricas,
        incluindo winsorização, nulificação de categorias inválidas e imputação.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        dominio_numerico (dict): Limites inferior e superior por variável numérica.
        dominio_categorico (dict): Categorias válidas por variável categórica.
        dicionario_imputacao (dict): Valores de imputação por variável.

    Retorno:
        tuple: DataFrame tratado e diagnósticos antes e depois do tratamento.

    Referências:
        ---
    """

    df = df.copy()

    fora_dominio_antes = []

    # ==========================================================
    # Diagnóstico antes
    # ==========================================================

    for col, limites in dominio_numerico.items():

        qtd = (
            (
                df[col] < limites["inferior"]
            )
            |
            (
                df[col] > limites["superior"]
            )
        ).sum()

        fora_dominio_antes.append({
            "variavel": col,
            "qtd_fora": qtd,
            "%_fora": round(
                qtd / len(df) * 100,
                2
            )
        })

    for col, categorias in dominio_categorico.items():

        qtd = (
            ~df[col].isin(categorias)
        ).sum()

        fora_dominio_antes.append({
            "variavel": col,
            "qtd_fora": qtd,
            "%_fora": round(
                qtd / len(df) * 100,
                2
            )
        })

    fora_dominio_antes = (
        pd.DataFrame(fora_dominio_antes)
        .sort_values(
            "qtd_fora",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ==========================================================
    # Numéricas
    # ==========================================================

    for col, limites in dominio_numerico.items():

        df[col] = df[col].clip(
            lower=limites["inferior"],
            upper=limites["superior"]
        )

    # ==========================================================
    # Categóricas
    # ==========================================================

    for col, categorias in dominio_categorico.items():

        df.loc[
            ~df[col].isin(categorias),
            col
        ] = pd.NA

        df[col] = df[col].fillna(
            dicionario_imputacao[col]
        )

    # ==========================================================
    # Diagnóstico depois
    # ==========================================================

    fora_dominio_depois = []

    for col, limites in dominio_numerico.items():

        qtd = (
            (
                df[col] < limites["inferior"]
            )
            |
            (
                df[col] > limites["superior"]
            )
        ).sum()

        fora_dominio_depois.append({
            "variavel": col,
            "qtd_fora": qtd,
            "%_fora": round(
                qtd / len(df) * 100,
                2
            )
        })

    for col, categorias in dominio_categorico.items():

        qtd = (
            ~df[col].isin(categorias)
        ).sum()

        fora_dominio_depois.append({
            "variavel": col,
            "qtd_fora": qtd,
            "%_fora": round(
                qtd / len(df) * 100,
                2
            )
        })

    fora_dominio_depois = (
        pd.DataFrame(fora_dominio_depois)
        .sort_values(
            "qtd_fora",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        df,
        fora_dominio_antes,
        fora_dominio_depois
    )


def get_high_correlations(
    corr: pd.DataFrame,
    threshold: float = 0.70
):
    """
    Descrição:
        Retorna pares de variáveis com correlação absoluta acima do limite definido.

    Parâmetros:
        corr (pd.DataFrame): Matriz de correlação.
        threshold (float): Limite mínimo de correlação absoluta.

    Retorno:
        pd.DataFrame: Tabela com pares de variáveis altamente correlacionadas.

    Referências:
        ---
    """

    corr_pairs = (
        corr.abs()
        .stack()
        .reset_index()
    )

    corr_pairs.columns = [
        "variavel_1",
        "variavel_2",
        "correlacao"
    ]

    # remove diagonal

    corr_pairs = corr_pairs[
        corr_pairs["variavel_1"]
        != corr_pairs["variavel_2"]
    ]

    # remove duplicados

    corr_pairs["par"] = corr_pairs.apply(
        lambda x: tuple(
            sorted([
                x["variavel_1"],
                x["variavel_2"]
            ])
        ),
        axis=1
    )

    corr_pairs = (
        corr_pairs
        .drop_duplicates("par")
        .drop(columns="par")
    )

    corr_pairs = (
        corr_pairs[
            corr_pairs["correlacao"]
            >= threshold
        ]
        .sort_values(
            "correlacao",
            ascending=False
        )
        .reset_index(drop=True)
    )

    corr_pairs["correlacao"] = (
        corr_pairs["correlacao"] * 100
    ).round(2)

    return corr_pairs


def build_woe_dictionary(
    df: pd.DataFrame,
    tipos_var: dict,
    target_col: str = "target"
):
    """
    Descrição:
        Calcula o Weight of Evidence (WOE) para as variáveis categóricas da base.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo variáveis categóricas e target.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.
        target_col (str): Nome da coluna alvo.

    Retorno:
        tuple: Dicionário WOE por variável e relatório detalhado do cálculo.

    Referências:
        ---
    """

    woe_dictionary = {}
    woe_report = []

    total_good = (
        df[target_col] == 0
    ).sum()

    total_bad = (
        df[target_col] == 1
    ).sum()

    for col in tipos_var["categorical"]:

        if col not in df.columns:
            continue

        grouped = (
            df.groupby(col, dropna=False)[target_col]
            .agg(
                total="count",
                bad="sum"
            )
            .reset_index()
        )

        grouped["good"] = (
            grouped["total"]
            - grouped["bad"]
        )

        # Smoothing para evitar divisão por zero
        grouped["dist_good"] = (
            grouped["good"] + 0.5
        ) / (total_good + 0.5)

        grouped["dist_bad"] = (
            grouped["bad"] + 0.5
        ) / (total_bad + 0.5)

        grouped["woe"] = np.log(
            grouped["dist_good"]
            / grouped["dist_bad"]
        )

        # Dicionário WOE da variável
        woe_dictionary[col] = dict(
            zip(
                grouped[col],
                grouped["woe"]
            )
        )

        # Relatório padronizado
        aux = pd.DataFrame({
            "variavel": col,
            "categoria": grouped[col],
            "total": grouped["total"],
            "good": grouped["good"],
            "bad": grouped["bad"],
            "woe": grouped["woe"].round(6)
        })

        woe_report.append(aux)

    if len(woe_report) > 0:

        woe_report = (
            pd.concat(
                woe_report,
                ignore_index=True
            )
            .sort_values(
                ["variavel", "woe"],
                ascending=[True, False]
            )
            .reset_index(drop=True)
        )

    else:

        woe_report = pd.DataFrame(
            columns=[
                "variavel",
                "categoria",
                "total",
                "good",
                "bad",
                "woe"
            ]
        )

    return (
        woe_dictionary,
        woe_report
    )


def apply_woe(
    df: pd.DataFrame,
    tipos_var: dict,
    woe_dictionary: dict
):
    """
    Descrição:
        Aplica a transformação WOE nas variáveis categóricas e atualiza o
        dicionário de tipos das variáveis.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.
        woe_dictionary (dict): Dicionário com os pesos WOE por categoria.

    Retorno:
        tuple: DataFrame transformado e dicionário de tipos atualizado.

    Referências:
        ---
    """

    df = df.copy()

    categorical_cols = (
        tipos_var["categorical"]
        .copy()
    )

    for col in categorical_cols:

        new_col = f"{col}_woe"

        df[new_col] = (
            df[col]
            .map(
                woe_dictionary[col]
            )
            .astype("float64")
        )

        df.drop(
            columns=[col],
            inplace=True
        )

        tipos_var["numeric"].append(
            new_col
        )

    tipos_var["categorical"] = []

    return (
        df,
        tipos_var
    )


def build_normalization_dictionary(
    df: pd.DataFrame,
    tipos_var: dict
):
    """
    Descrição:
        Calcula os parâmetros mínimo e máximo para normalização Min-Max.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.

    Retorno:
        tuple: Dicionário de normalização e relatório dos parâmetros calculados.

    Referências:
        ---
    """

    normalization_dictionary = {}

    report = []

    cols = (
        tipos_var["numeric"]
        + tipos_var["integer"]
    )

    for col in cols:

        minimo = df[col].min()
        maximo = df[col].max()

        normalization_dictionary[col] = {
            "min": minimo,
            "max": maximo
        }

        report.append({
            "variavel": col,
            "min": minimo,
            "max": maximo
        })

    report = pd.DataFrame(report)

    return (
        normalization_dictionary,
        report
    )


def apply_normalization(
    df: pd.DataFrame,
    tipos_var: dict,
    normalization_dictionary: dict
):
    """
    Descrição:
        Aplica a normalização Min-Max nas variáveis numéricas e inteiras.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.
        normalization_dictionary (dict): Dicionário com parâmetros min/max por variável.

    Retorno:
        pd.DataFrame: DataFrame normalizado.

    Referências:
        ---
    """

    df = df.copy()

    cols = (
        tipos_var["numeric"]
        + tipos_var["integer"]
    )

    for col in cols:

        minimo = (
            normalization_dictionary[col]["min"]
        )

        maximo = (
            normalization_dictionary[col]["max"]
        )

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


def calculate_information_value(
    df: pd.DataFrame,
    tipos_var: dict,
    target_col: str = "target",
    n_bins: int = 10
):
    """
    Descrição:
        Calcula o Information Value (IV) para variáveis binárias, inteiras e
        numéricas, discretizando as contínuas em quantis.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo variáveis explicativas e target.
        tipos_var (dict): Dicionário com a classificação das variáveis por tipo.
        target_col (str): Nome da coluna alvo.
        n_bins (int): Quantidade máxima de faixas para discretização.

    Retorno:
        pd.DataFrame: Relatório com IV e classificação de força preditiva.

    Referências:
        ---
    """

    total_good = (
        df[target_col] == 0
    ).sum()

    total_bad = (
        df[target_col] == 1
    ).sum()

    iv_report = []

    variables = (
        tipos_var["binary"]
        + tipos_var["integer"]
        + tipos_var["numeric"]
    )

    for col in variables:

        try:

            temp = df[[col, target_col]].copy()

            # --------------------------------------------------
            # Binárias
            # --------------------------------------------------

            if col in tipos_var["binary"]:

                temp["_bin"] = temp[col]

            # --------------------------------------------------
            # Numéricas
            # --------------------------------------------------

            else:

                temp["_bin"] = pd.qcut(
                    temp[col],
                    q=min(
                        n_bins,
                        temp[col].nunique()
                    ),
                    duplicates="drop"
                )

            grouped = (
                temp.groupby("_bin", observed=False)[target_col]
                .agg(
                    total="count",
                    bad="sum"
                )
                .reset_index()
            )

            grouped["good"] = (
                grouped["total"]
                - grouped["bad"]
            )

            grouped["dist_good"] = (
                grouped["good"] + 0.5
            ) / (total_good + 0.5)

            grouped["dist_bad"] = (
                grouped["bad"] + 0.5
            ) / (total_bad + 0.5)

            grouped["woe"] = np.log(
                grouped["dist_good"]
                / grouped["dist_bad"]
            )

            grouped["iv_component"] = (
                grouped["dist_good"]
                - grouped["dist_bad"]
            ) * grouped["woe"]

            iv = grouped[
                "iv_component"
            ].sum()

            if iv < 0.01:

                classificacao = "Muito Fraco"

            elif iv < 0.05:

                classificacao = "Fraco"

            elif iv < 0.15:

                classificacao = "Moderado"

            elif iv < 0.30:

                classificacao = "Forte"

            else:

                classificacao = "Muito Forte"

            iv_report.append({
                "variavel": col,
                "iv": round(iv, 4),
                "classificacao": classificacao
            })

        except Exception:

            continue

    return (
        pd.DataFrame(iv_report)
        .sort_values(
            "iv",
            ascending=False
        )
        .reset_index(drop=True)
    )