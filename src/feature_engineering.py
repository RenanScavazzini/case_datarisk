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


def prepare_model_dataset(df: pd.DataFrame, target_col: str = None):
    """
    Descrição:
        Prepara o dataset para modelagem: garante presença de colunas
        numéricas, aplica imputação por mediana e codifica variáveis
        categóricas com um `OrdinalEncoder` simples.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada contendo features brutas.
        target_col (str, opcional): Nome da coluna target. Se fornecido,
            retorna também o vetor `y`.

    Retorno:
        tuple: `(X, y, df_processed)` onde `X` é o DataFrame de features,
               `y` é a Series do target (ou `None`) e `df_processed` é o
               DataFrame original transformado.

    Referências:
        ---
    """
    df = df.copy()
    numeric_cols = [
        "valor_credito",
        "valor_bem",
        "valor_parcela",
        "valor_entrada",
        "percentual_entrada",
        "qtd_parcelas_planejadas",
        "taxa_juros_padrao",
        "taxa_juros_promocional",
        "idade",
        "qtd_filhos",
        "qtd_membros_familia",
        "renda_anual",
        "nota_regiao_cliente",
        "nota_regiao_cliente_cidade",
        "customer_parcel_count",
        "customer_avg_delay",
        "customer_max_delay",
        "customer_late30_rate",
        "customer_late60_rate",
        "customer_late90_rate",
        "customer_payment_ratio",
        "customer_loan_count",
        "customer_approved_count",
        "customer_canceled_count",
        "customer_refused_count",
        "customer_unused_offer_count",
        "customer_credit_sum",
        "customer_credit_mean",
        "customer_parcel_sum",
        "customer_parcel_mean",
        "customer_interest_rate_mean",
        "loan_to_annual_income",
        "installment_to_monthly_income",
        "income_per_family_member",
    ]
    category_cols = [
        "tipo_contrato",
        "tipo_produto",
        "finalidade_emprestimo",
        "area_venda",
        "dia_semana_solicitacao",
        "flag_seguro_contratado",
        "tipo_renda",
        "ocupacao",
        "tipo_organizacao",
        "nivel_educacao",
        "estado_civil",
        "tipo_moradia",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in category_cols:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].fillna("missing").astype(str)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    present_cat_cols = [col for col in category_cols if col in df.columns]
    if present_cat_cols:
        df[present_cat_cols] = encoder.fit_transform(df[present_cat_cols])

    feature_columns = [
        col for col in numeric_cols + present_cat_cols
        if col in df.columns
    ]
    X = df[feature_columns].copy()
    y = df[target_col] if target_col is not None and target_col in df.columns else None
    return X, y, df
