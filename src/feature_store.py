import pandas as pd

from .data_loader import (
    load_base_cadastral,
    load_base_submissao,
    load_historico_emprestimos,
    load_historico_parcelas,
)


"""
Descrição:
    Módulo que cria a feature store do projeto, agregando atributos
    de perfil cadastral, histórico de pagamentos e características do
    contrato atual.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""


def build_customer_profile() -> pd.DataFrame:
    """
    Descrição:
        Constrói variáveis de perfil do cliente a partir da base cadastral,
        incluindo idade e indicadores binários (possui carro/imóvel).

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Perfil do cliente com colunas selecionadas.

    Referências:
        ---
    """
    customer = load_base_cadastral().copy()
    customer["idade"] = (
        pd.Timestamp.now().year - customer["data_nascimento"].dt.year
    ).clip(lower=18)
    customer["possui_carro"] = customer["possui_carro"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
    customer["possui_imovel"] = customer["possui_imovel"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
    return customer[
        [
            "id_cliente",
            "idade",
            "qtd_filhos",
            "qtd_membros_familia",
            "renda_anual",
            "tipo_renda",
            "ocupacao",
            "tipo_organizacao",
            "nivel_educacao",
            "estado_civil",
            "tipo_moradia",
            "possui_carro",
            "possui_imovel",
            "nota_regiao_cliente",
            "nota_regiao_cliente_cidade",
        ]
    ]


def build_customer_history_features() -> pd.DataFrame:
    """
    Descrição:
        Agrega informações históricas de pagamentos por cliente, como
        taxas de atraso e proporção de pagamento realizado.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Agregações por `id_cliente` com métricas históricas.

    Referências:
        ---
    """
    emprestimos = load_historico_emprestimos().copy()
    parcelas = load_historico_parcelas().copy()
    parcelas["delay_days"] = (
        parcelas["data_real_pagamento"] - parcelas["data_prevista_pagamento"]
    ).dt.days
    parcelas["lat30"] = parcelas["delay_days"] > 30
    parcelas["lat60"] = parcelas["delay_days"] > 60
    parcelas["lat90"] = parcelas["delay_days"] > 90

    payment_agg = (
        parcelas.groupby("id_cliente")
        .agg(
            customer_parcel_count=("numero_parcela", "count"),
            customer_avg_delay=("delay_days", "mean"),
            customer_max_delay=("delay_days", "max"),
            customer_late30_rate=("lat30", "mean"),
            customer_late60_rate=("lat60", "mean"),
            customer_late90_rate=("lat90", "mean"),
            customer_payment_ratio=("valor_pago_parcela", "sum"),
            customer_expected_ratio=("valor_previsto_parcela", "sum"),
        )
        .reset_index()
    )
    payment_agg["customer_payment_ratio"] = (
        payment_agg["customer_payment_ratio"] / payment_agg["customer_expected_ratio"]
    ).fillna(1.0)
    payment_agg = payment_agg.drop(columns=["customer_expected_ratio"])

    contract_agg = (
        emprestimos.groupby("id_cliente")
        .agg(
            customer_loan_count=("id_contrato", "nunique"),
            customer_approved_count=("status_contrato", lambda x: (x == "Approved").sum()),
            customer_canceled_count=("status_contrato", lambda x: (x == "Canceled").sum()),
            customer_refused_count=("status_contrato", lambda x: (x == "Refused").sum()),
            customer_unused_offer_count=("status_contrato", lambda x: (x == "Unused offer").sum()),
            customer_credit_sum=("valor_credito", "sum"),
            customer_credit_mean=("valor_credito", "mean"),
            customer_parcel_sum=("valor_parcela", "sum"),
            customer_parcel_mean=("valor_parcela", "mean"),
            customer_interest_rate_mean=("taxa_juros_padrao", "mean"),
        )
        .reset_index()
    )

    features = contract_agg.merge(payment_agg, on="id_cliente", how="left")
    features["customer_late30_rate"] = features["customer_late30_rate"].fillna(0)
    features["customer_late60_rate"] = features["customer_late60_rate"].fillna(0)
    features["customer_late90_rate"] = features["customer_late90_rate"].fillna(0)
    features["customer_avg_delay"] = features["customer_avg_delay"].fillna(0)
    features["customer_max_delay"] = features["customer_max_delay"].fillna(0)
    features["customer_payment_ratio"] = features["customer_payment_ratio"].fillna(1.0)
    return features


def build_current_contract_features() -> pd.DataFrame:
    """
    Descrição:
        Extrai características do contrato atual que serão usadas como
        features diretas (valores, taxas, finalidade, etc.).

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: DataFrame com uma linha por `id_contrato` e colunas do contrato.

    Referências:
        ---
    """
    emprestimos = load_historico_emprestimos().copy()
    return emprestimos[
        [
            "id_contrato",
            "id_cliente",
            "tipo_contrato",
            "valor_credito",
            "valor_bem",
            "valor_parcela",
            "valor_entrada",
            "percentual_entrada",
            "qtd_parcelas_planejadas",
            "taxa_juros_padrao",
            "taxa_juros_promocional",
            "tipo_produto",
            "finalidade_emprestimo",
            "area_venda",
            "dia_semana_solicitacao",
            "hora_solicitacao",
            "flag_seguro_contratado",
        ]
    ].copy()


def build_feature_store() -> pd.DataFrame:
    """
    Descrição:
        Consolida o `feature_store` unindo perfil do cliente, histórico
        e atributos do contrato atual, além de criar métricas derivadas.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Feature store pronta para modelagem.

    Referências:
        ---
    """
    customer_profile = build_customer_profile()
    customer_history = build_customer_history_features()
    contract_features = build_current_contract_features()
    feature_store = (
        contract_features
        .merge(customer_profile, on="id_cliente", how="left")
        .merge(customer_history, on="id_cliente", how="left")
    )

    feature_store["loan_to_annual_income"] = feature_store["valor_credito"] / feature_store["renda_anual"].replace(0, pd.NA)
    feature_store["installment_to_monthly_income"] = (
        feature_store["valor_parcela"] / (feature_store["renda_anual"] / 12).replace(0, pd.NA)
    )
    feature_store["income_per_family_member"] = (
        feature_store["renda_anual"] / feature_store["qtd_membros_familia"].replace(0, pd.NA)
    )
    feature_store["percentual_entrada"] = feature_store["percentual_entrada"].fillna(0)

    return feature_store


def build_score_feature_store() -> pd.DataFrame:
    """
    Descrição:
        Constrói o conjunto de features para as solicitações de score
        (base_submissao), reutilizando as agregações históricas.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: DataFrame de score pronto para predição.

    Referências:
        ---
    """
    submissao = load_base_submissao().copy()
    customer_profile = build_customer_profile()
    customer_history = build_customer_history_features()

    score_df = (
        submissao
        .merge(customer_profile, on="id_cliente", how="left")
        .merge(customer_history, on="id_cliente", how="left")
    )

    score_df["loan_to_annual_income"] = score_df["valor_credito"] / score_df["renda_anual"].replace(0, pd.NA)
    score_df["installment_to_monthly_income"] = (
        score_df["valor_parcela"] / (score_df["renda_anual"] / 12).replace(0, pd.NA)
    )
    score_df["income_per_family_member"] = (
        score_df["renda_anual"] / score_df["qtd_membros_familia"].replace(0, pd.NA)
    )
    score_df["customer_late30_rate"] = score_df["customer_late30_rate"].fillna(0)
    score_df["customer_late60_rate"] = score_df["customer_late60_rate"].fillna(0)
    score_df["customer_late90_rate"] = score_df["customer_late90_rate"].fillna(0)
    score_df["customer_avg_delay"] = score_df["customer_avg_delay"].fillna(0)
    score_df["customer_max_delay"] = score_df["customer_max_delay"].fillna(0)
    score_df["customer_payment_ratio"] = score_df["customer_payment_ratio"].fillna(1.0)

    return score_df
