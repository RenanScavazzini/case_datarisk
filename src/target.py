"""
Descrição:
    Módulo responsável pela construção de definições de target
    para inadimplência a partir das parcelas dos contratos.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd

from .data_loader import load_historico_emprestimos, load_historico_parcelas


def build_contract_target() -> pd.DataFrame:
    """
    Descrição:
        Constrói indicadores de atraso por contrato a partir do histórico
        de parcelas, incluindo o atraso máximo e flags `ever_30/60/90`.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: DataFrame com uma linha por `id_contrato` e colunas:
            - max_delay
            - ever_30
            - ever_60
            - ever_90

    Referências:
        ---
    """
    parcelas = load_historico_parcelas().copy()
    parcelas["delay_days"] = (
        parcelas["data_real_pagamento"] - parcelas["data_prevista_pagamento"]
    ).dt.days
    contract_target = (
        parcelas.groupby("id_contrato")
        .agg(
            max_delay=("delay_days", "max"),
            ever_30=("delay_days", lambda x: (x > 30).any()),
            ever_60=("delay_days", lambda x: (x > 60).any()),
            ever_90=("delay_days", lambda x: (x > 90).any()),
        )
        .reset_index()
    )
    return contract_target


def build_population_target() -> pd.DataFrame:
    """
    Descrição:
        Mescla a definição de target por contrato com o histórico de
        empréstimos para permitir análise e treinamento por contrato.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Histórico de empréstimos enriquecido com colunas de target.

    Referências:
        ---
    """
    emprestimos = load_historico_emprestimos()
    target = build_contract_target()
    return emprestimos.merge(target, on="id_contrato", how="left")


def choose_target_definition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descrição:
        Gera a coluna `target` a partir da flag `ever_60`.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo colunas `ever_60`.

    Retorno:
        pd.DataFrame: DataFrame com coluna `target` (0/1).

    Referências:
        ---
    """
    candidate = df.copy()
    candidate["target"] = candidate["ever_60"].astype(int)
    return candidate
