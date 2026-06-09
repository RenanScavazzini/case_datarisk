"""
Descrição:
    Módulo responsável pela construção de definições de target
    para inadimplência a partir das parcelas dos contratos.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 08/06/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

from typing import Optional

import pandas as pd

from .data_loader import load_historico_emprestimos, load_historico_parcelas
from .population import _filter_recent_loan_history, _select_representative_contracts


def build_contract_target() -> pd.DataFrame:
    """
    Descrição:
        Constrói indicadores de atraso por contrato a partir do histórico
        de parcelas, incluindo o atraso máximo e flags `ever_30/45/60/90`.

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
            ever_45=("delay_days", lambda x: (x > 45).any()),
            ever_60=("delay_days", lambda x: (x > 60).any()),
            ever_90=("delay_days", lambda x: (x > 90).any()),
        )
        .reset_index()
    )
    return contract_target


def build_population_target(
    active_population: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Descrição:
        Constrói o target da população ativa utilizando o próprio contrato
        representativo que originou cada observação da base.

        A unidade de modelagem do projeto é cliente + safra. Portanto,
        o target deve refletir o comportamento do contrato associado àquela
        observação e não o comportamento de outros contratos históricos
        do cliente.

        Para cada contrato representativo são calculados:
            - max_delay
            - ever_30
            - ever_45
            - ever_60
            - ever_90

        O target final é definido a partir da flag ever_45.

    Parâmetros:
        active_population (Optional[pd.DataFrame]):
            População ativa final.

    Retorno:
        pd.DataFrame:
            Base com uma linha por observação da população ativa contendo:
                - id_cliente
                - safra_mes
                - max_delay
                - ever_30
                - ever_45
                - ever_60
                - ever_90
                - target

    Referências:
        ---
    """
    emprestimos = _filter_recent_loan_history(
        load_historico_emprestimos().copy()
    )

    parcelas = load_historico_parcelas().copy()

    contract_target = build_contract_target()

    if active_population is not None:

        active_client_ids = set(
            active_population["id_cliente"].dropna().unique()
        )

        emprestimos = emprestimos[
            emprestimos["id_cliente"].isin(active_client_ids)
        ].copy()

    representative_contracts = _select_representative_contracts(
        emprestimos,
        parcelas
    )

    population_target = representative_contracts.merge(
        contract_target,
        on="id_contrato",
        how="left"
    )

    population_target = population_target[
        [
            "id_cliente",
            "safra_mes",
            "id_contrato",
            "max_delay",
            "ever_30",
            "ever_45",
            "ever_60",
            "ever_90",
        ]
    ].copy()

    population_target = population_target.fillna(
        {
            "max_delay": 0,
            "ever_30": False,
            "ever_45": False,
            "ever_60": False,
            "ever_90": False,
        }
    )

    population_target["target"] = (
        population_target["ever_45"]
        .astype(bool)
        .astype(int)
    )

    return population_target


def choose_target_definition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descrição:
        Gera a coluna `target` a partir da flag `ever_45`.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo colunas `ever_45`.

    Retorno:
        pd.DataFrame: DataFrame com coluna `target` (0/1).

    Referências:
        ---
    """
    candidate = df.copy()
    candidate["target"] = (
        candidate["ever_45"]
        .fillna(False)
        .astype(int)
    )
    return candidate

def print_target_distribution(df: pd.DataFrame, name: str):
    """
    Descrição:
        Exibe a distribuição do target por safra, incluindo contagens de
        good/bad e percentual de inadimplência, com linha de totais ao final.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas `target` e
            `safra_mes` (ou `safra`).
        name (str): Rótulo identificador da distribuição exibida no cabeçalho.

    Retorno:
        ---

    Referências:
        ---
    """
    print(f"\n--- Distribuição Target: {name} ---")
    safra_col = "safra_mes" if "safra_mes" in df.columns else "safra"

    res_safra = (
        df.groupby(safra_col)["target"]
        .agg(["count", "sum"])
        .reset_index()
    )

    res_safra.columns = ["safra", "total", "bad"]
    res_safra["good"] = res_safra["total"] - res_safra["bad"]
    res_safra["% bad"] = (res_safra["bad"] / res_safra["total"] * 100).round(2)
    res_safra = res_safra[["safra", "total", "good", "bad", "% bad"]]
    
    total_count = res_safra["total"].sum()
    total_bad = res_safra["bad"].sum()
    total_good = res_safra["good"].sum()
    total_pct_bad = round((total_bad / total_count * 100), 2) if total_count > 0 else 0
    
    res_safra["safra"] = res_safra["safra"].astype(str)
    
    total_row = pd.DataFrame([{
        "safra": "TOTAL",
        "total": total_count,
        "good": total_good,
        "bad": total_bad,
        "% bad": total_pct_bad
    }])
    
    final_res = pd.concat([res_safra, total_row], ignore_index=True)
    print(final_res.to_string(index=False))
