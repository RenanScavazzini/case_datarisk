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


def build_population_target(active_population: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Descrição:
        Constrói um target em nível de cliente a partir do histórico de
        parcelas e do histórico de empréstimos remanescente ao lado do
        contrato representativo da população ativa.

        A regra de negócio considera como inadimplência o cliente que teve
        pelo menos um atraso superior a 60 dias em contratos distintos do
        contrato que compõe a população ativa.
        A base de submissão não participa da definição do target; apenas o
        histórico de empréstimos e o histórico de parcelas são utilizados.
        Quando fornecida, a população ativa é usada para definir o universo
        de clientes elegíveis ao target.

    Parâmetros:
        active_population (Optional[pd.DataFrame]): População ativa final.

    Retorno:
        pd.DataFrame: DataFrame com uma linha por `id_cliente` e as colunas:
            - max_delay
            - ever_30
            - ever_60
            - ever_90
            - contracts_with_delays
            - target

    Referências:
        ---
    """
    emprestimos = _filter_recent_loan_history(load_historico_emprestimos().copy())
    parcelas = load_historico_parcelas().copy()

    if active_population is not None and "id_cliente" in active_population.columns:
        active_client_ids = set(active_population["id_cliente"].dropna().unique())
        active_emprestimos = emprestimos[emprestimos["id_cliente"].isin(active_client_ids)]
        representative_contracts = _select_representative_contracts(active_emprestimos, parcelas)
    else:
        representative_contracts = _select_representative_contracts(emprestimos, parcelas)

    remaining_emprestimos = emprestimos.loc[
        ~emprestimos["id_contrato"].isin(representative_contracts["id_contrato"])
    ].copy()

    if remaining_emprestimos.empty:
        client_target = (
            representative_contracts["id_cliente"]
            .drop_duplicates()
            .to_frame()
            .assign(
                max_delay=0,
                ever_30=False,
                ever_45=False,
                ever_60=False,
                ever_90=False,
                contracts_with_delays=0,
                target=0,
            )
        )
        return client_target

    other_parcelas = parcelas.merge(
        remaining_emprestimos[["id_contrato", "id_cliente"]],
        on="id_contrato",
        how="inner",
        suffixes=("", "_target"),
    )
    other_parcelas["id_cliente"] = other_parcelas["id_cliente_target"]
    other_parcelas = other_parcelas.drop(columns=["id_cliente_target"])
    other_parcelas["delay_days"] = (
        other_parcelas["data_real_pagamento"]
        - other_parcelas["data_prevista_pagamento"]
    ).dt.days

    client_target = (
        other_parcelas.groupby("id_cliente")
        .agg(
            max_delay=("delay_days", "max"),
            ever_30=("delay_days", lambda x: (x > 30).any()),
            ever_45=("delay_days", lambda x: (x > 45).any()),
            ever_60=("delay_days", lambda x: (x > 60).any()),
            ever_90=("delay_days", lambda x: (x > 90).any()),
            contracts_with_delays=("id_contrato", "nunique"),
        )
        .reset_index()
    )

    active_clients = representative_contracts[["id_cliente"]].drop_duplicates()
    client_target = active_clients.merge(client_target, on="id_cliente", how="left")
    client_target = client_target.fillna(
        {
            "max_delay": 0,
            "ever_30": False,
            "ever_45": False,
            "ever_60": False,
            "ever_90": False,
            "contracts_with_delays": 0,
        }
    )
    client_target["target"] = client_target["ever_45"].astype(int)
    return client_target


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
    print(f"\n--- Distribuição Target: {name} ---")
    res_safra = df.groupby("safra")["target"].agg(["count", "sum"]).reset_index()
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
