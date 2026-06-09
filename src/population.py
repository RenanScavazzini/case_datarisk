"""
Descrição:
    Módulo responsável pela definição e persistência das populações
    ativa e de score utilizadas ao longo do projeto.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 08/06/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

from pathlib import Path
import pandas as pd

from .data_loader import (
    load_base_cadastral,
    load_base_submissao,
    load_historico_emprestimos,
    load_historico_parcelas,
)

PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed"

FINAL_POPULATION_COLUMNS = [
    "id_cliente",
    "data_solicitacao",
    "dia_semana_solicitacao_submissao",
    "hora_solicitacao_submissao",
    "tipo_contrato_submissao",
    "valor_credito_submissao",
    "valor_bem_submissao",
    "valor_parcela_submissao",
    "sexo",
    "data_nascimento",
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
    "safra_mes",
    "qtd_contratos_aceitos_historico",
    "qtd_contratos_recusados_historico",
    "soma_valor_credito_ativo_historico",
    "media_valor_credito_aceito_historico",
]

HISTORICAL_CREDIT_FEATURE_COLUMNS = [
    "qtd_contratos_aceitos_historico",
    "qtd_contratos_recusados_historico",
    "soma_valor_credito_ativo_historico",
    "media_valor_credito_aceito_historico",
]


def _deduplicate_customer_vintage(
    df: pd.DataFrame,
    sort_columns: list[str],
) -> pd.DataFrame:
    """
    Descrição:
        Mantém uma única linha por combinação de cliente e safra,
        preservando o evento mais recente conforme as colunas de ordenação.

    Parâmetros:
        df (pd.DataFrame): DataFrame com registros por cliente e safra.
        sort_columns (list[str]): Colunas adicionais usadas para ordenar os eventos.

    Retorno:
        pd.DataFrame: DataFrame deduplicado por cliente e safra.

    Referências:
        ---
    """
    return (
        df.sort_values(
            ["id_cliente", "safra_mes", *sort_columns],
            na_position="first",
        )
        .drop_duplicates(["id_cliente", "safra_mes"], keep="last")
        .reset_index(drop=True)
    )


def _filter_recent_loan_history(emprestimos: pd.DataFrame) -> pd.DataFrame:
    """
    Descrição:
        Filtra o histórico de empréstimos para o período relevante de modelagem.

    Parâmetros:
        emprestimos (pd.DataFrame): Histórico completo de empréstimos.

    Retorno:
        pd.DataFrame: Histórico filtrado dentro da janela temporal elegível.

    Referências:
        ---
    """
    emprestimos = emprestimos.copy()
    emprestimos["data_decisao"] = pd.to_datetime(
        emprestimos["data_decisao"], errors="coerce"
    )
    min_date = pd.Timestamp("2020-01-01")
    max_date = pd.Timestamp("2025-01-31")
    return emprestimos[
        emprestimos["data_decisao"].between(min_date, max_date)
    ]


def _select_representative_contracts(emprestimos: pd.DataFrame, parcelas: pd.DataFrame) -> pd.DataFrame:
    """
    Descrição:
        Seleciona os contratos elegíveis e representativos por cliente e safra
        a partir do histórico de empréstimos e parcelas.

    Parâmetros:
        emprestimos (pd.DataFrame): Histórico de contratos de empréstimo.
        parcelas (pd.DataFrame): Histórico de parcelas associadas aos contratos.

    Retorno:
        pd.DataFrame: Contratos representativos por cliente e safra.

    Referências:
        ---
    """
    emprestimos = _filter_recent_loan_history(emprestimos)
    contratos_com_parcelas = set(parcelas["id_contrato"])
    emprestimos = emprestimos.query("status_contrato == 'Approved'").copy()
    emprestimos = emprestimos[emprestimos["id_contrato"].isin(contratos_com_parcelas)].copy()
    emprestimos["safra_mes"] = emprestimos["data_decisao"].dt.to_period("M").astype(str)
    emprestimos = _deduplicate_customer_vintage(
        emprestimos,
        sort_columns=["data_decisao", "id_contrato"],
    )
    return emprestimos


def _build_historical_credit_features(
    population: pd.DataFrame,
    emprestimos: pd.DataFrame,
    date_column: str = "data_solicitacao",
) -> pd.DataFrame:
    """
    Descrição:
        Adiciona métricas históricas de crédito à população até a data de
        referência de cada observação.

    Parâmetros:
        population (pd.DataFrame): População base para enriquecimento.
        emprestimos (pd.DataFrame): Histórico de empréstimos do cliente.
        date_column (str): Coluna de data usada como referência temporal.

    Retorno:
        pd.DataFrame: População enriquecida com métricas históricas de crédito.

    Referências:
        ---
    """
    population = population.copy()
    emprestimos = emprestimos.copy()
    emprestimos["data_decisao"] = pd.to_datetime(
        emprestimos["data_decisao"], errors="coerce"
    )

    population_index = population.drop(
        columns=HISTORICAL_CREDIT_FEATURE_COLUMNS,
        errors="ignore",
    ).reset_index().rename(columns={"index": "_row_id"})
    merged = population_index[["_row_id", "id_cliente", date_column]].merge(
        emprestimos[
            ["id_cliente", "data_decisao", "status_contrato", "valor_credito"]
        ],
        on="id_cliente",
        how="left",
    )
    merged = merged[merged["data_decisao"] <= merged[date_column]].copy()
    merged["valor_credito_aceito"] = merged["valor_credito"].where(
        merged["status_contrato"] == "Approved", pd.NA
    )

    agg = (
        merged.groupby("_row_id")
        .agg(
            qtd_contratos_aceitos_historico=(
                "status_contrato",
                lambda x: (x == "Approved").sum(),
            ),
            qtd_contratos_recusados_historico=(
                "status_contrato",
                lambda x: ((x != "Approved") & x.notna()).sum(),
            ),
            soma_valor_credito_ativo_historico=(
                "valor_credito_aceito",
                "sum",
            ),
            media_valor_credito_aceito_historico=(
                "valor_credito_aceito",
                "mean",
            ),
        )
        .reindex(population_index["_row_id"])
        .fillna(0)
        .reset_index()
    )

    population = population_index.merge(agg, on="_row_id", how="left").drop(
        columns=["_row_id"]
    )
    population["qtd_contratos_aceitos_historico"] = population["qtd_contratos_aceitos_historico"].astype(int)
    population["qtd_contratos_recusados_historico"] = population["qtd_contratos_recusados_historico"].astype(int)
    population["soma_valor_credito_ativo_historico"] = population["soma_valor_credito_ativo_historico"].astype(float)
    population["media_valor_credito_aceito_historico"] = population["media_valor_credito_aceito_historico"].astype(float)
    return population


def _standardize_population_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descrição:
        Ajusta o DataFrame retornado para o esquema final padronizado das
        populações ativa e de score.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser padronizado.

    Retorno:
        pd.DataFrame: DataFrame com colunas padronizadas e ordenadas.

    Referências:
        ---
    """
    df = df.rename(
        columns={
            "dia_semana_solicitacao": "dia_semana_solicitacao_submissao",
            "hora_solicitacao": "hora_solicitacao_submissao",
            "tipo_contrato": "tipo_contrato_submissao",
            "valor_credito": "valor_credito_submissao",
            "valor_bem": "valor_bem_submissao",
            "valor_parcela": "valor_parcela_submissao",
        }
    )
    for col in FINAL_POPULATION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[FINAL_POPULATION_COLUMNS]


def build_active_population() -> pd.DataFrame:
    """
    Descrição:
        Constrói a população ativa em nível de cliente + safra.

        A população ativa é montada a partir da base cadastral e de todos os
        contratos aprovados do histórico que possuem parcelamento. Dessa forma,
        o mesmo cliente pode aparecer em diferentes safras quando tiver mais de
        um crédito aprovado em épocas distintas. Quando houver mais de um
        contrato no mesmo cliente + safra, é mantido o contrato mais recente.

        O campo `safra_mes` é calculado a partir de `data_decisao` do contrato.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: DataFrame contendo clientes elegíveis e seus contratos
        aprovados relevantes para a população ativa.

    Referências:
        ---
    """
    cadastral = load_base_cadastral().copy()
    emprestimos = load_historico_emprestimos().copy()
    parcelas = load_historico_parcelas()

    representative_contracts = _select_representative_contracts(
        emprestimos,
        parcelas,
    )

    active = cadastral.merge(
        representative_contracts,
        on="id_cliente",
        how="inner",
    )
    active["data_solicitacao"] = active["data_decisao"]
    active["safra_mes"] = active["data_solicitacao"].dt.to_period("M").astype(str)
    active = _deduplicate_customer_vintage(
        active,
        sort_columns=["data_solicitacao", "id_contrato"],
    )
    active = _standardize_population_columns(active)
    active = _build_historical_credit_features(
        active,
        emprestimos,
        date_column="data_solicitacao",
    )
    return active


# backward compatibility alias
build_population_train = build_active_population


def build_population_score() -> pd.DataFrame:
    """
    Descrição:
        Constrói a população de score a partir da base de submissão e cadastro.

        A unidade da população de score também é cliente + safra. Quando
        houver mais de uma solicitação no mesmo mês para o mesmo cliente, é
        mantida a solicitação mais recente. O histórico de empréstimos é usado
        apenas para construir métricas históricas até a data de referência.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: DataFrame pronto para cálculo de score.

    Referências:
        ---
    """
    submissao = load_base_submissao().copy()
    cadastral = load_base_cadastral().copy()

    score = submissao.merge(cadastral, on="id_cliente", how="left")
    score["safra_mes"] = score["data_solicitacao"].dt.to_period("M").astype(str)
    score = _deduplicate_customer_vintage(
        score,
        sort_columns=["data_solicitacao"],
    )
    score = _standardize_population_columns(score)
    score = _build_historical_credit_features(
        score,
        load_historico_emprestimos().copy(),
        date_column="data_solicitacao",
    )
    return score


def save_population_active(df: pd.DataFrame):
    """
    Descrição:
        Persiste a população ativa em `data/processed/population_active.parquet`.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser salvo.

    Retorno:
        ---

    Referências:
        ---
    """
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH / "population_active.parquet", index=False)


# backward compatibility alias
save_population_train = save_population_active


def save_population_score(df: pd.DataFrame):
    """
    Descrição:
        Persiste a população de score em `data/processed/population_score.parquet`.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser salvo.

    Retorno:
        ---

    Referências:
        ---
    """
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH / "population_score.parquet", index=False)
