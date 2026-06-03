"""
Descrição:
    Módulo responsável pela definição e persistência das populações
    de treino e score utilizadas ao longo do projeto.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

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


def build_population_train() -> pd.DataFrame:
    """
    Descrição:
        Constrói a população de treino a partir do histórico de empréstimos,
        filtrando contratos aprovados e com histórico de parcelas.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: DataFrame contendo contratos elegíveis para treino.

    Referências:
        ---
    """
    emprestimos = load_historico_emprestimos()
    parcelas = load_historico_parcelas()
    emprestimos = emprestimos.query("status_contrato == 'Approved'").copy()
    contratos_com_parcelas = set(parcelas["id_contrato"])
    emprestimos = emprestimos[emprestimos["id_contrato"].isin(contratos_com_parcelas)].copy()
    return emprestimos


def build_population_score() -> pd.DataFrame:
    """
    Descrição:
        Constrói a população de score a partir da base de submissão,
        realizando o join com dados cadastrais do cliente.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: DataFrame pronto para cálculo de score.

    Referências:
        ---
    """
    submissao = load_base_submissao()
    cadastral = load_base_cadastral()
    score = submissao.merge(cadastral, on="id_cliente", how="left")
    return score


def save_population_train(df: pd.DataFrame):
    """
    Descrição:
        Persiste a população de treino em `data/processed/population_train.parquet`.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser salvo.

    Retorno:
        ---

    Referências:
        ---
    """
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH / "population_train.parquet", index=False)


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
