"""
Descrição:
    Módulo responsável pelo carregamento dos dados brutos do projeto
    e conversão de colunas de data para tipos adequados.

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

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed"


def parse_dates(df: pd.DataFrame, columns):
    """
    Descrição:
        Converte colunas de texto para objetos datetime no DataFrame.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        columns (list): Lista de nomes de colunas a converter.

    Retorno:
        pd.DataFrame: DataFrame com colunas convertidas para datetime.

    Referências:
        ---
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_parquet(path: Path) -> pd.DataFrame:
    """
    Descrição:
        Carrega um arquivo Parquet e retorna um DataFrame.

    Parâmetros:
        path (Path): Caminho para o arquivo parquet.

    Retorno:
        pd.DataFrame: DataFrame carregado do parquet.

    Referências:
        pandas.read_parquet documentation
    """
    return pd.read_parquet(path)


def load_base_cadastral() -> pd.DataFrame:
    """
    Descrição:
        Carrega a base cadastral (informações dos clientes) e
        aplica a conversão de datas para `data_nascimento`.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Base cadastral com `data_nascimento` como datetime.

    Referências:
        ---
    """
    df = load_parquet(RAW_PATH / "base_cadastral.parquet")
    return parse_dates(df, ["data_nascimento"])


def load_base_submissao() -> pd.DataFrame:
    """
    Descrição:
        Carrega a base de submissão (pedidos de crédito) e
        converte `data_solicitacao` para datetime.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Base de submissão com `data_solicitacao` como datetime.

    Referências:
        ---
    """
    df = load_parquet(RAW_PATH / "base_submissao.parquet")
    return parse_dates(df, ["data_solicitacao"])


def load_historico_emprestimos() -> pd.DataFrame:
    """
    Descrição:
        Carrega o histórico de empréstimos e realiza a conversão das
        colunas de data relevantes para datetime.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Histórico de empréstimos com colunas de data convertidas.

    Referências:
        ---
    """
    df = load_parquet(RAW_PATH / "historico_emprestimos.parquet")
    return parse_dates(
        df,
        [
            "data_decisao",
            "data_liberacao",
            "data_primeiro_vencimento",
            "data_ultimo_vencimento_original",
            "data_ultimo_vencimento",
            "data_encerramento",
        ],
    )


def load_historico_parcelas() -> pd.DataFrame:
    """
    Descrição:
        Carrega o histórico de parcelas e converte as colunas de data
        `data_prevista_pagamento` e `data_real_pagamento`.

    Parâmetros:
        ---

    Retorno:
        pd.DataFrame: Histórico de parcelas com datas convertidas.

    Referências:
        ---
    """
    df = load_parquet(RAW_PATH / "historico_parcelas.parquet")
    return parse_dates(df, ["data_prevista_pagamento", "data_real_pagamento"])
