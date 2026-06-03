"""
Descrição:
    Módulo com regras de política de crédito e utilitários para gerar
    submissões finais a partir de probabilidades previstas.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import pandas as pd
from pathlib import Path

from .utils import safe_ratio

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "submissions"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def build_credit_policy(predictions: pd.DataFrame, cutoff: float = 0.5) -> pd.DataFrame:
    """
    Descrição:
        Cria decisões de crédito com base na probabilidade prevista,
        classificando em faixas de risco e aplicando um cutoff para
        aprovação.

    Parâmetros:
        predictions (pd.DataFrame): DataFrame contendo uma coluna `probability`.
        cutoff (float): Limiar de probabilidade acima do qual o pedido é negado.

    Retorno:
        pd.DataFrame: DataFrame com colunas adicionais `risk_score`, `decision` e `risk_band`.

    Referências:
        ---
    """
    scores = predictions.copy()
    scores["risk_score"] = scores["probability"]
    scores["decision"] = scores["risk_score"].apply(
        lambda x: "Approve" if x < cutoff else "Deny"
    )
    scores["risk_band"] = pd.cut(
        scores["risk_score"],
        bins=[-0.01, 0.02, 0.05, 0.1, 0.2, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )
    return scores


def save_submission(df: pd.DataFrame, filename: str = "submissao_case.csv"):
    """
    Descrição:
        Salva o DataFrame de submissão como CSV em `outputs/submissions`.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser salvo.
        filename (str): Nome do arquivo CSV de saída.

    Retorno:
        ---

    Referências:
        ---
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH / filename, index=False)
