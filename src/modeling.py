"""
Descrição:
    Módulo de treinamento e serialização de modelos.
    Fornece função para treinar modelos (LogisticRegression e opcionalmente LightGBM)
    e salvar os artefatos em `outputs/models`.

Autor:
    Renan Douglas Floriano Scavazzini
    Email: renanscavazzini@gmail.com

Versão:
    1.0 - 12/05/2026

Copyright:
    Copyright (c) 2026 Renan Douglas Floriano Scavazzini
"""

import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .feature_engineering import prepare_model_dataset
from .utils import evaluate_classification

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

MODEL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def train_models(df, target_col="target"):
    """
    Descrição:
        Treina modelos de classificação binária usando um pipeline simples.
        Treina `LogisticRegression` sempre e `LightGBM` caso esteja instalado.

    Parâmetros:
        df (pd.DataFrame): Dataset que contém as features e a coluna target.
        target_col (str): Nome da coluna target no DataFrame.

    Retorno:
        dict: Dicionário com modelos treinados, métricas (`results`) e
              dados de validação (`X_test`, `y_test`, `y_pred_proba`).

    Referências:
        scikit-learn documentation; LightGBM docs
    """
    X, y, processed = prepare_model_dataset(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=2000, solver='saga', random_state=42)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    results = {"logistic": evaluate_classification(y_test, lr_probs)}
    y_pred_proba = {"logistic": lr_probs}
    models = {"logistic": lr}

    if LGBMClassifier is not None:
        lgbm = LGBMClassifier(random_state=42, n_estimators=200)
        lgbm.fit(X_train, y_train)
        lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
        results["lightgbm"] = evaluate_classification(y_test, lgbm_probs)
        y_pred_proba["lightgbm"] = lgbm_probs
        models["lightgbm"] = lgbm
        joblib.dump(lgbm, MODEL_PATH / "final_model.pkl")
    else:
        print("LightGBM is not installed. Training only Logistic Regression.")
        joblib.dump(lr, MODEL_PATH / "final_model.pkl")

    joblib.dump(lr, MODEL_PATH / "logistic_model.pkl")

    return {
        "models": models,
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba,
    }


def load_model(name="final_model.pkl"):
    """
    Descrição:
        Carrega e retorna um modelo serializado em `outputs/models`.

    Parâmetros:
        name (str): Nome do arquivo do modelo a ser carregado.

    Retorno:
        object: Objeto do modelo carregado via `joblib`.

    Referências:
        joblib documentation
    """
    return joblib.load(MODEL_PATH / name)
