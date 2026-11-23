"""Funciones de preprocesamiento para el Heart Failure Prediction Dataset.

El mismo pipeline se usa en el notebook (entrenamiento) y en la app (predicción),
así no hay diferencias entre cómo se entrena y cómo se predice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "HeartDisease"

# Modelo completo: las 11 variables del dataset (requiere ECG / prueba de esfuerzo).
FEATURES_FULL = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# Modelo rápido: solo lo que una persona puede responder sin un electrocardiograma.
FEATURES_QUICK = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "ExerciseAngina",
]

CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# Valores permitidos de cada variable categórica.
CATEGORIES = {
    "Sex": ["M", "F"],
    "ChestPainType": ["TA", "ATA", "NAP", "ASY"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"],
}


def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV original de Kaggle."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza del dataset.

    En el dataset original un colesterol o una presión en 0 no es un valor real:
    es un dato faltante. Se reemplaza por NaN para que el imputador lo trate
    igual que cuando el usuario responde "No sé".
    """
    df = df.copy()
    for col in ("Cholesterol", "RestingBP"):
        df.loc[df[col] == 0, col] = np.nan
    return df


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """Imputación + one-hot. Random Forest no necesita escalado."""
    num = [c for c in features if c in NUMERIC]
    cat = [c for c in features if c in CATEGORICAL]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        categories=[CATEGORIES[c] for c in cat],
                        handle_unknown="ignore",
                        sparse_output=False,
                    )),
                ]),
                cat,
            ),
        ],
        verbose_feature_names_out=True,
    )


def build_pipeline(features: list[str], random_state: int = 42, **rf_params) -> Pipeline:
    """Pipeline completo: preprocesamiento + Random Forest."""
    params = {"n_estimators": 300, "random_state": random_state, "n_jobs": -1}
    params.update(rf_params)
    return Pipeline([
        ("prep", build_preprocessor(features)),
        ("rf", RandomForestClassifier(**params)),
    ])


def chest_pain_from_answers(has_pain: bool, central: bool = False,
                            exertion: bool = False, relief: bool = False) -> str:
    """Convierte preguntas en lenguaje sencillo al código ChestPainType.

    Se usan los tres criterios clásicos de angina:
      1. dolor u opresión en el centro del pecho,
      2. aparece con el esfuerzo,
      3. se alivia con el reposo.
    3 criterios -> TA (angina típica), 2 -> ATA (atípica),
    0-1 -> NAP (dolor no anginoso). Sin dolor -> ASY (asintomático).
    """
    if not has_pain:
        return "ASY"
    score = int(central) + int(exertion) + int(relief)
    if score == 3:
        return "TA"
    if score == 2:
        return "ATA"
    return "NAP"
