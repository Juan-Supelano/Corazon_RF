"""Carga de modelos, predicción y explicación de cada resultado."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURES_FULL, FEATURES_QUICK

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODEL_FILES = {
    "rapido": MODELS_DIR / "rf_rapido.pkl",
    "completo": MODELS_DIR / "rf_completo.pkl",
}
FEATURES = {"rapido": FEATURES_QUICK, "completo": FEATURES_FULL}

# Nombres legibles para mostrar en la interfaz.
LABELS = {
    "Age": "Edad",
    "Sex": "Sexo",
    "ChestPainType": "Tipo de dolor en el pecho",
    "RestingBP": "Presión arterial en reposo",
    "Cholesterol": "Colesterol",
    "FastingBS": "Glucosa en ayunas > 120",
    "RestingECG": "ECG en reposo",
    "MaxHR": "Frecuencia cardiaca máxima",
    "ExerciseAngina": "Dolor de pecho con ejercicio",
    "Oldpeak": "Depresión del ST (Oldpeak)",
    "ST_Slope": "Pendiente del ST",
}

VALUE_LABELS = {
    "Sex": {"M": "Hombre", "F": "Mujer"},
    "ChestPainType": {"TA": "Angina típica", "ATA": "Angina atípica",
                      "NAP": "Dolor no anginoso", "ASY": "Sin dolor"},
    "RestingECG": {"Normal": "Normal", "ST": "Anomalía ST-T", "LVH": "Hipertrofia VI"},
    "ExerciseAngina": {"Y": "Sí", "N": "No"},
    "ST_Slope": {"Up": "Ascendente", "Flat": "Plana", "Down": "Descendente"},
    "FastingBS": {1: "Sí", 0: "No"},
}

UNITS = {"Age": "años", "RestingBP": "mmHg", "Cholesterol": "mg/dL", "MaxHR": "lpm"}

# Umbrales del semáforo sobre la probabilidad estimada.
LOW, HIGH = 0.35, 0.65


@lru_cache(maxsize=None)
def load_model(mode: str):
    return joblib.load(MODEL_FILES[mode])


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def risk_level(proba: float) -> dict:
    if proba < LOW:
        return {"key": "bajo", "label": "Riesgo bajo", "color": "green"}
    if proba < HIGH:
        return {"key": "medio", "label": "Riesgo moderado", "color": "amber"}
    return {"key": "alto", "label": "Riesgo alto", "color": "red"}


def display_value(feature: str, value) -> str:
    """Texto para mostrar un valor ingresado (o 'No sé' si falta)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "No sé (se estimó)"
    if feature in VALUE_LABELS:
        key = int(value) if feature == "FastingBS" else value
        return VALUE_LABELS[feature].get(key, str(value))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    unit = UNITS.get(feature)
    return f"{value} {unit}" if unit else str(value)


def explain(pipeline, X: pd.DataFrame, features: list[str]) -> tuple[float, dict]:
    """Contribución de cada variable a la probabilidad (método de Saabas).

    En cada árbol se recorre el camino desde la raíz hasta la hoja. Cada vez que
    un nodo divide por una variable, el cambio en la proporción de casos
    positivos entre el nodo padre y el hijo se le asigna a esa variable.
    Promediando sobre todos los árboles:
        probabilidad = base + suma(contribuciones)
    Es exacto para Random Forest y no requiere librerías extra como SHAP.
    """
    prep = pipeline.named_steps["prep"]
    rf = pipeline.named_steps["rf"]
    Xt = np.asarray(prep.transform(X), dtype=np.float32)
    names = prep.get_feature_names_out()

    # Cada columna transformada (p. ej. 'cat__Sex_M') pertenece a una variable original.
    owner = []
    for n in names:
        raw = n.split("__", 1)[1]
        owner.append(next(f for f in features if raw == f or raw.startswith(f + "_")))

    contrib_t = np.zeros(len(names))
    base = 0.0
    for tree in rf.estimators_:
        t = tree.tree_
        values = t.value[:, 0, :]
        values = values / values.sum(axis=1, keepdims=True)
        pos = values[:, 1]
        node = 0
        base += pos[0]
        while t.children_left[node] != -1:
            f = t.feature[node]
            child = (t.children_left[node] if Xt[0, f] <= t.threshold[node]
                     else t.children_right[node])
            contrib_t[f] += pos[child] - pos[node]
            node = child
    n_trees = len(rf.estimators_)
    base /= n_trees
    contrib_t /= n_trees

    contrib = {f: 0.0 for f in features}
    for i, f in enumerate(owner):
        contrib[f] += float(contrib_t[i])
    return float(base), contrib


def predict(mode: str, data: dict) -> dict:
    """Predice el riesgo para un paciente.

    data: diccionario con las variables del modo elegido. Un valor None
    significa "No sé" y lo completa el imputador del pipeline.
    """
    if mode not in MODEL_FILES:
        raise ValueError(f"Modo desconocido: {mode}")
    features = FEATURES[mode]
    row = {f: (np.nan if data.get(f) is None else data[f]) for f in features}
    X = pd.DataFrame([row], columns=features)

    pipeline = load_model(mode)
    proba = float(pipeline.predict_proba(X)[0, 1])
    base, contrib = explain(pipeline, X, features)

    # Voto de cada árbol: 1 si ese árbol clasifica al paciente como enfermo.
    Xt = np.asarray(pipeline.named_steps["prep"].transform(X), dtype=np.float32)
    votes = [int(t.predict_proba(Xt)[0, 1] > 0.5) for t in pipeline.named_steps["rf"].estimators_]

    factors = sorted(
        (
            {
                "feature": f,
                "label": LABELS[f],
                "value": display_value(f, row[f]),
                "impact": round(c * 100, 1),  # puntos porcentuales
            }
            for f, c in contrib.items()
        ),
        key=lambda d: abs(d["impact"]),
        reverse=True,
    )
    return {
        "mode": mode,
        "probability": round(proba * 100, 1),
        "base": round(base * 100, 1),
        "level": risk_level(proba),
        "votes": votes,
        "votes_yes": sum(votes),
        "factors": factors,
        "missing": [LABELS[f] for f in features if data.get(f) is None],
    }
