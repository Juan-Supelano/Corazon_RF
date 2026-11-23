"""Construye y ejecuta notebooks/01_training.ipynb.

Se usa un ejecutor propio (sin Jupyter) para que el notebook quede guardado
con sus salidas: texto, tablas y gráficos.

Uso:
  python scripts/build_notebook.py            # notebook completo
  python scripts/build_notebook.py --stage eda   # solo la parte de EDA
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "01_training.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n")}


# ----------------------------------------------------------------------------
# Celdas
# ----------------------------------------------------------------------------
EDA = [
    md("""
# Predicción de enfermedad cardiaca con Random Forest

**Inteligencia Artificial I · Actividad 3**
Integrantes: Juan Supelano · pipereinaa · Edwinvas178-bot

Objetivo: entrenar dos clasificadores Random Forest con el *Heart Failure Prediction Dataset*:

| Modelo | Variables | Usuario |
|---|---|---|
| **Rápido** | 7 (edad, sexo, síntomas, presión, colesterol, glucosa) | Cualquier persona |
| **Completo** | 11 (incluye ECG y prueba de esfuerzo) | Personal de salud |

La comparación entre los dos muestra cuánta precisión se pierde al quitar los datos clínicos.
"""),
    md("## 1. Carga y exploración de datos"),
    code("""
import sys, json, warnings
from datetime import date
sys.path.append('..')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from src.preprocessing import (load_raw, clean, build_pipeline, FEATURES_FULL,
                               FEATURES_QUICK, CATEGORICAL, NUMERIC, TARGET)

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 110
IMG = '../app/static/img/'   # las figuras se reutilizan en la página "El modelo"

df = load_raw('../data/raw/heart.csv')
print(f'Registros: {df.shape[0]} | Columnas: {df.shape[1]}')
df.head()
"""),
    code("df.info()"),
    code("df.describe().T.round(2)"),
    md("""
**Diccionario de variables**

| Variable | Descripción |
|---|---|
| Age | Edad en años |
| Sex | M / F |
| ChestPainType | TA angina típica, ATA angina atípica, NAP dolor no anginoso, ASY asintomático |
| RestingBP | Presión arterial en reposo (mmHg) |
| Cholesterol | Colesterol sérico (mg/dL) |
| FastingBS | 1 si la glucosa en ayunas > 120 mg/dL |
| RestingECG | Normal, ST (anomalía ST-T), LVH (hipertrofia ventricular izquierda) |
| MaxHR | Frecuencia cardiaca máxima alcanzada |
| ExerciseAngina | Angina inducida por ejercicio (Y/N) |
| Oldpeak | Depresión del segmento ST |
| ST_Slope | Pendiente del ST en el pico del ejercicio (Up, Flat, Down) |
| **HeartDisease** | **Variable objetivo: 1 enfermedad cardiaca, 0 normal** |
"""),
    md("## 2. Análisis exploratorio (EDA)\n### 2.1 Valores nulos y valores imposibles"),
    code("""
print('Nulos por columna:')
print(df.isna().sum().to_string())
print()
print('Registros con valor 0:')
print((df[['RestingBP', 'Cholesterol']] == 0).sum().to_string())
"""),
    md("""
`Cholesterol` tiene 172 vacíos y `RestingBP` 1. En el CSV original de Kaggle estos faltantes venían como **0**,
un valor fisiológicamente imposible; en nuestra copia ya están como vacíos. La función `clean()` convierte los
ceros a `NaN` por si se usa el archivo original, y el pipeline los imputa con la mediana. Es el mismo mecanismo
que usa la app cuando el usuario responde "No sé".
"""),
    md("### 2.2 Variable objetivo"),
    code("""
counts = df[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 3.2))
sns.barplot(x=['Normal (0)', 'Enfermedad (1)'], y=counts.values, ax=ax, palette=['#2b8a3e', '#c92a2a'])
for i, v in enumerate(counts.values):
    ax.text(i, v + 8, f'{v} ({v / len(df):.1%})', ha='center')
ax.set_ylabel('Pacientes'); ax.set_title('Distribución de la variable objetivo')
plt.tight_layout(); plt.show()
"""),
    md("Las clases están razonablemente balanceadas, así que *accuracy* es una métrica válida; igual se reportan precision, recall, F1 y ROC-AUC."),
    md("### 2.3 Distribución de variables numéricas"),
    code("""
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
for ax, col in zip(axes, num_cols):
    sns.histplot(data=df[df[col] != 0] if col in ('RestingBP', 'Cholesterol') else df,
                 x=col, hue=TARGET, kde=True, ax=ax, palette=['#2b8a3e', '#c92a2a'], legend=False)
    ax.set_title(col)
fig.suptitle('Distribuciones por clase (verde = normal, rojo = enfermedad)', y=1.04)
plt.tight_layout(); plt.show()
"""),
    md("### 2.4 Variables categóricas frente a la variable objetivo"),
    code("""
cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), cat_cols):
    rate = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, color='#d9480f')
    ax.set_ylim(0, 1); ax.set_title(f'% con enfermedad según {col}'); ax.set_ylabel('')
    for i, v in enumerate(rate.values):
        ax.text(i, v + .02, f'{v:.0%}', ha='center')
plt.tight_layout(); plt.show()
"""),
    md("""
Observaciones:
- **ST_Slope plana**, **angina con ejercicio** y **dolor tipo ASY** tienen tasas de enfermedad muy altas.
- Que *asintomático* (ASY) sea el grupo de mayor riesgo parece raro, pero tiene explicación: son pacientes
  que llegaron al estudio por otra razón clínica y no presentaban dolor típico (isquemia silente).
- Los hombres tienen una tasa bastante mayor que las mujeres.
"""),
    md("### 2.5 Correlaciones"),
    code("""
corr = clean(df)[NUMERIC + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, square=True)
ax.set_title('Matriz de correlación (variables numéricas)')
plt.tight_layout(); plt.show()
"""),
    md("Ninguna pareja de variables tiene una correlación tan alta como para ser redundante. Oldpeak (+) y MaxHR (−) son las numéricas más relacionadas con la enfermedad."),
    md("### 2.6 Outliers"),
    code("""
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=clean(df)[col], ax=ax, color='#e3edf7')
    ax.set_title(col); ax.set_ylabel('')
plt.tight_layout(); plt.show()

q1, q3 = df[num_cols].quantile([.25, .75]).values
iqr = q3 - q1
outliers = ((df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)).sum()
print('Outliers por criterio IQR:'); print(outliers.to_string())
"""),
    md("""
Los valores extremos (colesterol de 500+, presión de 200) son clínicamente posibles, así que **no se eliminan**.
Random Forest es robusto frente a outliers porque divide por umbrales y no por distancias.
"""),
]

TRAINING = [
    md("## 3. Preprocesamiento"),
    code("""
from sklearn.model_selection import train_test_split

data = clean(df)
data.to_csv('../data/processed/heart_clean.csv', index=False)

X = data.drop(columns=TARGET)
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pd.concat([X_train, y_train], axis=1).to_csv('../data/processed/train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('../data/processed/test.csv', index=False)
print(f'Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}')
print(f'Colesterol faltante en train: {X_train.Cholesterol.isna().sum()}')
"""),
    md("""
El preprocesamiento está dentro de un `Pipeline` de scikit-learn (`src/preprocessing.py`):

1. **Imputación**: mediana para numéricas, moda para categóricas.
2. **One-hot encoding** para las categóricas.
3. **Sin escalado**: Random Forest no lo necesita porque compara cada variable contra un umbral.

Como el imputador se ajusta solo con los datos de entrenamiento, no hay fuga de información hacia el conjunto de prueba.
"""),
    md("## 4. Entrenamiento del modelo\n### 4.1 Comparación rápida con otros algoritmos (validación cruzada, 11 variables)"),
    code("""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing import build_preprocessor

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidatos = {
    'Regresión logística': Pipeline([('prep', build_preprocessor(FEATURES_FULL)),
                                     ('sc', StandardScaler()), ('m', LogisticRegression(max_iter=1000))]),
    'Árbol de decisión': Pipeline([('prep', build_preprocessor(FEATURES_FULL)),
                                   ('m', DecisionTreeClassifier(random_state=42))]),
    'Random Forest': build_pipeline(FEATURES_FULL),
}
comparacion = {n: cross_val_score(m, X_train[FEATURES_FULL], y_train, cv=cv, scoring='accuracy')
               for n, m in candidatos.items()}
pd.DataFrame({n: {'accuracy media': s.mean(), 'desv. estándar': s.std()}
              for n, s in comparacion.items()}).T.round(3)
"""),
    md("""
Random Forest y la regresión logística quedan prácticamente empatados (≈85 % de accuracy), y ambos superan al árbol de
decisión individual (≈80 %, con más varianza entre folds). Random Forest promedia muchos árboles entrenados con muestras
y variables distintas, lo que reduce el sobreajuste de un solo árbol. Además captura interacciones no lineales sin
transformar las variables y permite explicar cada predicción, que es lo que muestra la app.
"""),
    md("### 4.2 Búsqueda de hiperparámetros (GridSearchCV)"),
    code("""
from sklearn.model_selection import GridSearchCV

param_grid = {
    'rf__n_estimators': [200, 400],
    'rf__max_depth': [None, 6, 10],
    'rf__min_samples_leaf': [1, 3, 5],
    'rf__max_features': ['sqrt', 0.5],
}
modelos, busquedas = {}, {}
for nombre, feats in [('rapido', FEATURES_QUICK), ('completo', FEATURES_FULL)]:
    gs = GridSearchCV(build_pipeline(feats, n_jobs=1), param_grid, cv=cv,
                      scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train[feats], y_train)
    busquedas[nombre] = gs
    modelos[nombre] = gs.best_estimator_
    print(f'{nombre:9s} ROC-AUC CV = {gs.best_score_:.3f} | mejores parámetros: {gs.best_params_}')
"""),
    md("## 5. Evaluación en el conjunto de prueba"),
    code("""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

FEATS = {'rapido': FEATURES_QUICK, 'completo': FEATURES_FULL}
resultados = {}
for nombre, modelo in modelos.items():
    Xt = X_test[FEATS[nombre]]
    pred = modelo.predict(Xt)
    proba = modelo.predict_proba(Xt)[:, 1]
    cv_acc = cross_val_score(modelo, X_train[FEATS[nombre]], y_train, cv=cv, scoring='accuracy')
    resultados[nombre] = {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, proba),
        'cv_accuracy_mean': cv_acc.mean(),
        'cv_accuracy_std': cv_acc.std(),
    }
tabla = pd.DataFrame(resultados).round(3)
tabla.columns = ['Rápido (7 var.)', 'Completo (11 var.)']
tabla
"""),
    code("""
for nombre, modelo in modelos.items():
    print(f'=== Modelo {nombre} ===')
    print(classification_report(y_test, modelo.predict(X_test[FEATS[nombre]]),
                                target_names=['Normal', 'Enfermedad'], digits=3))
"""),
    md("### 5.1 Matrices de confusión"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
for ax, (nombre, modelo) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, modelo.predict(X_test[FEATS[nombre]]))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax,
                xticklabels=['Normal', 'Enfermedad'], yticklabels=['Normal', 'Enfermedad'],
                annot_kws={'size': 16})
    ax.set_title(f'Modelo {nombre}'); ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
plt.tight_layout(); plt.savefig(IMG + 'confusion_matrices.png', bbox_inches='tight'); plt.show()
"""),
    md("### 5.2 Curvas ROC"),
    code("""
fig, ax = plt.subplots(figsize=(6, 5))
for nombre, modelo in modelos.items():
    proba = modelo.predict_proba(X_test[FEATS[nombre]])[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, lw=2, label=f'{nombre} (AUC = {resultados[nombre]["roc_auc"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Azar')
ax.set_xlabel('Tasa de falsos positivos'); ax.set_ylabel('Tasa de verdaderos positivos (recall)')
ax.set_title('Curvas ROC en el conjunto de prueba'); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig(IMG + 'roc_curves.png', bbox_inches='tight'); plt.show()
"""),
    md("### 5.3 Importancia de variables"),
    code("""
def importancia_por_variable(modelo, feats):
    prep, rf = modelo.named_steps['prep'], modelo.named_steps['rf']
    nombres = prep.get_feature_names_out()
    imp = pd.Series(rf.feature_importances_, index=nombres)
    # Se suman las columnas one-hot de una misma variable original.
    grupo = [next(f for f in feats if n.split('__', 1)[1] == f or n.split('__', 1)[1].startswith(f + '_'))
             for n in nombres]
    return imp.groupby(grupo).sum().sort_values()

imp_full = importancia_por_variable(modelos['completo'], FEATURES_FULL)
imp_quick = importancia_por_variable(modelos['rapido'], FEATURES_QUICK)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [11, 7]})
imp_full.plot.barh(ax=axes[0], color='#d9480f'); axes[0].set_title('Modelo completo')
imp_quick.plot.barh(ax=axes[1], color='#1c5d99'); axes[1].set_title('Modelo rápido')
for ax in axes: ax.set_xlabel('Importancia (reducción media de Gini)')
plt.tight_layout(); plt.savefig(IMG + 'feature_importance.png', bbox_inches='tight'); plt.show()
"""),
    md("""
En el modelo completo dominan **ST_Slope**, **ChestPainType**, **Oldpeak** y **ExerciseAngina**, que son justamente
las variables de la prueba de esfuerzo. Sin ellas, el modelo rápido se apoya en el tipo de dolor, la angina con ejercicio, la edad y el colesterol.
"""),
    md("### 5.4 Explicación de una predicción individual"),
    code("""
from src.predict import explain

paciente = X_test.iloc[[0]]
base, contrib = explain(modelos['completo'], paciente[FEATURES_FULL], FEATURES_FULL)
proba = modelos['completo'].predict_proba(paciente[FEATURES_FULL])[0, 1]
print(f'Probabilidad base (promedio de entrenamiento): {base:.3f}')
print(f'Base + suma de contribuciones:                 {base + sum(contrib.values()):.3f}')
print(f'predict_proba del modelo:                      {proba:.3f}   (real = {y_test.iloc[0]})')
pd.DataFrame({'valor': paciente[FEATURES_FULL].iloc[0], 'contribución': pd.Series(contrib)}) \\
  .sort_values('contribución', key=abs, ascending=False).round(3)
"""),
    md("La suma de la base más las contribuciones coincide con `predict_proba`. Esta descomposición es la que muestra la app en la página de resultado."),
    md("## 6. Serialización del modelo"),
    code("""
joblib.dump(modelos['rapido'], '../models/rf_rapido.pkl', compress=3)
joblib.dump(modelos['completo'], '../models/rf_completo.pkl', compress=3)

def limpiar_params(p):
    return {k.replace('rf__', ''): (v if v is not None else 'None') for k, v in p.items()}

metadata = {
    'trained_at': str(date.today()),
    'sklearn_version': sklearn.__version__,
    'dataset': {'n_rows': int(len(df)), 'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
                'positive_rate': round(float(y.mean()) * 100, 1)},
    'models': {n: {'features': FEATS[n], 'params': limpiar_params(busquedas[n].best_params_),
                   'test': {k: round(float(v), 4) for k, v in resultados[n].items() if not k.startswith('cv_')},
                   'cv_accuracy_mean': round(float(resultados[n]['cv_accuracy_mean']), 4),
                   'cv_accuracy_std': round(float(resultados[n]['cv_accuracy_std']), 4)}
               for n in modelos},
}
with open('../models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Verificación: el modelo cargado desde disco predice igual que el de memoria.
recargado = joblib.load('../models/rf_completo.pkl')
assert (recargado.predict(X_test) == modelos['completo'].predict(X_test[FEATURES_FULL])).all()
print('Modelos guardados y verificados en ../models/')
"""),
]


def conclusions(meta: dict) -> dict:
    r, c = meta["models"]["rapido"]["test"], meta["models"]["completo"]["test"]
    return md(f"""
## 7. Conclusiones

- El **modelo completo** obtuvo accuracy de **{c['accuracy']:.1%}**, recall de **{c['recall']:.1%}** y ROC-AUC de **{c['roc_auc']:.3f}** en el conjunto de prueba.
- El **modelo rápido**, sin datos de ECG ni prueba de esfuerzo, obtuvo accuracy de **{r['accuracy']:.1%}**, recall de **{r['recall']:.1%}** y ROC-AUC de **{r['roc_auc']:.3f}**.
- La diferencia ({(c['accuracy'] - r['accuracy']) * 100:.1f} puntos de accuracy) es el costo de no pedirle al usuario los datos clínicos. A cambio, el modelo rápido lo puede usar cualquier persona respondiendo preguntas sencillas.
- En este problema importa más el **recall** (no dejar pasar a un enfermo) que la precision. Por eso la app muestra la probabilidad con un semáforo en lugar de un sí/no.
- Limitaciones: el dataset tiene 918 pacientes de cinco hospitales de EE. UU. y Europa (décadas de 1980 y 1990), por lo que el modelo no debe usarse para diagnóstico real.
""")


# ----------------------------------------------------------------------------
# Ejecutor
# ----------------------------------------------------------------------------
def run_cells(cells: list[dict]) -> None:
    os.chdir(ROOT / "notebooks")
    sys.path.insert(0, str(ROOT))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns: dict = {"__name__": "__main__"}
    count = 0

    def fig_outputs():
        outs = []
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(buf.getvalue()).decode(),
                                  "text/plain": "<Figure>"}})
        plt.close("all")
        return outs

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        outputs = []
        current = {"buf": io.StringIO()}

        class Proxy(io.TextIOBase):
            def write(self, s):
                return current["buf"].write(s)

        stdout = Proxy()

        def flush():
            text = current["buf"].getvalue()
            if text:
                outputs.append({"output_type": "stream", "name": "stdout", "text": text})
            current["buf"] = io.StringIO()

        def show(*_, **__):
            flush()
            outputs.extend(fig_outputs())

        plt.show = show
        tree = ast.parse(cell["source"])
        last_expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                value = eval(compile(ast.Expression(last_expr.value), f"<cell {count}>", "eval"), ns) \
                    if last_expr else None
            flush()
            outputs.extend(fig_outputs())
            if value is not None:
                data = {"text/plain": repr(value)}
                if hasattr(value, "_repr_html_"):
                    data["text/html"] = value._repr_html_()
                outputs.append({"output_type": "execute_result", "execution_count": count,
                                "metadata": {}, "data": data})
        except Exception as e:
            outputs.append({"output_type": "error", "ename": type(e).__name__, "evalue": str(e),
                            "traceback": traceback.format_exc().splitlines()})
            cell["outputs"] = outputs
            raise RuntimeError(f"Error en la celda {count}:\n{cell['source']}") from e
        cell["outputs"] = outputs


def save(cells: list[dict]) -> None:
    for c in cells:  # formato estándar: source como lista de líneas
        c["source"] = c["source"].splitlines(keepends=True)
        for o in c.get("outputs", []):
            if "text" in o:
                o["text"] = o["text"].splitlines(keepends=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook guardado en {NB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["eda", "full"], default="full")
    args = parser.parse_args()
    cells = [dict(c) for c in (EDA if args.stage == "eda" else EDA + TRAINING)]
    run_cells(cells)
    if args.stage == "full":
        meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
        cells.append(conclusions(meta))
    save(cells)
