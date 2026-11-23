# Corazón RF: Predicción de enfermedad cardiaca con Random Forest

## Descripción
Aplicación web que estima la probabilidad de que una persona tenga enfermedad cardiaca. Tiene dos modos:

- **Tamizaje rápido**: un asistente de 4 pasos con preguntas en lenguaje sencillo (7 variables). Lo puede usar cualquier persona; si no conoce un dato, marca "No sé".
- **Evaluación clínica**: las 11 variables del dataset, incluidas las del electrocardiograma. Pensada para personal de salud.

El resultado se muestra como un semáforo de riesgo y una lista de los factores que más influyeron en la predicción. La app también lee exámenes de laboratorio en PDF para precargar el formulario.

> Proyecto académico. No reemplaza el diagnóstico de un profesional de la salud.

## Demostración
[Abrir la aplicación desplegada](https://corazon-rf.onrender.com)

> El plan gratuito de Render apaga el servicio tras 15 minutos sin uso; la primera carga puede tardar cerca de un minuto.

## Algoritmo utilizado
**Random Forest** (`sklearn.ensemble.RandomForestClassifier`).

Por qué es apropiado:
- Mezcla variables numéricas y categóricas sin necesidad de escalado.
- Es robusto frente a outliers (colesterol y presión tienen valores extremos reales).
- Captura interacciones no lineales entre síntomas y resultados del ECG.
- Permite explicar cada predicción: la contribución de cada variable se calcula recorriendo los árboles (método de Saabas).

Hiperparámetros elegidos con `GridSearchCV` (5 folds, métrica ROC-AUC).

### Métricas en el conjunto de prueba (20 %, 184 pacientes)

| Métrica | Rápido (7 variables) | Completo (11 variables) |
|---|---|---|
| Accuracy | 80.4 % | **89.7 %** |
| Precision | 83.7 % | **88.8 %** |
| Recall | 80.4 % | **93.1 %** |
| F1-score | 82.0 % | **90.9 %** |
| ROC-AUC | 0.899 | **0.928** |
| Accuracy CV (5 folds) | 78.9 % ± 4.0 | 85.4 % ± 3.6 |

## Dataset
- **Fuente:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (fedesoriano, Kaggle), que combina cinco bases de datos del UCI Machine Learning Repository.
- **Tamaño:** 918 registros, 11 variables y la variable objetivo `HeartDisease`.
- **Variables:** `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`.
- **Preprocesamiento:** los valores de colesterol y presión iguales a 0 se tratan como datos faltantes y se imputan con la mediana; las categóricas se codifican con one-hot.

## Valor agregado
| Funcionalidad | Qué resuelve |
|---|---|
| Dos modelos (rápido / completo) | El usuario común no conoce los datos del ECG. El modelo rápido usa solo lo que una persona puede responder, y el notebook compara cuánta precisión se pierde. |
| Asistente en lenguaje sencillo con "No sé" | En lugar de pedir "ChestPainType: TA/ATA/NAP/ASY", se hacen tres preguntas sobre el dolor y se traducen al código con los criterios clásicos de angina. |
| Explicación del resultado | Probabilidad con semáforo y aporte de cada variable en puntos porcentuales. |
| Lector de PDF (alcance limitado) | Lee PDFs digitales de laboratorio y precarga edad, sexo, presión, colesterol y glucosa. No procesa fotos ni escaneos: el OCR no cabe en los 512 MB del plan gratuito y fallaría con imágenes de baja calidad. |

## Instalación local

### Requisitos
- Python 3.12 o superior (recomendado 3.13)
- pip

### Pasos
```bash
git clone https://github.com/Juan-Supelano/Corazon_RF.git
cd Corazon_RF
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar la aplicación
```bash
python app/app.py
# abrir http://127.0.0.1:5000
```

### Volver a entrenar los modelos
Abrir y ejecutar `notebooks/01_training.ipynb` (genera `models/*.pkl`, `models/metadata.json` y las figuras de `app/static/img/`).

## Uso
1. **Inicio**: elegir el modo.
2. **Tamizaje rápido**: responder los 4 pasos (sobre ti, síntomas, exámenes, revisión) y pulsar *Calcular riesgo*.
3. **Evaluación clínica**: llenar las 11 variables (el botón *Cargar paciente de ejemplo* sirve para la demo).
4. **Leer PDF**: subir un examen de laboratorio digital (hay un PDF de ejemplo descargable en la misma página), revisar los valores encontrados y continuar al tamizaje.
5. **El modelo**: métricas, matrices de confusión, curvas ROC e importancia de variables.

API para integraciones:
```bash
curl -X POST https://corazon-rf.onrender.com/api/predict -H "Content-Type: application/json" \
  -d '{"mode": "rapido", "data": {"Age": 58, "Sex": "M", "ChestPainType": "ASY", "ExerciseAngina": "Y", "RestingBP": 140, "Cholesterol": null, "FastingBS": 0}}'
```

## Despliegue en Render
1. En Render: **New → Blueprint** y seleccionar este repositorio (usa `render.yaml`), o **New → Web Service** con:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app.app:app --workers 1 --threads 4 --timeout 60`
   - Instance type: **Free**
2. La versión de Python se toma de `.python-version` (3.13).
3. Cada `git push` a `main` vuelve a desplegar la app.

## Estructura del proyecto
```
Corazon_RF/
├── README.md
├── requirements.txt
├── render.yaml              # configuración de despliegue en Render
├── .python-version
├── data/
│   ├── raw/heart.csv        # dataset original
│   └── processed/           # datos limpios y división train/test
├── notebooks/
│   └── 01_training.ipynb    # EDA, entrenamiento, evaluación y serialización
├── models/
│   ├── rf_rapido.pkl        # Random Forest con 7 variables
│   ├── rf_completo.pkl      # Random Forest con 11 variables
│   └── metadata.json        # métricas e hiperparámetros
├── src/
│   ├── preprocessing.py     # limpieza y pipeline de scikit-learn
│   ├── predict.py           # predicción y explicación por variable
│   └── pdf_reader.py        # lector de exámenes en PDF
├── app/
│   ├── app.py               # aplicación Flask
│   ├── templates/           # páginas HTML (Jinja2)
│   └── static/              # CSS, JS, figuras y PDF de ejemplo
├── scripts/                 # generación del notebook y del PDF de ejemplo
└── docs/                    # presentación final
```

## Autores
- **Juan Supelano** ([@Juan-Supelano](https://github.com/Juan-Supelano)): estructura del repositorio, entrenamiento de los modelos, asistente de tamizaje y despliegue.
- **pipereinaa** ([@pipereinaa](https://github.com/pipereinaa)): análisis exploratorio, módulo de predicción y explicación, página de resultado.
- **Edwinvas178-bot** ([@Edwinvas178-bot](https://github.com/Edwinvas178-bot)): preprocesamiento, aplicación Flask base y diseño, lector de PDF.

## Licencia
MIT License. Ver [LICENSE](LICENSE).
