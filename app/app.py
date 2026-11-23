"""Aplicación web Flask: Corazón RF.

Páginas:
  /            Inicio
  /rapido      Tamizaje rápido (asistente paso a paso, 7 variables)
  /clinico     Evaluación clínica (11 variables, para personal de salud)
  /resultado   Resultado con semáforo y factores que influyeron
  /pdf         Lector de examen de laboratorio en PDF
  /modelo      Métricas y comparación de los dos modelos
  /api/predict API JSON
"""
from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pdf_reader import read_lab_pdf  # noqa: E402
from src.predict import LABELS, load_metadata, predict  # noqa: E402
from src.preprocessing import chest_pain_from_answers  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # PDFs de máximo 5 MB

# Rangos válidos para los campos numéricos.
RANGES = {
    "Age": (18, 100),
    "RestingBP": (70, 250),
    "Cholesterol": (80, 700),
    "MaxHR": (60, 220),
    "Oldpeak": (-3.0, 7.0),
}


class FormError(ValueError):
    pass


def _number(form, name: str, feature: str, optional: bool = False, cast=float):
    raw = (form.get(name) or "").strip().replace(",", ".")
    if raw == "" or form.get(f"{name}_nose"):
        if optional:
            return None
        raise FormError(f"Falta el campo: {LABELS[feature]}")
    try:
        value = cast(float(raw))
    except ValueError:
        raise FormError(f"{LABELS[feature]} debe ser un número") from None
    lo, hi = RANGES[feature]
    if not lo <= value <= hi:
        raise FormError(f"{LABELS[feature]} debe estar entre {lo} y {hi}")
    return value


def _choice(form, name: str, options: list[str], feature: str):
    value = form.get(name)
    if value not in options:
        raise FormError(f"Selecciona una opción en: {LABELS.get(feature, feature)}")
    return value


def parse_quick(form) -> dict:
    """Traduce las respuestas del asistente a las variables del modelo rápido."""
    yes = lambda n: form.get(n) == "si"  # noqa: E731
    has_pain = _choice(form, "has_pain", ["si", "no"], "ChestPainType") == "si"
    if has_pain:
        for q in ("central", "exertion", "relief"):
            _choice(form, q, ["si", "no"], "ChestPainType")
    fbs = _choice(form, "fbs", ["si", "no", "nose"], "FastingBS")
    return {
        "Age": _number(form, "age", "Age", cast=int),
        "Sex": _choice(form, "sex", ["M", "F"], "Sex"),
        "ChestPainType": chest_pain_from_answers(
            has_pain, yes("central"), yes("exertion"), yes("relief")),
        "ExerciseAngina": "Y" if has_pain and yes("exertion") else "N",
        "RestingBP": _number(form, "bp", "RestingBP", optional=True, cast=int),
        "Cholesterol": _number(form, "chol", "Cholesterol", optional=True, cast=int),
        "FastingBS": None if fbs == "nose" else int(fbs == "si"),
    }


def parse_clinical(form) -> dict:
    fbs = _choice(form, "fbs", ["1", "0"], "FastingBS")
    return {
        "Age": _number(form, "age", "Age", cast=int),
        "Sex": _choice(form, "sex", ["M", "F"], "Sex"),
        "ChestPainType": _choice(form, "cp", ["TA", "ATA", "NAP", "ASY"], "ChestPainType"),
        "RestingBP": _number(form, "bp", "RestingBP", optional=True, cast=int),
        "Cholesterol": _number(form, "chol", "Cholesterol", optional=True, cast=int),
        "FastingBS": int(fbs),
        "RestingECG": _choice(form, "ecg", ["Normal", "ST", "LVH"], "RestingECG"),
        "MaxHR": _number(form, "maxhr", "MaxHR", cast=int),
        "ExerciseAngina": _choice(form, "exang", ["Y", "N"], "ExerciseAngina"),
        "Oldpeak": _number(form, "oldpeak", "Oldpeak"),
        "ST_Slope": _choice(form, "slope", ["Up", "Flat", "Down"], "ST_Slope"),
    }


@app.get("/")
def home():
    return render_template("index.html", active="inicio")


@app.get("/rapido")
def rapido():
    # Valores que llegan precargados desde el lector de PDF.
    prefill = {k: request.args.get(k, "") for k in ("age", "sex", "bp", "chol", "fbs")}
    return render_template("rapido.html", active="rapido", prefill=prefill,
                           from_pdf=request.args.get("from") == "pdf")


@app.get("/clinico")
def clinico():
    return render_template("clinico.html", active="clinico", form={})


@app.post("/resultado")
def resultado():
    mode = request.form.get("mode")
    try:
        if mode == "rapido":
            data = parse_quick(request.form)
        elif mode == "clinico":
            data = parse_clinical(request.form)
        else:
            return redirect(url_for("home"))
    except FormError as e:
        template = "rapido.html" if mode == "rapido" else "clinico.html"
        ctx = {"prefill": {}} if mode == "rapido" else {"form": request.form}
        return render_template(template, active=mode, error=str(e), **ctx), 400

    result = predict("rapido" if mode == "rapido" else "completo", data)
    return render_template("resultado.html", active=mode, r=result)


@app.route("/pdf", methods=["GET", "POST"])
def pdf():
    if request.method == "GET":
        return render_template("pdf.html", active="pdf")
    file = request.files.get("archivo")
    if not file or not file.filename:
        return render_template("pdf.html", active="pdf", error="Selecciona un archivo PDF."), 400
    if not file.filename.lower().endswith(".pdf"):
        return render_template("pdf.html", active="pdf",
                               error="Solo se aceptan archivos .pdf (no imágenes)."), 400
    out = read_lab_pdf(file.read())  # se procesa en memoria, no se guarda en disco
    if not out["ok"]:
        return render_template("pdf.html", active="pdf", error=out["error"]), 422

    fields = out["fields"]
    params = {"from": "pdf"}
    keymap = {"Age": "age", "Sex": "sex", "RestingBP": "bp", "Cholesterol": "chol"}
    for feat, key in keymap.items():
        if feat in fields:
            params[key] = fields[feat]["value"]
    if "FastingBS" in fields:
        params["fbs"] = "si" if fields["FastingBS"]["value"] else "no"
    return render_template("pdf.html", active="pdf", fields=fields, labels=LABELS,
                           next_url=url_for("rapido", **params), filename=file.filename,
                           truncated=out["truncated"])


@app.errorhandler(413)
def too_large(_):
    return render_template("pdf.html", active="pdf", error="El archivo supera 5 MB."), 413


@app.get("/modelo")
def modelo():
    return render_template("modelo.html", active="modelo", meta=load_metadata())


@app.post("/api/predict")
def api_predict():
    """Ejemplo: {"mode": "rapido", "data": {"Age": 55, "Sex": "M", ...}}"""
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "rapido")
    try:
        result = predict(mode, payload.get("data", {}))
    except Exception as e:  # entrada inválida
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
