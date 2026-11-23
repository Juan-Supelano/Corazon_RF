"""Lector de exámenes de laboratorio en PDF (alcance limitado a propósito).

Qué hace:
  - Lee solo PDFs digitales (con texto seleccionable), máximo 10 páginas.
  - Busca con expresiones regulares: edad, sexo, presión arterial,
    colesterol total y glucosa en ayunas.
  - Devuelve sugerencias; el usuario las revisa antes de predecir.

Qué NO hace (y por qué):
  - No lee fotos ni PDFs escaneados: requeriría OCR, que no cabe en los
    512 MB de RAM del plan gratuito de Render y falla con imágenes borrosas.
  - No extrae ECG, MaxHR, Oldpeak ni ST_Slope: esos datos salen de una
    prueba de esfuerzo y no aparecen en un examen de sangre normal.
"""
from __future__ import annotations

import io
import re
import unicodedata

from pypdf import PdfReader

MAX_PAGES = 10


def _normalize(text: str) -> str:
    """Minúsculas y sin tildes, para que las regex sean más simples."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[ \t]+", " ", text.lower())


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _snippet(text: str, match: re.Match) -> str:
    start, end = max(match.start() - 10, 0), min(match.end() + 10, len(text))
    return text[start:end].replace("\n", " ").strip()


NUM = r"(\d{1,3}(?:[.,]\d+)?)"
SEP = r"\s*[:=\-]?\s*"

PATTERNS = {
    "Age": [rf"\b(?:edad|age){SEP}(\d{{1,3}})\b"],
    "Sex": [rf"\b(?:sexo|sex|genero){SEP}(masculino|femenino|hombre|mujer|male|female|m|f)\b"],
    "RestingBP": [
        rf"\b(?:presion arterial|tension arterial|blood pressure|p\.?a\.?|t\.?a\.?){SEP}(\d{{2,3}})\s*/\s*\d{{2,3}}",
        rf"\b(?:presion|tension) (?:arterial )?sistolica{SEP}(\d{{2,3}})",
    ],
    "Cholesterol": [
        rf"\b(?:colesterol total|total cholesterol|cholesterol, total){SEP}{NUM}",
        rf"\b(?:colesterol|cholesterol)(?!\s*(?:hdl|ldl|vldl|de alta|de baja)){SEP}{NUM}",
    ],
    "Glucose": [
        rf"\b(?:glucosa|glicemia|glucemia|glucose)(?: en ayunas| basal| fasting| pre)?{SEP}{NUM}",
    ],
}


def extract_text(file_bytes: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = reader.pages[:MAX_PAGES]
    text = "\n".join((p.extract_text() or "") for p in pages)
    return text, len(reader.pages)


def parse_lab_text(raw_text: str) -> dict:
    """Busca los campos en el texto y devuelve valores + evidencia."""
    text = _normalize(raw_text)
    found = {}
    for field, patterns in PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                found[field] = (m.group(1), _snippet(text, m))
                break

    result = {}
    if "Age" in found:
        age = int(found["Age"][0])
        if 18 <= age <= 100:
            result["Age"] = {"value": age, "display": f"{age} años", "evidence": found["Age"][1]}
    if "Sex" in found:
        v = found["Sex"][0]
        sex = "F" if v in ("f", "femenino", "mujer", "female") else "M"
        result["Sex"] = {"value": sex, "display": "Mujer" if sex == "F" else "Hombre",
                         "evidence": found["Sex"][1]}
    if "RestingBP" in found:
        bp = int(found["RestingBP"][0])
        if 70 <= bp <= 250:
            result["RestingBP"] = {"value": bp, "display": f"{bp} mmHg (sistólica)",
                                   "evidence": found["RestingBP"][1]}
    if "Cholesterol" in found:
        chol = _num(found["Cholesterol"][0])
        if chol < 20:  # viene en mmol/L
            chol = chol * 38.67
        chol = round(chol)
        if 80 <= chol <= 700:
            result["Cholesterol"] = {"value": chol, "display": f"{chol} mg/dL",
                                     "evidence": found["Cholesterol"][1]}
    if "Glucose" in found:
        glu = _num(found["Glucose"][0])
        if glu < 35:  # viene en mmol/L
            glu = glu * 18
        glu = round(glu)
        if 40 <= glu <= 600:
            fbs = int(glu > 120)
            result["FastingBS"] = {"value": fbs,
                                   "display": f"Glucosa {glu} mg/dL → {'Sí' if fbs else 'No'} supera 120",
                                   "evidence": found["Glucose"][1]}
    return result


def read_lab_pdf(file_bytes: bytes) -> dict:
    """Punto de entrada usado por la app."""
    try:
        text, n_pages = extract_text(file_bytes)
    except Exception:
        return {"ok": False, "error": "No se pudo abrir el archivo. ¿Es un PDF válido?"}
    if len(text.strip()) < 20:
        return {"ok": False, "error": ("El PDF no tiene texto seleccionable (parece escaneado o "
                                       "una foto). Ingresa los datos manualmente.")}
    fields = parse_lab_text(text)
    return {"ok": True, "fields": fields, "pages": n_pages,
            "truncated": n_pages > MAX_PAGES}
