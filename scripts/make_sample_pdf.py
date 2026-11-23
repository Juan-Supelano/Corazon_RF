"""Genera un examen de laboratorio FICTICIO para probar el lector de PDF.

Uso: python scripts/make_sample_pdf.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "ejemplo_laboratorio.pdf"

c = canvas.Canvas(str(OUT), pagesize=letter)
w, h = letter
c.setFont("Helvetica-Bold", 15)
c.drawString(60, h - 70, "Laboratorio Clínico de Ejemplo")
c.setFont("Helvetica", 9)
c.drawString(60, h - 85, "DOCUMENTO FICTICIO - generado para probar Corazón RF")
c.line(60, h - 95, w - 60, h - 95)

c.setFont("Helvetica", 11)
y = h - 125
for line in ["Paciente: Paciente de Prueba", "Edad: 58 años        Sexo: Masculino",
             "Fecha de toma: 2026-11-20", "Presión arterial: 145/92 mmHg"]:
    c.drawString(60, y, line)
    y -= 18

y -= 10
c.setFont("Helvetica-Bold", 11)
c.drawString(60, y, "Examen")
c.drawString(300, y, "Resultado")
c.drawString(420, y, "Referencia")
c.setFont("Helvetica", 11)
rows = [("Glucosa en ayunas", "132 mg/dL", "70 - 100"),
        ("Colesterol total", "245 mg/dL", "< 200"),
        ("Colesterol HDL", "41 mg/dL", "> 40"),
        ("Colesterol LDL", "158 mg/dL", "< 130"),
        ("Triglicéridos", "230 mg/dL", "< 150")]
for name, val, ref in rows:
    y -= 18
    c.drawString(60, y, name)
    c.drawString(300, y, val)
    c.drawString(420, y, ref)
c.save()
print(f"PDF creado en {OUT}")
