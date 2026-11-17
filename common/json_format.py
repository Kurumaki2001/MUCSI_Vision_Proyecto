"""Formato JSON común a los dos enfoques.

El mismo esquema sirve para las anotaciones (la verdad) y para los resultados de cada enfoque.
Coordenadas en píxeles de la imagen, con (0, 0) en la esquina superior izquierda; cajas como
[x0, y0, x1, y1].

Ejemplo de anotación:
    {
      "image": "photo_00000.jpg", "width": 1600, "height": 2133,
      "dots":    [{"id": 0, "x": 812.4, "y": 300.1, "box": [806.0, 293.7, 818.8, 306.5]}, ...],
      "numbers": [{"id": 0, "value": 1, "box": [822.0, 290.2, 834.5, 303.0]}, ...],
      "pairs":   [{"dot": 0, "number": 0, "distance": 10.2}, ...],
      "unpaired_dots": [], "unpaired_numbers": [],
      "metadata": {...}
    }

Diferencias entre anotaciones y resultados:
  - Los resultados añaden "confidence" a cada punto y a cada número, y "times_ms" con el total y
    el desglose por etapas. Las anotaciones no llevan ni confianzas ni tiempos.
  - En los resultados, "value" es null si el número no se ha podido leer.
  - La "box" de los puntos es opcional: el generador la escribe porque YOLO la necesita para
    entrenar, pero una anotación real hecha a mano puede llevar solo el centro.
  - En una anotación real mínima, los números pueden ir sin caja.
  - "metadata" es opcional (por ejemplo, cómo se generó una imagen sintética).
"""

import json
from pathlib import Path

REQUIRED_KEYS = ("image", "width", "height", "dots", "numbers", "pairs", "unpaired_dots", "unpaired_numbers")


def create(image, width, height, dots, numbers, pairs, unpaired_dots=(), unpaired_numbers=(),
           times_ms=None, metadata=None):
    """Construye un diccionario con el formato común y lo valida."""
    data = {
        "image": image,
        "width": int(width),
        "height": int(height),
        "dots": list(dots),
        "numbers": list(numbers),
        "pairs": list(pairs),
        "unpaired_dots": list(unpaired_dots),
        "unpaired_numbers": list(unpaired_numbers),
    }
    if times_ms is not None:
        data["times_ms"] = times_ms
    if metadata:
        data["metadata"] = metadata
    validate(data)
    return data


def validate(data):
    """Comprueba la estructura y la coherencia básica. Lanza ValueError con todos los problemas."""
    problems = [f"falta la clave '{key}'" for key in REQUIRED_KEYS if key not in data]
    if problems:
        raise ValueError("JSON no válido: " + "; ".join(problems))

    dot_ids = [dot["id"] for dot in data["dots"]]
    number_ids = [number["id"] for number in data["numbers"]]
    if len(set(dot_ids)) != len(dot_ids):
        problems.append("hay ids de punto repetidos")
    if len(set(number_ids)) != len(number_ids):
        problems.append("hay ids de número repetidos")

    for item in data["dots"] + data["numbers"]:
        box = item.get("box")
        if box is not None and (len(box) != 4 or box[0] > box[2] or box[1] > box[3]):
            problems.append(f"caja mal formada: {box}")

    for pair in data["pairs"]:
        if pair["dot"] not in dot_ids or pair["number"] not in number_ids:
            problems.append(f"pareja con ids inexistentes: {pair}")
    paired_dots = [pair["dot"] for pair in data["pairs"]]
    if len(set(paired_dots)) != len(paired_dots):
        problems.append("un punto aparece en más de una pareja")
    paired_numbers = [pair["number"] for pair in data["pairs"]]
    if len(set(paired_numbers)) != len(paired_numbers):
        problems.append("un número aparece en más de una pareja")

    if problems:
        raise ValueError("JSON no válido: " + "; ".join(problems))


def save(data, path):
    """Escribe el JSON en UTF-8."""
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path):
    """Lee un JSON con el formato común."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
