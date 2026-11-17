"""Evaluación común a los dos enfoques: compara los JSON de resultados con los de anotación.

Métricas, sumando los aciertos y fallos de todas las imágenes:
  - Puntos: precisión, recall y F1. Cada punto predicho se asigna como mucho a un punto real
    (algoritmo húngaro sobre la distancia entre centros) y acierta si está a menos de
    dot_tolerance_px píxeles.
  - Números: % de números reales leídos bien, es decir, con una caja predicha que se solapa con
    la suya y con el mismo valor (también uno a uno). Solo se calcula si la anotación tiene las
    cajas de los números. El solape se mide como intersección / área de la caja menor
    (>= number_overlap) y no con IoU: el OCR da cajas algo más holgadas que las reales y, con
    IoU, una lectura correcta podía contar como fallo. Una caja que fusiona dos números sigue
    contando como fallo, porque su valor no coincide.
  - Parejas: % de puntos reales cuyo número asignado tiene el valor correcto. Para cada punto real
    se busca su punto predicho y se mira el valor del número que el sistema le ha asignado. No
    necesita cajas, así que vale también para las anotaciones reales mínimas.
  - Imágenes perfectas: % de imágenes sin puntos perdidos ni sobrantes y con todas las parejas bien.
  - Tiempo medio por imagen, del campo "times_ms" de los resultados.

Uso, desde la carpeta del proyecto:
    python -m common.evaluation --annotations data/synthetic/test/annotations --results <carpeta> [<carpeta> ...]

Los ficheros se emparejan por nombre. Con varias carpetas de resultados (por ejemplo, una por
perturbación en la prueba de robustez) se saca una fila por carpeta. El informe de cada carpeta,
con el detalle por imagen, se guarda en <carpeta>/evaluation.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import json_format
from common.config import load_config
from common.matching import assign

REPORT_NAME = "evaluation.json"
COUNTERS = ("true_dots", "missed_dots", "extra_dots", "numbers_total", "numbers_read", "pairs_total", "pairs_correct")


def box_overlap(boxes_a, boxes_b):
    """Solape entre cada caja de boxes_a y cada una de boxes_b: área de la intersección / área de la menor.

    Vale 1 si una caja contiene a la otra y 0 si no se tocan.
    """
    a = np.asarray(boxes_a, dtype=float)[:, None, :]
    b = np.asarray(boxes_b, dtype=float)[None, :, :]
    width = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    height = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return width * height / np.maximum(np.minimum(area_a, area_b), 1e-9)


def match_dots(real_dots, predicted_dots, tolerance):
    """{id del punto real: id del punto predicho}, uno a uno y a menos de 'tolerance' píxeles."""
    if not real_dots or not predicted_dots:
        return {}
    real = np.array([[dot["x"], dot["y"]] for dot in real_dots], dtype=float)
    predicted = np.array([[dot["x"], dot["y"]] for dot in predicted_dots], dtype=float)
    distances = np.hypot(real[:, None, 0] - predicted[None, :, 0], real[:, None, 1] - predicted[None, :, 1])
    return {real_dots[i]["id"]: predicted_dots[j]["id"] for i, j in assign(distances, tolerance)}


def assigned_values(data):
    """{id de punto: valor del número con el que está emparejado} (None si el número no se leyó)."""
    values = {number["id"]: number.get("value") for number in data["numbers"]}
    return {pair["dot"]: values[pair["number"]] for pair in data["pairs"]}


def evaluate_image(annotation, result, cfg):
    """Aciertos y fallos de una imagen, como contadores que después se suman."""
    dot_matches = match_dots(annotation["dots"], result["dots"], cfg["dot_tolerance_px"])
    counts = {"true_dots": len(dot_matches),
              "missed_dots": len(annotation["dots"]) - len(dot_matches),
              "extra_dots": len(result["dots"]) - len(dot_matches)}

    # Números: solo si la anotación tiene la caja de todos ellos.
    real_numbers = annotation["numbers"]
    counts["numbers_total"] = counts["numbers_read"] = 0
    if real_numbers and all(number.get("box") for number in real_numbers):
        counts["numbers_total"] = len(real_numbers)
        predicted = [number for number in result["numbers"] if number.get("box")]
        if predicted:
            overlap = box_overlap([number["box"] for number in real_numbers], [number["box"] for number in predicted])
            matches = assign(1 - overlap, 1 - cfg["number_overlap"])
            counts["numbers_read"] = sum(real_numbers[i]["value"] == predicted[j].get("value") for i, j in matches)

    # Parejas: el número que el sistema asigna a cada punto real (a través de su punto predicho).
    real_values = assigned_values(annotation)
    predicted_values = assigned_values(result)
    evaluable = [dot["id"] for dot in annotation["dots"] if real_values.get(dot["id"]) is not None]
    counts["pairs_total"] = len(evaluable)
    counts["pairs_correct"] = sum(1 for dot_id in evaluable if dot_id in dot_matches
                                  and predicted_values.get(dot_matches[dot_id]) == real_values[dot_id])

    counts["perfect"] = (counts["missed_dots"] == 0 and counts["extra_dots"] == 0
                         and counts["pairs_correct"] == counts["pairs_total"])
    counts["time_ms"] = result.get("times_ms", {}).get("total")
    return counts


def percentage(part, total):
    """Porcentaje con dos decimales, o None si no hay nada que medir."""
    return round(100 * part / total, 2) if total else None


def summarize(per_image):
    """Suma los contadores de todas las imágenes y calcula las métricas finales (en %)."""
    totals = {key: sum(counts[key] for counts in per_image) for key in COUNTERS}
    precision = percentage(totals["true_dots"], totals["true_dots"] + totals["extra_dots"])
    recall = percentage(totals["true_dots"], totals["true_dots"] + totals["missed_dots"])
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 2) if precision + recall else 0.0
    times = [counts["time_ms"] for counts in per_image if counts["time_ms"] is not None]
    return {
        "images": len(per_image),
        "dot_precision": precision,
        "dot_recall": recall,
        "dot_f1": f1,
        "numbers_read": percentage(totals["numbers_read"], totals["numbers_total"]),
        "pairs_correct": percentage(totals["pairs_correct"], totals["pairs_total"]),
        "perfect_images": percentage(sum(counts["perfect"] for counts in per_image), len(per_image)),
        "mean_time_ms": round(float(np.mean(times)), 1) if times else None,
        "totals": totals,
    }


def evaluate_folder(annotations_folder, results_folder, cfg):
    """Evalúa todos los resultados de una carpeta. Devuelve (resumen, detalle por imagen).

    Si falta el resultado de alguna imagen, cuenta como si el sistema no hubiera encontrado nada.
    """
    per_image, missing = [], 0
    for annotation_path in sorted(Path(annotations_folder).glob("*.json")):
        if annotation_path.name == REPORT_NAME:
            continue
        annotation = json_format.load(annotation_path)
        result_path = Path(results_folder) / annotation_path.name
        if result_path.exists():
            result = json_format.load(result_path)
        else:
            missing += 1
            result = {"dots": [], "numbers": [], "pairs": []}
        counts = evaluate_image(annotation, result, cfg)
        per_image.append({"image": annotation["image"], **counts})
    if not per_image:
        raise SystemExit(f"No hay anotaciones en {annotations_folder}.")
    summary = summarize(per_image)
    summary["missing_results"] = missing
    return summary, per_image


def print_table(rows):
    """Tabla con una fila por carpeta de resultados."""
    def show(value):
        return "-" if value is None else f"{value:.2f}"

    print(f"{'resultados':30s} {'imágenes':>8s} {'P puntos':>9s} {'R puntos':>9s} {'F1':>7s} "
          f"{'números':>8s} {'parejas':>8s} {'perfectas':>9s} {'ms/img':>8s}")
    for name, s in rows:
        print(f"{name[:30]:30s} {s['images']:8d} {show(s['dot_precision']):>9s} {show(s['dot_recall']):>9s} "
              f"{show(s['dot_f1']):>7s} {show(s['numbers_read']):>8s} {show(s['pairs_correct']):>8s} "
              f"{show(s['perfect_images']):>9s} {show(s['mean_time_ms']):>8s}")
    print("Porcentajes, salvo ms/img. '-' = no se puede calcular (p. ej., números sin caja en la anotación).")


def main():
    parser = argparse.ArgumentParser(description="Compara resultados con anotaciones (formato común).")
    parser.add_argument("--annotations", required=True, help="carpeta con los JSON de anotación")
    parser.add_argument("--results", required=True, nargs="+", help="una o varias carpetas con JSON de resultados")
    args = parser.parse_args()

    cfg = load_config()["evaluation"]
    rows = []
    for folder in args.results:
        summary, per_image = evaluate_folder(args.annotations, folder, cfg)
        if summary["missing_results"]:
            print(f"Aviso: en {folder} faltan {summary['missing_results']} resultados (cuentan como vacíos).")
        report = {"annotations": str(args.annotations), "results": str(folder), "config": cfg,
                  "summary": summary, "per_image": per_image}
        (Path(folder) / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append((Path(folder).name, summary))
    print_table(rows)


if __name__ == "__main__":
    main()
