"""Inferencia del enfoque de visión clásica: solo OpenCV y reglas, sin redes ni entrenamiento.

Pasos para cada foto (el tiempo de cada uno queda en el JSON):
  1. paper: la hoja se separa de la mesa (HSV + Otsu + contornos; ClassicVision/preprocess.py).
  2. ink: tinta frente a papel, corrigiendo la iluminación (cierre morfológico = fondo; la
     oscuridad es 1 - foto / fondo) y umbralizando.
  3. dots: puntos = manchas redondas, rellenas y del tamaño típico (contornos y reglas de forma;
     los puntos pegados a su número se separan con una apertura). ClassicVision/dots.py.
  4. ocr: el resto de la tinta se separa en caracteres, se agrupan en números y cada dígito se
     reconoce comparándolo con plantillas de dígitos (ClassicVision/digits.py), probando la foto
     derecha y girada 90°, 180° y 270°. Si la lectura recupera poco de la secuencia 1..N (números
     impresos muy claros), se repite con umbrales más bajos o con la oscuridad suavizada y se
     queda la mejor (retry_faint_numbers).
  5. aruco (si aruco.enabled): hoja de marcadores -> coordenadas en mm (common/aruco_coords.py).
     Lo que cae dentro de un marcador se descarta antes de emparejar.
  6. matching: algoritmo húngaro con la distancia punto-rectángulo (common/matching.py), igual
     que en el enfoque de visión artificial, y diagnóstico de la secuencia 1..N.

Las salidas son las mismas que las de ArtificialVision/pipeline.py, así que los dos enfoques se
evalúan con el mismo script (common/evaluation.py): por foto, <nombre>.json (formato común),
<nombre>.jpg (dibujo) y, si se encuentra la hoja de ArUco, <nombre>_mm.csv.

Uso, desde la carpeta del proyecto:
    python -m ClassicVision.pipeline data/real
    python -m ClassicVision.pipeline foto.jpg --join
    python -m ClassicVision.pipeline data/synthetic/test/images --output <carpeta> --debug
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ClassicVision.digits import build_templates, read_numbers_any_orientation, reading_score, template_fonts
from ClassicVision.dots import detect_dots, erase_dots
from ClassicVision.preprocess import darkness, ink_mask, paper_mask
from common import json_format
from common.aruco_coords import ArucoReference, add_mm_coordinates, describe_aruco, draw_reference, save_path_csv
from common.config import load_config, project_path
from common.drawing import draw_annotation
from common.images import check_unique_image_stems, read_image, write_image
from common.matching import match_dots_numbers, max_pair_distance, sequence_check

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class Pipeline:
    """Prepara las plantillas una sola vez y procesa fotos con process()."""

    def __init__(self, config):
        self.cfg = config["classic"]
        self.max_distance_factor = config["matching"]["max_distance_factor"]
        self.templates = build_templates(template_fonts(self.cfg["template_fonts"]))
        self.aruco_cfg = config.get("aruco") or {}
        self.aruco = ArucoReference(self.aruco_cfg) if self.aruco_cfg.get("enabled") else None
        self.last_reference = None  # hoja de ArUco de la última foto (para dibujarla)
        self.last_masks = None      # máscaras intermedias de la última foto (para --debug)

    def retry_faint_numbers(self, numbers, rotation, dark, paper, dots):
        """Segundo intento de lectura para fichas con los números muy claros.

        Si la primera lectura recupera menos de retry_below de la secuencia 1..N (N = puntos), se
        vuelve a separar la tinta del texto con los ajustes de cada entrada de retries: umbrales
        más bajos (para no perder las partes claras de los trazos) o la oscuridad suavizada con un
        filtro Gaussiano (une los trozos de un trazo roto). Se queda la lectura con mejor
        reading_score. Si la primera ya esta bien, no se toca nada
        Devuelve (números, giro, número de intento elegido: 0 = el normal).
        """
        expected = len(dots)
        best, best_rotation, best_attempt = numbers, rotation, 0
        if reading_score(numbers, expected)[0] >= self.cfg["retry_below"] * expected:
            return best, best_rotation, best_attempt
        for attempt, variant in enumerate(self.cfg.get("retries", []), start=1):
            settings = {**self.cfg, **variant}
            source = cv2.GaussianBlur(dark, (0, 0), variant["blur"]) if variant.get("blur") else dark
            text_ink = erase_dots(ink_mask(source, paper, settings), dots, self.cfg)
            candidate, candidate_rotation = read_numbers_any_orientation(text_ink, dark, self.templates,
                                                                         self.cfg, expected)
            if reading_score(candidate, expected) > reading_score(best, expected):
                best, best_rotation, best_attempt = candidate, candidate_rotation, attempt
        return best, best_rotation, best_attempt

    def process(self, image, image_name):
        """Procesa una foto (BGR) y devuelve el resultado en el formato común."""
        times = {}
        start = step = time.perf_counter()

        def lap(stage):
            nonlocal step
            now = time.perf_counter()
            times[stage] = round(1000 * (now - step), 1)
            step = now

        paper, sheet_polygon = paper_mask(image, self.cfg)
        lap("paper")
        reference = None
        if self.aruco is not None:
            reference = self.aruco.locate(image)
            margin = self.aruco_cfg.get("exclusion_margin", 0.15)
            for _, quad in reference["all_markers"]:
                center = quad.mean(0)
                grown = np.round(center + (quad - center) * (1 + margin)).astype(np.int32)
                cv2.fillConvexPoly(paper, grown, 0)
            lap("aruco")
        dark = darkness(image, self.cfg)
        ink = ink_mask(dark, paper, self.cfg)
        lap("ink")
        dots, text_ink, dot_radius = detect_dots(ink, self.cfg)
        lap("dots")
        numbers, rotation = read_numbers_any_orientation(text_ink, dark, self.templates, self.cfg, len(dots))
        numbers, rotation, attempt = self.retry_faint_numbers(numbers, rotation, dark, paper, dots)
        lap("ocr")

        dots = [{"id": i, "x": d["x"], "y": d["y"], "confidence": d["confidence"], "box": d["box"]}
                for i, d in enumerate(dots)]
        numbers = [{"id": i, **n} for i, n in enumerate(numbers)]
        max_distance = max_pair_distance(numbers, self.max_distance_factor) or 0.0
        pairs, unpaired_dots, unpaired_numbers = match_dots_numbers(dots, numbers, max_distance)
        lap("matching")

        metadata = {"approach": "ClassicVision", "max_pair_distance": round(max_distance, 2),
                    "text_rotation_deg": rotation,  # giro con el que se han leído los números
                    "ocr_attempt": attempt,         # 0 = lectura normal; 1, 2... = segundo intento (classic.retries)
                    "dot_radius_px": round(dot_radius, 2) if dot_radius else None,
                    "sheet_polygon": sheet_polygon.round(1).tolist() if sheet_polygon is not None else None,
                    "sequence": sequence_check([number["value"] for number in numbers])}
        height, width = image.shape[:2]
        result = json_format.create(image_name, width, height, dots, numbers, pairs, unpaired_dots,
                                    unpaired_numbers, times_ms=times, metadata=metadata)
        if reference is not None:
            metadata["aruco"] = add_mm_coordinates(result, reference)
            times["aruco"] = round(times["aruco"] + 1000 * (time.perf_counter() - step), 1)
        times["total"] = round(1000 * (time.perf_counter() - start), 1)
        self.last_reference = reference
        self.last_masks = {"paper": paper, "ink": ink, "text_ink": text_ink}
        return result


def main():
    parser = argparse.ArgumentParser(description="Detecta, lee y empareja puntos y números con visión clásica.")
    parser.add_argument("source", type=Path, help="una foto o una carpeta con fotos")
    parser.add_argument("--output", type=Path, help="carpeta de salida (por defecto, classic.output)")
    parser.add_argument("--join", action="store_true", help="dibujar la línea que une los puntos en orden")
    parser.add_argument("--debug", action="store_true", help="guardar también las máscaras de hoja y tinta")
    args = parser.parse_args()

    config = load_config()
    cfg = config["classic"]
    output = args.output or project_path(cfg["output"])
    images = [args.source] if args.source.is_file() else sorted(
        p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No hay fotos en {args.source}.")
    check_unique_image_stems(images)
    output.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(config)
    for i, path in enumerate(images, start=1):
        csv_path = output / f"{path.stem}_mm.csv"
        csv_path.unlink(missing_ok=True)
        image = read_image(path)
        result = pipeline.process(image, path.name)
        json_format.save(result, output / f"{path.stem}.json")
        drawing = draw_annotation(image, result, join_in_order=args.join or cfg["join_in_order"])
        if pipeline.last_reference is not None:
            draw_reference(drawing, pipeline.last_reference)
        write_image(output / f"{path.stem}.jpg", drawing, 90)
        if args.debug:
            masks = pipeline.last_masks
            debug = cv2.merge([masks["text_ink"], masks["ink"], masks["paper"] // 3])
            write_image(output / f"{path.stem}_debug.png", debug)
        aruco = result["metadata"].get("aruco")
        if aruco is not None and aruco["found"]:
            save_path_csv(aruco, csv_path)
        print(f"[{i}/{len(images)}] {path.name}: {summary_line(result)}")
    print(f"Resultados en {output}")


def summary_line(result):
    """Resumen de una línea de un resultado, con avisos de la secuencia y de ArUco."""
    text = (f"{len(result['dots'])} puntos, {len(result['numbers'])} números, {len(result['pairs'])} parejas, "
            f"{result['times_ms']['total']:.0f} ms")
    sequence = result["metadata"]["sequence"]
    if sequence and (sequence["missing"] or sequence["repeated"] or sequence["out_of_range"]):
        text += (f" | secuencia 1..{sequence['n']}: faltan {sequence['missing']}, repetidos {sequence['repeated']}, "
                 f"fuera de rango {sequence['out_of_range']}")
    aruco = result["metadata"].get("aruco")
    if aruco is not None:
        text += f" | {describe_aruco(aruco)}"
    return text


if __name__ == "__main__":
    main()
