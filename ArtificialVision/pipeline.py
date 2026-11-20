"""Inferencia del enfoque de visión artificial: YOLO26 detecta, EasyOCR lee y el algoritmo húngaro empareja.

Pasos para cada foto (el tiempo de cada uno queda en el JSON):
  1. Escala de trabajo. Si la foto es más grande que max_side, se reduce: el modelo se entrenó con
     fotos de hasta unos 2900 px (4-8 px/mm). Reducirla hasta imgsz, en cambio, encogería los
     puntos.
  2. Detección. La foto de trabajo se trocea en teselas de imgsz con solape, como en el
     entrenamiento, y YOLO las procesa todas juntas. Cada tesela se queda solo con los objetos
     cuyo centro cae en su zona propia (las zonas compartidas se reparten a medias): así ningún
     objeto sale dos veces y, como el solape es mayor que cualquier objeto, se ven enteros.
  3. Borrado de puntos. Sobre la foto original se tapan con un círculo algo mayor que cada punto,
     del color del papel que lo rodea (ver erase_dots), para que el OCR no los lea como ceros ni
     los pegue a los números.
  4. Lectura con EasyOCR, solo dígitos, en uno de dos modos:
     - page: EasyOCR busca el texto en toda la foto con su propio detector y lo lee. Puede fusionar
       números cercanos en una caja, así que se ajustan sus parámetros de fusión (page_params).
     - crops: EasyOCR lee cada caja "number" de YOLO como una sola línea, sin su detector. Si una
       caja no se puede leer, se queda en el emparejamiento con value null, para que su punto no
       se quede con un número ajeno.
     En los dos modos se descartan las lecturas con poca confianza.
  5. Emparejamiento (common/matching.py) y diagnóstico de la secuencia 1..N.
  6. Coordenadas reales (ArtificialVision/aruco_coords.py, si aruco.enabled). Se busca la hoja de
     marcadores ArUco de clase y, con la homografía imagen -> hoja, cada punto recibe x_mm, y_mm.
     La detección de los marcadores se hace justo después de YOLO, para descartar lo que YOLO
     haya detectado dentro de un marcador (sus cuadrados negros pueden parecer puntos).

Salida por foto, en la carpeta de salida: <nombre>.json (formato común) y <nombre>.jpg (dibujo).
Si se encuentra la hoja de ArUco, además <nombre>_mm.csv: número, x_mm, y_mm en orden numérico.

Uso, desde la carpeta del proyecto:
    python -m ArtificialVision.pipeline data/real                      # carpeta, modo de config.yaml
    python -m ArtificialVision.pipeline foto.jpg --mode page --join    # una foto, modo page
    python -m ArtificialVision.pipeline data/synthetic/test/images --output <carpeta>
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ArtificialVision.aruco_coords import (ArucoReference, add_mm_coordinates, draw_reference, inside_markers,
                                           save_path_csv)
from ArtificialVision.prepare_yolo import DOT_CLASS, tile_positions
from common import json_format
from common.config import load_config, project_path
from common.drawing import draw_annotation
from common.images import read_image, write_image
from common.matching import match_dots_numbers, max_pair_distance, sequence_check

OCR_MODES = ("page", "crops")
DIGITS = "0123456789"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# -----------------------------------------------------------------------------
# Escala y teselas
# -----------------------------------------------------------------------------
def working_image(image, max_side):
    """Reduce la foto si su lado mayor supera max_side. Devuelve (imagen, escala aplicada)."""
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, scale


def tile_cores(length, size, overlap):
    """Teselas que cubren un lado de 'length' píxeles y la zona propia de cada una.

    Devuelve [(inicio, inicio_zona, fin_zona), ...]. La zona que comparten dos teselas vecinas se
    reparte por la mitad, así que cada posición del lado pertenece a la zona propia de una sola.
    """
    starts = tile_positions(length, size, overlap)
    cores = []
    for i, start in enumerate(starts):
        core_start = 0 if i == 0 else (starts[i - 1] + size + start) / 2       # mitad del solape anterior
        core_end = length if i == len(starts) - 1 else (start + size + starts[i + 1]) / 2
        cores.append((start, core_start, core_end))
    return cores


def tile_grid(width, height, size, overlap):
    """Todas las teselas de una imagen: [(x0, y0, zona propia [x_inicio, y_inicio, x_fin, y_fin]), ...]."""
    return [(x0, y0, (cx0, cy0, cx1, cy1))
            for y0, cy0, cy1 in tile_cores(height, size, overlap)
            for x0, cx0, cx1 in tile_cores(width, size, overlap)]


def collect_detections(tile, boxes, classes, confidences, dots, numbers):
    """Pasa las detecciones de una tesela a coordenadas de la imagen y se queda con las de su zona propia."""
    x0, y0, (cx0, cy0, cx1, cy1) = tile
    for (a0, b0, a1, b1), object_class, confidence in zip(boxes, classes, confidences):
        box = [a0 + x0, b0 + y0, a1 + x0, b1 + y0]
        xc, yc = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if cx0 <= xc < cx1 and cy0 <= yc < cy1:  # si no, es de otra tesela
            (dots if int(object_class) == DOT_CLASS else numbers).append(
                {"box": [float(v) for v in box], "confidence": float(confidence)})


# -----------------------------------------------------------------------------
# Borrado de puntos y lectura
# -----------------------------------------------------------------------------
def erase_dots(image, dots, margin):
    """Copia de la foto con cada punto tapado por un círculo del color del papel que lo rodea.

    El enunciado propone pintarlos de blanco, y en papel blanco es lo mismo. Pero si el papel
    sale gris (sombra, poca luz), un círculo blanco deja un borde marcado que el OCR puede leer
    como un trazo más: un "22" pegado a su punto se leía "222". Por eso el círculo se pinta con la
    mediana de los píxeles de un anillo alrededor del punto, que es el color del papel aunque el
    anillo roce algún número. En validación mejoró los dos modos de OCR, sobre todo el page.
    """
    erased = image.copy()
    height, width = image.shape[:2]
    for dot in dots:
        x0, y0, x1, y1 = dot["box"]
        radius = max(x1 - x0, y1 - y0) / 2 * (1 + margin) + 1
        cx, cy = dot["x"] - 0.5, dot["y"] - 0.5  # -0.5: convenio de píxeles de OpenCV
        outer = radius + 3                       # anillo de 3 px justo fuera del círculo
        xa, xb = max(0, int(cx - outer)), min(width, int(cx + outer) + 2)
        ya, yb = max(0, int(cy - outer)), min(height, int(cy + outer) + 2)
        yy, xx = np.mgrid[ya:yb, xa:xb]
        distance = np.hypot(xx - cx, yy - cy)
        ring = image[ya:yb, xa:xb][(distance > radius) & (distance <= outer)]
        color = np.median(ring, axis=0) if len(ring) else (255, 255, 255)
        cv2.circle(erased, (round(cx), round(cy)), round(radius), tuple(int(c) for c in color), -1)
    return erased


def digits_value(text):
    """Valor de una lectura si son solo dígitos; None si está vacía o tiene otros caracteres."""
    text = text.strip()
    return int(text) if text.isdigit() else None


def polygon_box(polygon):
    """Caja [x0, y0, x1, y1] que encierra un polígono de EasyOCR (lista de esquinas [x, y])."""
    xs, ys = [float(p[0]) for p in polygon], [float(p[1]) for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def read_page(reader, image, cfg):
    """Modo page: EasyOCR busca y lee los números en toda la foto. Devuelve la lista de números."""
    numbers = []
    for polygon, text, confidence in reader.readtext(image, allowlist=DIGITS, **cfg["page_params"]):
        value = digits_value(text)
        if value is not None and confidence >= cfg["ocr_min_confidence"]:
            numbers.append({"box": polygon_box(polygon), "value": value, "confidence": float(confidence)})
    return numbers


def read_crops(reader, image, numbers, cfg, batch_size):
    """Modo crops: EasyOCR lee cada caja "number" de YOLO como una sola línea, sin su propio detector.

    Cada caja se amplía con un margen. EasyOCR recorta cada región y la redimensiona a 64 px de
    alto, así que las pequeñas se amplían. Las lecturas vacías o dudosas dejan value en None, pero
    el número no se quita: sigue contando en el emparejamiento.
    """
    height, width = image.shape[:2]
    regions = []  # formato de EasyOCR: [x_min, x_max, y_min, y_max]
    for number in numbers:
        x0, y0, x1, y1 = number["box"]
        pad = cfg["crop_margin"] * (y1 - y0) + 1
        regions.append([max(0, int(x0 - pad)), min(width, int(np.ceil(x1 + pad))),
                        max(0, int(y0 - pad)), min(height, int(np.ceil(y1 + pad)))])
    readings = reader.recognize(image, horizontal_list=regions, free_list=[], allowlist=DIGITS,
                                batch_size=batch_size, detail=1) if regions else []

    # EasyOCR devuelve las lecturas ordenadas de arriba abajo: se asocian a su región por las coordenadas.
    by_region = {}
    for polygon, text, confidence in readings:
        (x_min, y_min), _, (x_max, y_max), _ = polygon
        by_region.setdefault((int(x_min), int(x_max), int(y_min), int(y_max)), []).append((text, confidence))
    for number, region in zip(numbers, regions):
        found = by_region.get(tuple(region))
        text, confidence = found.pop(0) if found else ("", 0.0)
        value = digits_value(text)
        number["detection_confidence"] = number.pop("confidence")
        number["value"] = value if value is not None and confidence >= cfg["ocr_min_confidence"] else None
        number["confidence"] = float(confidence)
    return numbers


# -----------------------------------------------------------------------------
# Tubería completa
# -----------------------------------------------------------------------------
class Pipeline:
    """Carga YOLO y EasyOCR una sola vez y procesa fotos con process()."""

    def __init__(self, config):
        import easyocr  # se importan aquí porque tardan en cargar
        import torch
        from ultralytics import YOLO

        self.cfg = config["inference"]
        self.size = config["training"]["imgsz"]
        self.overlap = config["yolo"]["overlap"]
        self.max_distance_factor = config["matching"]["max_distance_factor"]
        gpu = torch.cuda.is_available()
        self.device = 0 if gpu else "cpu"
        self.ocr_batch = 32 if gpu else 1
        self.model = YOLO(str(project_path(self.cfg["model"])))
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.aruco_cfg = config.get("aruco") or {}
        self.aruco = ArucoReference(self.aruco_cfg) if self.aruco_cfg.get("enabled") else None
        self.last_reference = None  # hoja de ArUco encontrada en la última foto procesada
        # Calentamiento: la primera llamada a cada modelo es mucho más lenta (reserva memoria,
        # compila...) y falsearía el tiempo de la primera foto.
        blank = np.full((self.size, self.size, 3), 255, np.uint8)
        self.model.predict(blank, imgsz=self.size, device=self.device, verbose=False)
        self.reader.readtext(blank, allowlist=DIGITS)

    def detect(self, image):
        """Puntos y números de la foto (coordenadas de la foto original)."""
        work, scale = working_image(image, self.cfg["max_side"])
        height, width = work.shape[:2]
        tiles = tile_grid(width, height, self.size, self.overlap)
        crops = [work[y0:y0 + self.size, x0:x0 + self.size] for x0, y0, _ in tiles]
        results = self.model.predict(crops, imgsz=self.size, conf=self.cfg["detection_confidence"],
                                     max_det=self.cfg["max_detections"], device=self.device, verbose=False)
        dots, numbers = [], []
        for tile, result in zip(tiles, results):
            boxes = result.boxes
            collect_detections(tile, boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                               boxes.conf.cpu().numpy(), dots, numbers)
        for item in dots + numbers:  # de la foto de trabajo a la original
            item["box"] = [round(v / scale, 2) for v in item["box"]]
        for dot in dots:
            x0, y0, x1, y1 = dot["box"]
            dot["x"], dot["y"] = round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)
        return dots, numbers, scale

    def process(self, image, image_name, ocr_mode):
        """Procesa una foto (BGR) y devuelve el resultado en el formato común."""
        if ocr_mode not in OCR_MODES:
            raise ValueError(f"Modo de OCR desconocido: {ocr_mode}. Opciones: {OCR_MODES}")
        times = {}
        start = step = time.perf_counter()

        def lap(stage):
            nonlocal step
            now = time.perf_counter()
            times[stage] = round(1000 * (now - step), 1)
            step = now

        dots, numbers, scale = self.detect(image)
        lap("detection")
        reference = None
        if self.aruco is not None:
            reference = self.aruco.locate(image)
            margin = self.aruco_cfg.get("exclusion_margin", 0.15)
            dots = [d for d in dots if not inside_markers(d["x"], d["y"], reference["all_markers"], margin)]
            numbers = [n for n in numbers if not inside_markers((n["box"][0] + n["box"][2]) / 2,
                                                                (n["box"][1] + n["box"][3]) / 2,
                                                                reference["all_markers"], margin)]
            lap("aruco")
        erased = erase_dots(image, dots, self.cfg["erase_margin"])
        lap("erase_dots")
        if ocr_mode == "page":
            numbers = read_page(self.reader, erased, self.cfg)
        else:
            numbers = read_crops(self.reader, erased, numbers, self.cfg, self.ocr_batch)
        lap("ocr")

        dots = [{"id": i, "x": dot["x"], "y": dot["y"], "confidence": round(dot["confidence"], 4), "box": dot["box"]}
                for i, dot in enumerate(dots)]
        numbers = [{"id": i, "value": number["value"], "box": [round(v, 2) for v in number["box"]],
                    "confidence": round(number["confidence"], 4),
                    **{key: round(number[key], 4) for key in ("detection_confidence",) if key in number}}
                   for i, number in enumerate(numbers)]
        max_distance = max_pair_distance(numbers, self.max_distance_factor) or 0.0
        pairs, unpaired_dots, unpaired_numbers = match_dots_numbers(dots, numbers, max_distance)
        lap("matching")

        metadata = {"approach": "ArtificialVision", "ocr_mode": ocr_mode, "working_scale": round(scale, 4),
                    "max_pair_distance": round(max_distance, 2),
                    "sequence": sequence_check([number["value"] for number in numbers])}
        height, width = image.shape[:2]
        result = json_format.create(image_name, width, height, dots, numbers, pairs, unpaired_dots,
                                    unpaired_numbers, times_ms=times, metadata=metadata)
        if reference is not None:
            metadata["aruco"] = add_mm_coordinates(result, reference)  # añade x_mm, y_mm a cada punto
            times["aruco"] = round(times["aruco"] + 1000 * (time.perf_counter() - step), 1)
        times["total"] = round(1000 * (time.perf_counter() - start), 1)
        self.last_reference = reference  # para dibujar los marcadores y los ejes (main)
        return result


def describe(result, max_rms_mm=None):
    """Resumen de una línea de un resultado, con el aviso de la secuencia si falta o se repite algo."""
    text = (f"{len(result['dots'])} puntos, {len(result['numbers'])} números, {len(result['pairs'])} parejas, "
            f"{result['times_ms']['total']:.0f} ms")
    sequence = result["metadata"]["sequence"]
    if sequence and (sequence["missing"] or sequence["repeated"] or sequence["out_of_range"]):
        text += (f" | secuencia 1..{sequence['n']}: faltan {sequence['missing']}, repetidos {sequence['repeated']}, "
                 f"fuera de rango {sequence['out_of_range']}")
    aruco = result["metadata"].get("aruco")
    if aruco is not None:
        if not aruco["found"]:
            text += f" | ArUco: hoja no encontrada (marcadores {aruco['markers']}), sin coordenadas en mm"
        else:
            text += f" | ArUco {aruco['markers']}: error {aruco['rms_mm']:.2f} mm"
            if max_rms_mm is not None and aruco["rms_mm"] > max_rms_mm:
                text += f" (¡más de {max_rms_mm} mm! revisa que la hoja esté plana y en el plano de la ficha)"
    return text


def main():
    parser = argparse.ArgumentParser(description="Detecta, lee y empareja los puntos y números de fotos.")
    parser.add_argument("source", type=Path, help="una foto o una carpeta con fotos")
    parser.add_argument("--mode", choices=OCR_MODES, help="modo de OCR (por defecto, el de config.yaml)")
    parser.add_argument("--output", type=Path, help="carpeta de salida (por defecto, inference.output/<modo>)")
    parser.add_argument("--join", action="store_true", help="dibujar la línea que une los puntos en orden")
    args = parser.parse_args()

    config = load_config()
    cfg = config["inference"]
    mode = args.mode or cfg["ocr_mode"]
    output = args.output or project_path(cfg["output"]) / mode
    if args.source.is_file():
        images = [args.source]
    else:
        images = sorted(path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No hay fotos en {args.source}.")
    output.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelos... (modo de OCR: {mode})")
    pipeline = Pipeline(config)
    for i, path in enumerate(images, start=1):
        image = read_image(path)
        result = pipeline.process(image, path.name, mode)
        json_format.save(result, output / f"{path.stem}.json")
        drawing = draw_annotation(image, result, join_in_order=args.join or cfg["join_in_order"])
        if pipeline.last_reference is not None:
            draw_reference(drawing, pipeline.last_reference)
            if pipeline.last_reference["found"]:
                save_path_csv(result["metadata"]["aruco"], output / f"{path.stem}_mm.csv")
        write_image(output / f"{path.stem}.jpg", drawing, 90)
        print(f"[{i}/{len(images)}] {path.name}: {describe(result, config.get('aruco', {}).get('max_rms_mm'))}")
    print(f"Resultados en {output}")


if __name__ == "__main__":
    main()
