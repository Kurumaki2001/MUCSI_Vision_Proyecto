"""Rotuladores Edding 1200: dónde está cada uno, cómo está girado y de qué color es (azul, rojo, negro).

Pasos para cada foto (el tiempo de cada uno queda en el JSON):
  1. detection: YOLO (entrenado con una sola clase, "rotulador") da una caja por rotulador.
  2. segmentation: SAM 2.1 recibe cada caja como pista y devuelve la silueta del rotulador
     (pens.segmenter: sam). Con pens.segmenter: grabcut se usa GrabCut (tema 5), sin red.
  3. geometry y color (pens/analysis.py): rectángulo de área mínima de la silueta -> centro,
     largo y ángulo del eje; colores en HSV dentro de la silueta -> azul, rojo o negro.
  4. aruco: con la hoja de marcadores (common/aruco_coords.py, la misma que para los puntos), el
     centro y los extremos del eje se pasan a milímetros sobre la mesa. El ángulo en mm se calcula
     con los extremos ya convertidos, porque la perspectiva cambia los ángulos de la imagen.

Salida por foto, en la carpeta de salida:
  <nombre>_rotuladores.json  todos los datos (píxeles, mm, fracciones de color, tiempos)
  <nombre>_rotuladores.jpg   dibujo: silueta, rectángulo, eje y color de cada rotulador
  <nombre>_rotuladores_mm.csv  color, x_mm, y_mm, angulo_deg, largo_mm (con referencia ArUco válida)

Uso, desde la carpeta del proyecto:
    python -m ArtificialVision.pens.pipeline foto.jpg
    python -m ArtificialVision.pens.pipeline carpeta/ --segmenter grabcut
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ArtificialVision.pens.analysis import largest_blob, pen_color, pen_geometry, segment_grabcut
from common.aruco_coords import (ArucoReference, aruco_metadata, describe_aruco, draw_reference, to_mm,
                                 write_robot_csv)
from common.config import load_config, project_path
from common.images import check_unique_image_stems, read_image, write_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DRAW_COLORS = {"azul": (230, 120, 0), "rojo": (40, 40, 230), "negro": (30, 30, 30), "desconocido": (0, 200, 255)}


class PenPipeline:
    """Carga YOLO (y SAM) una sola vez y procesa fotos con process()."""

    def __init__(self, config, segmenter=None):
        import torch
        from ultralytics import YOLO

        self.cfg = config["pens"]
        self.segmenter = segmenter or self.cfg["segmenter"]
        self.device = 0 if torch.cuda.is_available() else "cpu"
        model_path = project_path(self.cfg["model"])
        if not model_path.exists():
            raise SystemExit(f"No existe el modelo {model_path}. Entrénalo antes: ver el docstring de "
                             "ArtificialVision/pens/train.py")
        self.yolo = YOLO(str(model_path))
        self.sam = None
        if self.segmenter == "sam":
            from ultralytics import SAM
            self.sam = SAM(str(project_path(self.cfg["sam_model"])))
        aruco_cfg = config.get("aruco") or {}
        self.aruco = ArucoReference(aruco_cfg) if aruco_cfg.get("enabled") else None
        self.last_reference = None

    def detect(self, image):
        """Cajas [x0, y0, x1, y1] y confianzas de los rotuladores (YOLO)."""
        result = self.yolo.predict(image, imgsz=self.cfg["imgsz"], conf=self.cfg["detection_confidence"],
                                   device=self.device, verbose=False)[0]
        return result.boxes.xyxy.cpu().numpy().tolist(), result.boxes.conf.cpu().numpy().tolist()

    def segment(self, image, boxes):
        """Una máscara (uint8 0/255, tamaño de la foto) o None por cada caja de YOLO."""
        if self.sam is None:
            return [segment_grabcut(image, box) for box in boxes]
        height, width = image.shape[:2]
        masks = []
        for box in boxes:
            # SAM puede descartar una máscara. Una consulta por caja conserva el índice de YOLO.
            result = self.sam(image, bboxes=[box], verbose=False)[0]
            if result.masks is None or len(result.masks.data) == 0:
                masks.append(None)
                continue
            data = result.masks.data[0].cpu().numpy()
            mask = (data > 0.5).astype(np.uint8) * 255
            if mask.shape != (height, width):  # por si SAM devuelve otra resolución
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append(largest_blob(mask))
        return masks

    def process(self, image, image_name):
        """Procesa una foto (BGR). Devuelve (resultado como diccionario, máscaras)."""
        times = {}
        start = step = time.perf_counter()

        def lap(stage):
            nonlocal step
            now = time.perf_counter()
            times[stage] = round(1000 * (now - step), 1)
            step = now

        boxes, confidences = self.detect(image)
        lap("detection")
        masks = self.segment(image, boxes)
        lap("segmentation")
        reference = self.aruco.locate(image) if self.aruco is not None else None
        lap("aruco")
        result = analyse(image, image_name, boxes, confidences, masks, reference, self.cfg)
        lap("geometry_color")
        times["total"] = round(1000 * (time.perf_counter() - start), 1)
        result["times_ms"] = times
        result["segmenter"] = self.segmenter
        self.last_reference = reference
        return result, masks


def analyse(image, image_name, boxes, confidences, masks, reference, cfg):
    """Geometría, color y milímetros de cada rotulador (separado de process() para probarlo sin YOLO)."""
    found = reference is not None and reference["found"]
    pens = []
    for i, (box, confidence, mask) in enumerate(zip(boxes, confidences, masks)):
        geometry = pen_geometry(mask) if mask is not None else None
        if geometry is None:
            continue
        color, fractions = pen_color(image, mask, cfg["color"])
        pen = {"id": i, "color": color, "confidence": round(float(confidence), 4),
               "color_fractions": {k: round(v, 4) for k, v in fractions.items()},
               "box": [round(float(v), 1) for v in box],
               "center_px": [round(v, 1) for v in geometry["center"]],
               "angle_deg_px": round(geometry["angle_deg"], 1),
               "length_px": round(geometry["length"], 1), "width_px": round(geometry["width"], 1),
               "ends_px": [[round(v, 1) for v in end] for end in geometry["ends"]],
               "box_points_px": [[round(v, 1) for v in p] for p in geometry["box_points"]]}
        if found:
            points = to_mm([geometry["center"], *geometry["ends"]], reference["homography"], reference["lens"],
                           reference["shape"])
            center, end_a, end_b = points
            axis = end_b - end_a
            pen.update(center_mm=[round(float(v), 1) for v in center],
                       angle_deg=round(float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0), 1),
                       length_mm=round(float(np.linalg.norm(axis)), 1))
        pens.append(pen)
    height, width = image.shape[:2]
    aruco = aruco_metadata(reference) if reference is not None else None
    if found:
        aruco.update(units="mm",
                     frame="hoja ArUco: origen en su esquina superior izquierda, X a la derecha, Y hacia "
                           "abajo; angle_deg desde el eje X hacia el Y, de 0 a 180")
    return {"image": image_name, "width": width, "height": height, "pens": pens, "aruco": aruco}


def draw_pens(image, result, masks):
    """Dibujo: silueta coloreada, rectángulo de área mínima, eje y etiqueta de cada rotulador."""
    output = image.copy()
    layer = output.copy()
    by_id = {pen["id"]: pen for pen in result["pens"]}
    for i, mask in enumerate(masks):
        if i in by_id:
            layer[mask > 0] = DRAW_COLORS[by_id[i]["color"]]
    output = cv2.addWeighted(layer, 0.35, output, 0.65, 0)
    scale = max(0.6, image.shape[0] / 1500)
    for pen in result["pens"]:
        color = DRAW_COLORS[pen["color"]]
        cv2.polylines(output, [np.int32(np.round(np.array(pen["box_points_px"]) - 0.5))], True, color, 3, cv2.LINE_AA)
        (ax, ay), (bx, by) = pen["ends_px"]
        cv2.line(output, (round(ax), round(ay)), (round(bx), round(by)), (255, 255, 255), 2, cv2.LINE_AA)
        cx, cy = pen["center_px"]
        cv2.circle(output, (round(cx), round(cy)), 6, (0, 255, 0), -1)
        text = pen["color"]
        if "center_mm" in pen:
            text += f" ({pen['center_mm'][0]:.0f}, {pen['center_mm'][1]:.0f}) mm {pen['angle_deg']:.0f} grados"
        for thickness, ink in ((5, (255, 255, 255)), (2, color)):
            cv2.putText(output, text, (round(cx) + 10, round(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, ink,
                        thickness, cv2.LINE_AA)
    return output


def save_csv(result, path):
    """CSV para el robot: un rotulador por línea con su color, posición (mm), ángulo y largo."""
    lines = ["color,x_mm,y_mm,angulo_deg,largo_mm"]
    lines += [f"{p['color']},{p['center_mm'][0]},{p['center_mm'][1]},{p['angle_deg']},{p['length_mm']}"
              for p in result["pens"] if "center_mm" in p]
    write_robot_csv(lines, path)


def main():
    parser = argparse.ArgumentParser(description="Detecta los rotuladores, su color y su posición en mm.")
    parser.add_argument("source", type=Path, help="una foto o una carpeta con fotos")
    parser.add_argument("--output", type=Path, help="carpeta de salida (por defecto, pens.output)")
    parser.add_argument("--segmenter", choices=("sam", "grabcut"), help="silueta con SAM o con GrabCut")
    args = parser.parse_args()

    config = load_config()
    output = args.output or project_path(config["pens"]["output"])
    images = [args.source] if args.source.is_file() else sorted(
        p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No hay fotos en {args.source}.")
    check_unique_image_stems(images)
    output.mkdir(parents=True, exist_ok=True)

    pipeline = PenPipeline(config, args.segmenter)
    for i, path in enumerate(images, start=1):
        stem = output / f"{path.stem}_rotuladores"
        csv_path = Path(f"{stem}_mm.csv")
        csv_path.unlink(missing_ok=True)
        image = read_image(path)
        result, masks = pipeline.process(image, path.name)
        Path(f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        drawing = draw_pens(image, result, masks)
        if pipeline.last_reference is not None:
            draw_reference(drawing, pipeline.last_reference)
        write_image(f"{stem}.jpg", drawing, 90)
        aruco = result["aruco"]
        if aruco and aruco["found"]:
            save_csv(result, csv_path)
        summary = ", ".join(p["color"] + (f" ({p['center_mm'][0]:.0f}, {p['center_mm'][1]:.0f}) mm"
                                          if "center_mm" in p else "") for p in result["pens"])
        aruco_summary = f" | {describe_aruco(aruco)}" if aruco else ""
        print(f"[{i}/{len(images)}] {path.name}: {len(result['pens'])} rotuladores: {summary or '-'} "
              f"({result['times_ms']['total']:.0f} ms){aruco_summary}")
    print(f"Resultados en {output}")


if __name__ == "__main__":
    main()
