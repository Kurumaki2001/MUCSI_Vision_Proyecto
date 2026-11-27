"""Coordenadas reales (en milímetros) de los puntos, con los marcadores ArUco.

El pipeline da la posición de cada punto en píxeles de la foto. Para que un robot pueda ir a
cada punto hace falta pasarla a milímetros sobre la mesa. Para eso se usa la hoja de calibración
de clase: un A4 con 4 marcadores DICT_4X4_50 (IDs 0-3) de 80 mm de lado, cuyas
esquinas reales están en aruco_sheet/arucos_calibracion_A4.json.

Idea: la hoja de ArUco y la ficha de puntos están sobre el mismo plano (la mesa). Entre un plano
y su foto hay una homografía (una matriz 3x3), sea cual sea la perspectiva o la focal de la
cámara. Conocemos dónde están en milímetros las 16 esquinas de los 4 marcadores y las detectamos
en la foto, así que podemos calcular esa homografía con cv2.findHomography y aplicarla a
cualquier punto del plano, en particular a los centros de los puntos que detecta YOLO.

Pasos (ArucoReference.locate):
  1. Detección de los marcadores con cv2.aruco.ArucoDetector.
  2. Refinado sub-píxel de las esquinas ajustando rectas a los bordes del marcador (edge_refine:
     es más estable que cornerSubPix con fotos borrosas).
  3. Corrección de la distorsión radial de la lente (LensModel, con calibracion_camara.json,
     generado con aruco_calibrate.py para la cámara de clase). Solo se aplica si la foto
     tiene la misma relación de aspecto que la calibración.
  4. Homografía imagen -> hoja (mm) por mínimos cuadrados con todas las esquinas visibles. Con
     un solo marcador también se podría (4 esquinas), pero se extrapolaría mucho: por eso se
     piden al menos min_markers.
  5. Calidad: error (RMS y máximo, en mm) entre las esquinas proyectadas y sus posiciones
     reales. Una referencia ambigua o con un ajuste fuera de tolerancia se rechaza.

Sistema de coordenadas resultante (el del JSON de la hoja): origen en la esquina superior
izquierda de la hoja de ArUco, X hacia la derecha, Y hacia abajo, en milímetros. Pasarlo al
sistema del robot (otro origen y giro) será una transformación rígida aparte.

Uso independiente (sin YOLO ni OCR), desde la carpeta del proyecto:
    python -m common.aruco_coords foto.png                    # marcadores y error
    python -m common.aruco_coords foto.png --point 640 480    # un píxel -> mm
    python -m common.aruco_coords carpeta/ --validate         # validación cruzada

Para recalibrar la lente (otra cámara, resolución, enfoque o iluminación): aruco_capture.py y
aruco_calibrate.py.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from common.config import load_config, project_path
from common.images import check_unique_image_stems, read_image, write_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAGENTA = (255, 0, 255)
AXIS_X, AXIS_Y = (0, 0, 255), (0, 200, 0)   # BGR: eje X en rojo, eje Y en verde


def make_detector(dict_name="DICT_4X4_50"):
    """Detector ArUco sin refinado de OpenCV: las esquinas se refinan después con edge_refine."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def edge_refine(gray_f, q, offset_px=0.0, n_mod=6, iters=2, n_samples=40):
    """Refina las 4 esquinas de un marcador ajustando rectas a sus bordes exteriores.

    En cada lado se toman 40 perfiles de intensidad perpendiculares
    al borde, se busca en cada uno el cruce sub-píxel con el nivel medio entre negro y blanco, se
    ajusta una recta robusta (Huber) y las esquinas son las intersecciones de lados consecutivos.
    gray_f: imagen en float32 ya suavizada; q: esquinas (4, 2). Devuelve (4, 2) o None si falla.
    """
    q = q.astype(np.float64).copy()
    ts = np.linspace(0.12, 0.88, n_samples)
    for it in range(iters):
        c = q.mean(0)
        lines = []
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            L = np.linalg.norm(b - a)
            if L < 12:
                return None
            d = (b - a) / L
            nrm = np.array([d[1], -d[0]])
            if np.dot(nrm, (a + b) / 2 - c) < 0:
                nrm = -nrm                                   # normal hacia fuera
            w = 0.6 * L / n_mod                              # < 1 módulo: dentro solo hay borde negro
            ss = np.linspace(-w, w, max(9, int(2 * w) + 1))
            base = a[None, :] + (ts[:, None] * L) * d        # puntos sobre el borde
            P = base[:, None, :] + ss[None, :, None] * nrm
            prof = cv2.remap(gray_f, P[..., 0].astype(np.float32), P[..., 1].astype(np.float32),
                             cv2.INTER_LINEAR)
            nin = max(2, len(ss) // 5)
            lo = np.median(prof[:, :nin], axis=1)
            hi = np.median(prof[:, -nin:], axis=1)
            pts = []
            for i_t in range(len(ts)):
                if hi[i_t] - lo[i_t] < 20:                   # sin contraste (oclusión, reflejo)
                    continue
                pr, mid = prof[i_t], (lo[i_t] + hi[i_t]) / 2
                idx = np.where((pr[:-1] < mid) & (pr[1:] >= mid))[0]
                if len(idx) == 0:
                    continue
                j = idx[np.argmin(np.abs(idx - len(ss) // 2))]
                s = ss[j] + (mid - pr[j]) / (pr[j + 1] - pr[j]) * (ss[j + 1] - ss[j])
                pts.append(base[i_t] + s * nrm)
            if len(pts) < n_samples // 4:
                return None
            vx, vy, x0, y0 = cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            p0 = np.array([x0, y0], dtype=np.float64)
            if it == iters - 1:
                p0 = p0 + offset_px * nrm                    # compensación del borde
            lines.append((p0, np.array([vx, vy], dtype=np.float64)))
        new = []
        for k in range(4):                                   # esquina k = borde k-1 ∩ borde k
            (p1, v1), (p2, v2) = lines[(k - 1) % 4], lines[k]
            A = np.array([v1, -v2]).T
            if abs(np.linalg.det(A)) < 1e-6:
                return None
            t = np.linalg.solve(A, p2 - p1)
            new.append(p1 + t[0] * v1)
        new = np.array(new)
        if np.abs(new - q).max() > 0.25 * L:                 # salto absurdo -> descartar
            return None
        q = new
    return q


class LensModel:
    """Distorsión radial (k1, k2) y desplazamiento del borde, de calibracion_camara.json.

    Las coordenadas se normalizan por la
    semidiagonal de la imagen, así que el modelo vale para otra resolución con la misma relación
    de aspecto; el desplazamiento del borde (px) se reescala con el ancho.
    """

    def __init__(self, k1=0.0, k2=0.0, edge_offset_px=0.0, width=None, height=None):
        self.k1, self.k2, self.edge_offset_px = k1, k2, edge_offset_px
        self.width, self.height = width, height

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as file:
            d = json.load(file)
        return cls(d.get("k1", 0), d.get("k2", 0), d.get("edge_offset_px", 0), d.get("width"), d.get("height"))

    def save(self, path, extra=None):
        """Guarda el modelo en JSON (lo usa aruco_calibrate.py); 'extra' añade información."""
        data = {"k1": self.k1, "k2": self.k2, "edge_offset_px": self.edge_offset_px,
                "width": self.width, "height": self.height, **(extra or {})}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def fits(self, shape):
        """True si la foto tiene la relación de aspecto de la calibración (si no, el modelo no vale)."""
        if not (self.width and self.height):
            return True
        return abs(shape[1] / shape[0] - self.width / self.height) < 0.01

    def offset_for(self, shape):
        return self.edge_offset_px * shape[1] / self.width if self.width else self.edge_offset_px

    def undistort(self, pts, shape):
        if self.k1 == 0 and self.k2 == 0:
            return pts
        h, w = shape[:2]
        c = np.array([w / 2, h / 2])
        r = np.hypot(w, h) / 2
        v = (pts - c) / r
        r2 = (v ** 2).sum(-1, keepdims=True)
        return c + r * v * (1 + self.k1 * r2 + self.k2 * r2 ** 2)


def sharpness(gray):
    """Nitidez de una foto: varianza del Laplaciano (baja = borrosa o movida)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# -----------------------------------------------------------------------------
# Hoja de marcadores y referencia métrica
# -----------------------------------------------------------------------------
def load_sheet(path):
    """Lee el layout de la hoja (JSON en metros) y lo pasa a milímetros.

    Devuelve {"dictionary", "marker_size_mm", "corners": {id: (4, 2) esquinas TL, TR, BR, BL en mm}}.
    """
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    corners = {int(i): np.array(marker["corners_m"], dtype=np.float64) * 1000
               for i, marker in data["markers"].items()}
    return {"dictionary": data["dictionary"], "marker_size_mm": data["marker_size_m"] * 1000,
            "corners": corners}


def to_mm(points_px, homography, lens=None, shape=None):
    """Pasa puntos (N, 2) de píxeles de la foto a mm en el plano de la hoja."""
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    if lens is not None:
        points = lens.undistort(points, shape)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def fit_homography(detected, sheet_corners):
    """Homografía imagen -> hoja (mm) por mínimos cuadrados con las esquinas de varios marcadores.

    detected: {id: (4, 2) esquinas en px, ya sin distorsión}. Devuelve (H, rms_mm, max_mm),
    con None en los tres campos si el ajuste no es válido.
    Método 0 de findHomography = mínimos cuadrados con todos los puntos (sin RANSAC): las
    esquinas ya son fiables y así cada una aporta lo mismo.
    """
    ids = sorted(detected)
    if not ids:
        return None, None, None
    src = np.concatenate([detected[i] for i in ids])
    dst = np.concatenate([sheet_corners[i] for i in ids])
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        return None, None, None
    try:
        homography, _ = cv2.findHomography(src, dst, 0)
        if homography is None or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
            return None, None, None
        projected = to_mm(src, homography)
    except (cv2.error, np.linalg.LinAlgError):
        return None, None, None
    errors = np.linalg.norm(projected - dst, axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    return (homography, rms, float(errors.max())) if np.isfinite(rms) else (None, None, None)


def duplicate_sheet_ids(marker_ids, sheet_ids):
    """Devuelve los IDs repetidos que pertenecen a la hoja de referencia."""
    counts = Counter(marker_id for marker_id in marker_ids if marker_id in sheet_ids)
    return sorted(marker_id for marker_id, count in counts.items() if count > 1)


class ArucoReference:
    """Detecta la hoja de ArUco en una foto y da la homografía para pasar píxeles a milímetros."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sheet = load_sheet(project_path(cfg["sheet"]))
        # Se detecta con un diccionario más amplio que el de la hoja (DICT_4X4_1000 contiene los 50
        # de DICT_4X4_50 con los mismos IDs): así se encuentran también los marcadores de otras
        # fichas, para no confundirlos con puntos o números. Para medir solo se usan los de la hoja.
        self.detector = make_detector(cfg.get("detect_dictionary") or self.sheet["dictionary"])
        lens_path = cfg.get("lens_calibration")
        self.lens = LensModel.load(project_path(lens_path)) if lens_path else None

    def detect_markers(self, image):
        """Marcadores de la foto: (de la hoja {id: esquinas refinadas (4, 2) px}, todos [(id, esquinas)])."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}, []
        lens = self.lens if self.lens is not None and self.lens.fits(gray.shape) else None
        offset = lens.offset_for(gray.shape) if lens else 0.0
        smooth = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
        sheet_markers, all_markers = {}, []
        for quad, marker_id in zip(corners, ids.ravel()):
            marker_id, quad = int(marker_id), quad.reshape(4, 2).astype(np.float64)
            all_markers.append((marker_id, quad))
            if marker_id not in self.sheet["corners"] or marker_id in sheet_markers:
                continue  # el segundo ejemplar queda en all_markers para detectar la ambigüedad
            if self.cfg.get("refine_edges", True):
                refined = edge_refine(smooth, quad, offset)
                quad = refined if refined is not None else quad   # si el refinado falla, esquinas de OpenCV
            sheet_markers[marker_id] = quad
        return sheet_markers, all_markers

    def locate(self, image):
        """Busca la hoja en la foto. Devuelve un diccionario con todo lo necesario:

        found (bool), ids usados, markers {id: esquinas px}, all_markers (todos los detectados, para
        no confundirlos con puntos), homography (3x3, px sin distorsión -> mm), lens (o None),
        shape, rms_mm y max_mm (error en las esquinas de los marcadores). Si se rechaza la
        referencia, reason explica por qué y homography es None.
        """
        shape = image.shape[:2]
        markers, all_markers = self.detect_markers(image)
        lens = self.lens if self.lens is not None and self.lens.fits(shape) else None
        reference = {"found": False, "ids": sorted(markers), "markers": markers, "all_markers": all_markers,
                     "homography": None, "lens": lens, "shape": shape, "rms_mm": None, "max_mm": None}
        duplicate_ids = duplicate_sheet_ids((i for i, _ in all_markers), self.sheet["corners"])
        if duplicate_ids:
            reference.update(reason="duplicate_markers", duplicate_ids=duplicate_ids)
            return reference
        min_markers = self.cfg.get("min_markers", 2)
        if len(markers) < min_markers:
            reference.update(reason="insufficient_markers", min_markers=min_markers)
            return reference
        undistorted = {i: lens.undistort(q, shape) if lens else q for i, q in markers.items()}
        homography, rms, worst = fit_homography(undistorted, self.sheet["corners"])
        if homography is None:
            reference["reason"] = "invalid_homography"
            return reference
        reference.update(rms_mm=rms, max_mm=worst)
        max_rms = self.cfg.get("max_rms_mm", 1.0)
        if max_rms is not None and rms > max_rms:
            reference.update(reason="high_rms", max_rms_mm=max_rms)
            return reference
        reference.update(found=True, homography=homography)
        return reference


# -----------------------------------------------------------------------------
# Uso desde el pipeline
# -----------------------------------------------------------------------------
def inside_markers(x, y, all_markers, margin):
    """True si (x, y) cae dentro de algún marcador (agrandado un 'margin' de su tamaño).

    Los cuadrados negros y los bits de los ArUco pueden parecerle a YOLO puntos o números: lo que
    caiga dentro de un marcador se descarta antes del OCR y del emparejamiento.
    """
    for _, quad in all_markers:
        center = quad.mean(0)
        grown = (center + (quad - center) * (1 + margin)).astype(np.float32)
        if cv2.pointPolygonTest(grown, (float(x), float(y)), False) >= 0:
            return True
    return False


def aruco_metadata(reference):
    """Extrae el estado y los diagnósticos de ArUco que se guardan en JSON."""
    summary = {"found": reference["found"], "markers": reference["ids"],
               "lens_correction": reference["lens"] is not None}
    for key in ("rms_mm", "max_mm"):
        if reference.get(key) is not None:
            summary[key] = round(reference[key], 3)
    for key in ("reason", "duplicate_ids", "min_markers", "max_rms_mm"):
        if key in reference:
            summary[key] = reference[key]
    return summary


def describe_aruco(summary):
    """Resume el ajuste o indica por qué se omitieron las coordenadas en milímetros."""
    markers = summary["markers"]
    if summary["found"]:
        return f"ArUco {markers}: error {summary['rms_mm']:.2f} mm"
    reason = summary.get("reason")
    if reason == "duplicate_markers":
        detail = f"IDs duplicados {summary['duplicate_ids']}"
    elif reason == "high_rms":
        detail = f"error de ajuste {summary['rms_mm']:.2f} mm (límite {summary['max_rms_mm']:g} mm)"
    elif reason == "invalid_homography":
        detail = "homografía no válida"
    elif reason == "insufficient_markers":
        detail = f"se necesitan al menos {summary['min_markers']} marcadores"
    else:
        detail = "hoja no encontrada"
    return f"ArUco {markers}: {detail}, sin coordenadas en mm"


def add_mm_coordinates(result, reference):
    """Añade x_mm, y_mm a cada punto del resultado y devuelve el resumen para metadata["aruco"].

    El resumen incluye "path_mm": los puntos emparejados con un número leído, ordenados por ese
    número, que es la lista de destinos que necesitará el robot.
    """
    summary = aruco_metadata(reference)
    if not reference["found"]:
        return summary
    dots = result["dots"]
    coords = to_mm([[dot["x"], dot["y"]] for dot in dots], reference["homography"], reference["lens"],
                   reference["shape"])
    for dot, (x_mm, y_mm) in zip(dots, coords):
        dot["x_mm"], dot["y_mm"] = round(float(x_mm), 2), round(float(y_mm), 2)

    by_id = {dot["id"]: dot for dot in dots}
    values = {number["id"]: number.get("value") for number in result["numbers"]}
    path = sorted((values[pair["number"]], pair["dot"]) for pair in result["pairs"]
                  if values.get(pair["number"]) is not None)
    summary.update(units="mm", frame="hoja ArUco: origen en su esquina superior izquierda, X a la derecha, Y hacia abajo",
                   path_mm=[{"value": value, "dot": dot_id, "x_mm": by_id[dot_id]["x_mm"], "y_mm": by_id[dot_id]["y_mm"]}
                            for value, dot_id in path])
    return summary


def save_path_csv(summary, path):
    """Guarda la lista de destinos (número, x_mm, y_mm) en un CSV fácil de pasar al robot."""
    lines = ["value,x_mm,y_mm"] + [f"{p['value']},{p['x_mm']},{p['y_mm']}" for p in summary.get("path_mm", [])]
    write_robot_csv(lines, path)


def write_robot_csv(lines, path):
    """Publica un CSV completo para que el robot no pueda leer una escritura a medias."""
    path = Path(path)
    file = NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write("\n".join(lines) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def draw_reference(image, reference, axis_mm=50):
    """Dibuja los marcadores usados (magenta) y los ejes X (rojo) e Y (verde) del sistema en mm."""
    output = image
    for marker_id, quad in reference["markers"].items():
        cv2.polylines(output, [np.int32(np.round(quad))], True, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(output, f"id {marker_id}", tuple(np.int32(quad[0] + [0, -6])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, MAGENTA, 2, cv2.LINE_AA)
    if reference["found"]:
        # mm -> px con la inversa de la homografía (sin deshacer la distorsión de la lente: solo es
        # un dibujo, y la diferencia es de unos pocos píxeles)
        inverse = np.linalg.inv(reference["homography"])
        o, x, y = cv2.perspectiveTransform(np.float64([[[0, 0]], [[axis_mm, 0]], [[0, axis_mm]]]), inverse).reshape(-1, 2)
        cv2.arrowedLine(output, tuple(np.int32(o)), tuple(np.int32(x)), AXIS_X, 3, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(output, tuple(np.int32(o)), tuple(np.int32(y)), AXIS_Y, 3, cv2.LINE_AA, tipLength=0.15)
        cv2.putText(output, f"ArUco {reference['ids']}  error {reference['rms_mm']:.2f} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, MAGENTA, 2, cv2.LINE_AA)
    return output


# -----------------------------------------------------------------------------
# Validación: dejar un marcador fuera y predecir dónde está
# -----------------------------------------------------------------------------
def leave_one_out(reference_finder, image):
    """Error (mm) al predecir el centro de cada marcador con la homografía de los demás.

    Es una medida honesta de la precisión en un punto que NO se ha usado para calcular la
    homografía, como los puntos de la ficha. Necesita al menos 3 marcadores en la foto.
    Devuelve {id: error en mm}.
    """
    shape = image.shape[:2]
    markers, all_markers = reference_finder.detect_markers(image)
    sheet = reference_finder.sheet["corners"]
    if duplicate_sheet_ids((i for i, _ in all_markers), sheet):
        return {}
    lens = reference_finder.lens if reference_finder.lens is not None and reference_finder.lens.fits(shape) else None
    undistorted = {i: lens.undistort(q, shape) if lens else q for i, q in markers.items()}
    errors = {}
    if len(undistorted) < 3:
        return errors
    for left_out in undistorted:
        others = {i: q for i, q in undistorted.items() if i != left_out}
        homography, _, _ = fit_homography(others, sheet)
        if homography is not None:
            predicted = to_mm(undistorted[left_out], homography).mean(0)
            errors[left_out] = float(np.linalg.norm(predicted - sheet[left_out].mean(0)))
    return errors


def main():
    parser = argparse.ArgumentParser(description="Detecta la hoja de ArUco y pasa píxeles a milímetros.")
    parser.add_argument("source", type=Path, help="una foto o una carpeta con fotos")
    parser.add_argument("--point", type=float, nargs=2, action="append", metavar=("X", "Y"),
                        help="píxel a convertir a mm (se puede repetir)")
    parser.add_argument("--validate", action="store_true",
                        help="validación dejando un marcador fuera (necesita 3 o más marcadores por foto)")
    parser.add_argument("--output", type=Path, help="carpeta donde guardar las fotos con los ejes dibujados")
    parser.add_argument("--no-lens", action="store_true", help="no aplicar la calibración de la lente")
    args = parser.parse_args()

    cfg = dict(load_config()["aruco"])
    if args.no_lens:
        cfg["lens_calibration"] = None
    finder = ArucoReference(cfg)
    images = [args.source] if args.source.is_file() else sorted(
        p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.output:
        check_unique_image_stems(images)
        args.output.mkdir(parents=True, exist_ok=True)

    all_errors = []
    for path in images:
        image = read_image(path)
        reference = finder.locate(image)
        text = f"{path.name}: {describe_aruco(aruco_metadata(reference))}"
        if reference["found"]:
            text += f", máximo {reference['max_mm']:.2f} mm"
            text += ", con corrección de lente" if reference["lens"] else ", sin corrección de lente"
        if args.validate:
            errors = leave_one_out(finder, image)
            all_errors += list(errors.values())
            if errors:
                text += " | dejando uno fuera: " + ", ".join(f"{i}: {e:.2f} mm" for i, e in sorted(errors.items()))
        print(text)
        if reference["found"]:
            for x, y in args.point or []:
                (x_mm, y_mm), = to_mm([[x, y]], reference["homography"], reference["lens"], reference["shape"])
                print(f"   píxel ({x:.1f}, {y:.1f}) -> ({x_mm:.1f}, {y_mm:.1f}) mm")
        if args.output:
            write_image(args.output / f"{path.stem}_aruco.jpg", draw_reference(image.copy(), reference), 90)

    if args.validate and all_errors:
        errors = np.array(all_errors)
        print(f"\nValidación dejando un marcador fuera ({len(errors)} predicciones): media {errors.mean():.2f} mm, "
              f"mediana {np.median(errors):.2f} mm, máximo {errors.max():.2f} mm")


if __name__ == "__main__":
    main()
