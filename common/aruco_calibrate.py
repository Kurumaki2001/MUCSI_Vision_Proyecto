"""Calibración ligera de la lente a partir de fotos de la hoja de ArUco (de aruco_capture.py).

Estima y guarda en el fichero aruco.lens_calibration de config.yaml:
  - k1, k2: distorsión radial de la lente. NO usa las distancias reales: busca la distorsión con
    la que todas las esquinas encajan mejor en un plano (error de ajuste mínimo). Una lente
    distorsionada curva las rectas y las esquinas dejan de encajar en un plano.
  - edge_offset_px: cuánto hay que desplazar los bordes de los marcadores hacia fuera para
    compensar que la sobreexposición o el desenfoque "encogen" el negro. Este sí usa las
    distancias reales de la hoja (arucos_calibracion_A4.json): se elige el desplazamiento con el
    que, de media, las distancias medidas entre marcadores coinciden con las reales.

Para medir las distancias entre marcadores sin conocer la cámara, se ajusta a la vez una
homografía plano -> imagen y la posición y el giro de cada marcador en el plano (solve_plane).

Recalibrar si se cambia de cámara, de resolución, de enfoque o mucho la iluminación.

Uso, desde la carpeta del proyecto:
    python -m common.aruco_calibrate                  # fotos de aruco.captures
    python -m common.aruco_calibrate otra/carpeta --out prueba.json
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

from common.aruco_coords import IMAGE_EXTENSIONS, LensModel, edge_refine, make_detector, sharpness
from common.config import load_config, project_path
from common.images import read_image


# -----------------------------------------------------------------------------
# Geometría: marcadores coplanares sin conocer la cámara
# -----------------------------------------------------------------------------
def local_corners(size):
    """Esquinas de un marcador de lado 'size' centrado en 0, en el orden de OpenCV: TL, TR, BR, BL."""
    h = size / 2
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)


def rigid_fit(src, dst):
    """Giro y traslación 2D que mejor llevan src a dst (Procrustes con SVD). Devuelve (theta, t)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    r = (u @ vt).T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = (u @ vt).T
    return np.arctan2(r[1, 0], r[0, 0]), cd - r @ cs


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def project(G, points):
    """Aplica la homografía G a puntos (N, 2)."""
    p = np.c_[points, np.ones(len(points))] @ G.T
    return p[:, :2] / p[:, 2:3]


def solve_plane(detected, size):
    """Posición de cada marcador en el plano, suponiendo que todos están en el mismo plano.

    detected: {id: (4, 2) esquinas en px, sin distorsión}; size: lado del marcador.
    Incógnitas: la homografía G plano -> imagen (8 parámetros) y el giro y la posición (θ, tx, ty)
    de cada marcador salvo el de referencia (el mayor en la foto), que fija el origen. Se minimiza
    el error de reproyección de todas las esquinas a la vez con Levenberg-Marquardt.
    Devuelve ({id: centro (x, y) en las unidades de size}, error RMS de reproyección en px).
    """
    ids = sorted(detected)
    local = local_corners(size)
    ref = max(ids, key=lambda i: cv2.contourArea(detected[i].astype(np.float32)))
    H_ref, _ = cv2.findHomography(detected[ref], local)
    G0 = np.linalg.inv(H_ref)
    G0 /= G0[2, 2]

    # Valores iniciales: los demás marcadores se llevan al plano con la homografía del de referencia
    others = [i for i in ids if i != ref]
    poses0 = []
    for i in others:
        in_plane = cv2.perspectiveTransform(detected[i].reshape(-1, 1, 2), H_ref).reshape(-1, 2)
        theta, t = rigid_fit(local, in_plane)
        poses0 += [theta, t[0], t[1]]
    x0 = np.r_[G0.ravel()[:8], poses0]

    def unpack(x):
        G = np.r_[x[:8], 1.0].reshape(3, 3)
        poses = {ref: (0.0, np.zeros(2))}
        for k, i in enumerate(others):
            theta, tx, ty = x[8 + 3 * k: 11 + 3 * k]
            poses[i] = (theta, np.array([tx, ty]))
        return G, poses

    def residuals(x):
        G, poses = unpack(x)
        return np.concatenate([(project(G, local @ rotation(theta).T + t) - detected[i]).ravel()
                               for i, (theta, t) in poses.items()])

    x = least_squares(residuals, x0, method="lm").x if others else x0
    _, poses = unpack(x)
    return {i: poses[i][1] for i in ids}, float(np.sqrt(np.mean(residuals(x) ** 2)))


# -----------------------------------------------------------------------------
# Calibración
# -----------------------------------------------------------------------------
def main():
    cfg = load_config()["aruco"]
    parser = argparse.ArgumentParser(description="Calibra la distorsión de la lente con fotos de la hoja de ArUco.")
    parser.add_argument("folder", type=Path, nargs="?", default=project_path(cfg["captures"]),
                        help="carpeta con las fotos (por defecto aruco.captures)")
    parser.add_argument("--out", type=Path, default=project_path(cfg["lens_calibration"] or
                                                                 "aruco_sheet/calibracion_camara.json"))
    parser.add_argument("--min-markers", type=int, default=3, help="marcadores mínimos para usar una foto")
    parser.add_argument("--min-sharp", type=float, default=6.0, help="nitidez mínima para usar una foto")
    parser.add_argument("--no-offset", action="store_true", help="no estimar el desplazamiento de borde")
    args = parser.parse_args()

    sheet_path = project_path(cfg["sheet"])
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    size = sheet["marker_size_m"]
    centers_true = {int(k): np.array(v["center_m"]) for k, v in sheet["markers"].items()}
    true_distance = {(a, b): np.linalg.norm(centers_true[a] - centers_true[b])
                     for a, b in itertools.combinations(sorted(centers_true), 2)}

    # 1. Fotos útiles: nítidas y con suficientes marcadores de la hoja
    detector = make_detector(sheet.get("dictionary", "DICT_4X4_50"))
    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if args.folder.is_dir() else []
    frames, shape = [], None
    for path in files:
        gray = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2GRAY)
        if sharpness(gray) < args.min_sharp:
            continue
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        raw = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in centers_true}
        if len(raw) < args.min_markers:
            continue
        frames.append({"smooth": cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32), "raw": raw})
        shape = gray.shape
    if len(frames) < 3:
        raise SystemExit(f"Solo {len(frames)} fotos útiles en {args.folder} (≥{args.min_markers} marcadores y "
                         "nítidas). Captura más con python -m common.aruco_capture")
    print(f"{len(frames)} fotos útiles de {len(files)} ({shape[1]}x{shape[0]})")

    def refined(offset):
        """Esquinas refinadas de cada foto con un desplazamiento de borde dado."""
        out = []
        for frame in frames:
            corners = {i: q for i, q0 in frame["raw"].items()
                       if (q := edge_refine(frame["smooth"], q0, offset)) is not None}
            if len(corners) >= 2:
                out.append(corners)
        return out

    def evaluate(lens, detections):
        """(error de ajuste RMS en px, errores relativos de las distancias en %)."""
        rms2, errors = [], []
        for corners in detections:
            centers, rms = solve_plane({i: lens.undistort(q, shape) for i, q in corners.items()}, size)
            rms2.append(rms ** 2)
            errors += [(np.linalg.norm(centers[a] - centers[b]) / true_distance[(a, b)] - 1) * 100
                       for a, b in true_distance if a in centers and b in centers]
        return float(np.sqrt(np.mean(rms2))), np.array(errors)

    def report(name, lens, detections):
        rms, e = evaluate(lens, detections)
        print(f"  {name:30s} ajuste {rms:.2f} px | distancias: sesgo {e.mean():+.2f} %, "
              f"desv. {e.std():.2f} %, máx. {np.abs(e).max():.2f} %")

    detections = refined(0.0)
    print("Error frente a las distancias reales de la hoja:")
    report("sin corrección", LensModel(width=shape[1], height=shape[0]), detections)

    # 2. Distorsión radial: la que mejor hace encajar las esquinas en un plano
    result = minimize(lambda k: evaluate(LensModel(k[0], k[1]), detections)[0], [0.0, 0.0],
                      method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5})
    lens = LensModel(float(result.x[0]), float(result.x[1]), 0.0, shape[1], shape[0])
    report(f"+ distorsión k1={lens.k1:.3f} k2={lens.k2:.3f}", lens, detections)

    # 3. Desplazamiento de borde: el que anula el sesgo medio de las distancias
    if not args.no_offset:
        def squared_bias(offset):
            lens.edge_offset_px = offset
            return evaluate(lens, refined(offset))[1].mean() ** 2
        result = minimize_scalar(squared_bias, bounds=(-3, 3), method="bounded", options={"xatol": 0.02})
        lens.edge_offset_px = float(result.x)
        report(f"+ borde {lens.edge_offset_px:+.2f} px", lens, refined(lens.edge_offset_px))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.out, {"n_images": len(frames), "sheet": sheet_path.name})
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
