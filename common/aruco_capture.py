"""Captura fotos de la hoja de ArUco con la cámara, para recalibrar la lente (aruco_calibrate.py).

Las fotos se guardan SIN anotaciones (frame original, PNG sin pérdida) en la carpeta aruco.captures
de config.yaml. Para calibrar bien, conviene fotografiar la hoja entera, con los 4 marcadores
nítidos, desde ángulos, inclinaciones y distancias variadas, y con los marcadores repartidos por
todo el campo de visión (también por las esquinas de la imagen, donde la distorsión es mayor).

Uso, desde la carpeta del proyecto:
    python -m common.aruco_capture                       # ventana: ESPACIO/s guarda, a = auto, q/ESC sale
    python -m common.aruco_capture --list-cams           # qué índices de cámara dan imagen
    python -m common.aruco_capture --cam 1 --interval 2  # además guarda solo cada 2 s
    python -m common.aruco_capture --no-gui --interval 1 --count 20
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from common.aruco_coords import IMAGE_EXTENSIONS, load_sheet, make_detector
from common.config import load_config, project_path
from common.images import read_image, write_image


def backends():
    """Backends de vídeo de OpenCV a probar, según el sistema operativo."""
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(cam, width, height):
    """Abre la cámara (índice o ruta). Devuelve None si no da imagen."""
    source = int(cam) if str(cam).isdigit() else cam
    for backend in backends():
        capture = cv2.VideoCapture(source, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = capture.read()  # algunos backends abren la cámara pero no dan imagen
            if ok:
                return capture
        capture.release()
    return None


def list_cameras(max_index=6):
    """Prueba los índices 0..max_index-1 e imprime los que dan imagen."""
    found = False
    for i in range(max_index):
        capture = open_camera(i, 640, 480)
        if capture is not None:
            found = True
            print(f"  cámara {i}: {int(capture.get(3))}x{int(capture.get(4))}")
            capture.release()
    if not found:
        print("  no se ha encontrado ninguna cámara")


def check_capture_resolution(folder, shape):
    """Comprueba que la resolución de la cámara coincide con las fotos de la carpeta."""
    height, width = shape
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            old_height, old_width = read_image(path).shape[:2]
            if (old_height, old_width) != shape:
                raise SystemExit(f"La cámara está a {width}x{height} y {path.name} está a "
                                 f"{old_width}x{old_height}. Indica otra carpeta con --out.")


def main():
    cfg = load_config()["aruco"]
    parser = argparse.ArgumentParser(description="Guarda fotos de la hoja de ArUco para calibrar la lente.")
    parser.add_argument("--cam", default="0", help="índice o ruta de la cámara (por defecto 0)")
    parser.add_argument("--list-cams", action="store_true", help="listar las cámaras disponibles y salir")
    parser.add_argument("--out", type=Path, default=project_path(cfg["captures"]), help="carpeta de salida")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080,
                        help="1920x1080: la resolución máxima de la cámara del proyecto (USB Camera, 16:9)")
    parser.add_argument("--interval", type=float, default=0, help="guardado automático cada N s (0 = no)")
    parser.add_argument("--count", type=int, default=0, help="parar tras N fotos (0 = sin límite)")
    parser.add_argument("--no-gui", action="store_true", help="sin ventana (necesita --interval)")
    args = parser.parse_args()

    if args.list_cams:
        list_cameras()
        return
    if args.no_gui and args.interval <= 0:
        sys.exit("Sin ventana hace falta --interval para guardar algo.")
    args.out.mkdir(parents=True, exist_ok=True)

    capture = open_camera(args.cam, args.width, args.height)
    if capture is None:
        sys.exit(f"No se pudo abrir la cámara {args.cam}. Prueba --list-cams para ver cuáles hay.")
    print(f"Cámara {args.cam}: {int(capture.get(3))}x{int(capture.get(4))} -> {args.out}", flush=True)

    # El detector solo informa en la previsualización: en la foto guardada no se dibuja nada
    detector = make_detector(load_sheet(project_path(cfg["sheet"]))["dictionary"])
    saved, auto, last = 0, args.interval > 0, 0.0
    shape = None

    def save(frame):
        nonlocal saved
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        path = args.out / f"img_{stamp}.png"
        write_image(path, frame)
        saved += 1
        print(f"[{saved}] {path}", flush=True)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            if frame.shape[:2] != shape:
                shape = frame.shape[:2]
                check_capture_resolution(args.out, shape)
            now = time.time()
            if auto and now - last >= args.interval:
                save(frame)
                last = now

            if not args.no_gui:
                _, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                view = frame.copy()
                text = f"marcadores: {0 if ids is None else len(ids)}  guardadas: {saved}  auto: {'ON' if auto else 'off'}"
                cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(view, "ESPACIO/s guardar - a auto - q salir", (10, view.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Captura", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("s")):
                    save(frame)
                if key == ord("a"):
                    auto = not auto
                    args.interval = args.interval or 1.0
                    last = 0.0

            if args.count and saved >= args.count:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print(f"Total guardadas: {saved}")


if __name__ == "__main__":
    main()
