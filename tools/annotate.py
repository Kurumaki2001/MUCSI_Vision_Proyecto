"""Herramienta para anotar fotos reales con OpenCV: clic en cada punto y su número con el teclado.

Uso, desde la carpeta del proyecto:
    python -m tools.annotate data/real           # anota las fotos que aún no tienen anotación
    python -m tools.annotate data/real --redo    # vuelve a abrir también las ya anotadas, para corregirlas

Controles, con la ventana de la foto activa:
    clic izquierdo   nuevo punto en esa posición
    0-9              escribe el número del último punto
    Retroceso        borra la última cifra; si el punto no tiene cifras, borra el punto
    Intro            guarda y pasa a la siguiente foto
    Esc              guarda y sale
La ventana se puede agrandar para hacer clic con más precisión.

Guarda <carpeta>/annotations/<foto>.json con el formato común: puntos (id, x, y) y números (id,
value) sin caja, emparejados por id. Es la "anotación real mínima" del enunciado: sirve para
evaluar los puntos y las parejas, pero no la lectura de números, que necesita sus cajas.
"""

import argparse
from pathlib import Path

import cv2

from common import json_format
from common.images import read_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
KEY_ENTER, KEY_ESC, KEY_BACKSPACE = 13, 27, 8


class AnnotationState:
    """Puntos marcados y el número escrito para cada uno. No usa ventanas, así se puede probar."""

    def __init__(self):
        self.dots = []  # lista de [x, y, texto del número]

    def add_dot(self, x, y):
        self.dots.append([float(x), float(y), ""])

    def type_digit(self, digit):
        if self.dots:
            self.dots[-1][2] += digit

    def backspace(self):
        """Borra la última cifra del último punto o, si no tiene cifras, el punto."""
        if not self.dots:
            return
        if self.dots[-1][2]:
            self.dots[-1][2] = self.dots[-1][2][:-1]
        else:
            self.dots.pop()

    def to_annotation(self, image_name, width, height):
        """Anotación en el formato común. Los puntos sin número quedan como puntos sin pareja."""
        dots, numbers, pairs, unpaired = [], [], [], []
        for i, (x, y, text) in enumerate(self.dots):
            dots.append({"id": i, "x": round(x, 1), "y": round(y, 1)})
            if text:
                numbers.append({"id": i, "value": int(text)})
                pairs.append({"dot": i, "number": i, "distance": None})  # sin caja no hay distancia
            else:
                unpaired.append(i)
        return json_format.create(image_name, width, height, dots, numbers, pairs, unpaired_dots=unpaired)

    @classmethod
    def from_annotation(cls, data):
        """Recupera una anotación guardada, para corregirla."""
        state = cls()
        number_values = {number["id"]: number.get("value") for number in data["numbers"]}
        values = {pair["dot"]: number_values[pair["number"]] for pair in data["pairs"]}
        for dot in data["dots"]:
            value = values.get(dot["id"])
            state.dots.append([dot["x"], dot["y"], "" if value is None else str(value)])
        return state


def render(image, state):
    """Copia de la foto con los puntos marcados y sus números; el último, en otro color."""
    output = image.copy()
    size = max(1, round(image.shape[0] / 600))  # grosor proporcional a la foto
    for i, (x, y, text) in enumerate(state.dots):
        color = (0, 0, 255) if i == len(state.dots) - 1 else (0, 170, 0)  # BGR: rojo el último, verde el resto
        center = (round(x - 0.5), round(y - 0.5))  # -0.5: OpenCV pone el centro del píxel (0, 0) en (0, 0)
        cv2.circle(output, center, 4 * size, color, size, cv2.LINE_AA)
        cv2.putText(output, text or "?", (center[0] + 6 * size, center[1] - 6 * size),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * size, color, size, cv2.LINE_AA)
    cv2.putText(output, f"{len(state.dots)} puntos | Intro: siguiente | Esc: salir", (10, 30 * size),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * size, (255, 0, 0), size, cv2.LINE_AA)
    return output


def on_mouse(event, x, y, flags, state):
    """Clic izquierdo: nuevo punto. Con una ventana redimensionable, OpenCV ya da (x, y) en
    píxeles de la imagen; se suma medio píxel para usar el centro del píxel pulsado."""
    if event == cv2.EVENT_LBUTTONDOWN:
        state.add_dot(x + 0.5, y + 0.5)


def annotate_image(image_path, output_path, state):
    """Abre la ventana de anotación de una foto. Devuelve False si el usuario pulsa Esc (salir)."""
    image = read_image(image_path)
    height, width = image.shape[:2]
    window = f"Anotar {image_path.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = min(1.0, 1400 / width, 900 / height)  # que quepa en pantalla
    cv2.resizeWindow(window, round(width * scale), round(height * scale))
    cv2.setMouseCallback(window, on_mouse, state)
    keep_going = True
    while True:
        cv2.imshow(window, render(image, state))
        key = cv2.waitKey(30)
        if key == -1:
            continue
        key &= 0xFF
        if ord("0") <= key <= ord("9"):
            state.type_digit(chr(key))
        elif key == KEY_BACKSPACE:
            state.backspace()
        elif key in (KEY_ENTER, KEY_ESC):
            keep_going = key == KEY_ENTER
            break
    cv2.destroyWindow(window)
    json_format.save(state.to_annotation(image_path.name, width, height), output_path)
    print(f"Guardada {output_path} ({len(state.dots)} puntos)")
    return keep_going


def main():
    parser = argparse.ArgumentParser(description="Anota fotos reales: clic en cada punto y su número.")
    parser.add_argument("folder", type=Path, help="carpeta con las fotos (por ejemplo, data/real)")
    parser.add_argument("--redo", action="store_true", help="abrir también las fotos ya anotadas")
    args = parser.parse_args()

    images = sorted(path for path in args.folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No hay fotos en {args.folder}.")
    output_folder = args.folder / "annotations"
    output_folder.mkdir(exist_ok=True)
    for image_path in images:
        output_path = output_folder / f"{image_path.stem}.json"
        if output_path.exists() and not args.redo:
            continue
        state = AnnotationState.from_annotation(json_format.load(output_path)) if output_path.exists() else AnnotationState()
        if not annotate_image(image_path, output_path, state):
            break


if __name__ == "__main__":
    main()
