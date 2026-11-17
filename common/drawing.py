"""Dibujo de anotaciones o resultados (formato común) sobre una imagen, con OpenCV.

Lo usan el generador, para revisar a ojo que las cajas cuadran, y más adelante la inferencia de
los dos enfoques. Los colores van en BGR, el orden de OpenCV.
"""

import cv2
import numpy as np

GREEN = (0, 170, 0)      # puntos
BLUE = (255, 90, 0)      # números
ORANGE = (0, 140, 255)   # parejas
RED = (0, 0, 230)        # elementos sin pareja
PURPLE = (200, 0, 150)   # línea que une los puntos en orden


def _pixel(x, y):
    """Coordenadas enteras para las funciones de dibujo de OpenCV."""
    return (round(x), round(y))


def _box_center(box):
    return _pixel((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def draw_annotation(image, data, thickness=2, join_in_order=False):
    """Devuelve una copia de la imagen con los puntos, los números y las parejas dibujados.

    - Puntos en verde (su caja si la tienen; si no, un círculo) y números en azul con el valor
      encima; los que no tienen pareja, en rojo.
    - Una línea naranja de cada punto al centro de la caja de su número.
    - Con join_in_order=True, además, una línea morada que une los puntos en orden numérico,
      como al resolver la ficha.
    """
    output = image.copy()
    dots = {dot["id"]: dot for dot in data["dots"]}
    numbers = {number["id"]: number for number in data["numbers"]}
    unpaired_dots = set(data.get("unpaired_dots", []))
    unpaired_numbers = set(data.get("unpaired_numbers", []))
    font_scale = max(0.4, output.shape[0] / 2500)

    for pair in data["pairs"]:
        dot, number = dots[pair["dot"]], numbers[pair["number"]]
        cv2.line(output, _pixel(dot["x"], dot["y"]), _box_center(number["box"]), ORANGE,
                 max(1, thickness // 2), cv2.LINE_AA)

    if join_in_order:
        ordered = sorted((numbers[pair["number"]]["value"], pair["dot"]) for pair in data["pairs"]
                         if numbers[pair["number"]].get("value") is not None)
        path = np.array([_pixel(dots[i]["x"], dots[i]["y"]) for _, i in ordered], dtype=np.int32)
        if len(path) > 1:
            cv2.polylines(output, [path], False, PURPLE, thickness, cv2.LINE_AA)

    for dot in data["dots"]:
        color = RED if dot["id"] in unpaired_dots else GREEN
        if dot.get("box"):
            x0, y0, x1, y1 = dot["box"]
            cv2.rectangle(output, _pixel(x0, y0), _pixel(x1, y1), color, thickness)
        else:
            cv2.circle(output, _pixel(dot["x"], dot["y"]), 5, color, thickness, cv2.LINE_AA)

    for number in data["numbers"]:
        if not number.get("box"):
            continue
        color = RED if number["id"] in unpaired_numbers else BLUE
        x0, y0, x1, y1 = number["box"]
        cv2.rectangle(output, _pixel(x0, y0), _pixel(x1, y1), color, thickness)
        text = "?" if number.get("value") is None else str(number["value"])
        cv2.putText(output, text, _pixel(x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    return output
