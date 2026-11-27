"""Análisis de cada rotulador detectado: silueta, posición, orientación y color.

YOLO solo da una caja alrededor de cada rotulador. A partir de ella:

1. Silueta (máscara). Con SAM (en pens/pipeline.py, la caja de YOLO es la pista) o, sin SAM,
   con GrabCut (tema 5): la caja inicializa el algoritmo; lo de fuera es fondo y GrabCut separa
   el rotulador de la mesa dentro de la caja con modelos de color (GMM) y un corte de grafo.

2. Geometría (pen_geometry). El rectángulo de área mínima de la silueta (cv2.minAreaRect, tema 4)
   da el centro, el largo, el ancho y el ángulo del eje largo. Un rotulador es alargado: el robot
   necesita su centro (punto de agarre) y la dirección del eje para orientar la pinza.

3. Color (pen_color), en HSV (tema 4). El Edding 1200 tiene el cuerpo blanco/gris y el color de
   la tinta en el capuchón y en el tapón del otro extremo; además, el cuerpo lleva letras negras
   impresas. Por eso no sirve el color medio: dentro de la silueta se cuentan
     - píxeles de color: saturación (S) y brillo (V) suficientes y tono (H) de rojo (en OpenCV el
       rojo está en los dos extremos, 0-10 y 170-179) o de azul (~95-130);
     - píxeles negros: brillo muy bajo, sea cual sea el tono.
   Si hay bastante rojo o azul, gana el que más tenga; si no, es negro si hay bastante negro
   (el capuchón negro es mucho más que las letras del cuerpo); si no, "desconocido".
   Las fracciones se dan también en el resultado, para poder ajustar los umbrales con fotos reales.
"""

import cv2
import numpy as np

COLOR_NAMES = ("azul", "rojo", "negro")


# -----------------------------------------------------------------------------
# Silueta sin SAM: GrabCut
# -----------------------------------------------------------------------------
def segment_grabcut(image, box, iterations=5, margin=0.1):
    """Máscara (uint8 0/255, tamaño de la foto) del objeto de la caja [x0, y0, x1, y1] con GrabCut.

    Se trabaja sobre un recorte algo mayor que la caja (más rápido que la foto entera); el
    rectángulo de inicialización es la caja de YOLO dentro de ese recorte.
    """
    height, width = image.shape[:2]
    full = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return full
    pad_x, pad_y = margin * (x1 - x0) + 5, margin * (y1 - y0) + 5
    cx0, cy0 = int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y))
    cx1, cy1 = int(min(width, x1 + pad_x)), int(min(height, y1 + pad_y))
    crop = np.ascontiguousarray(image[cy0:cy1, cx0:cx1])
    rect = (int(x0 - cx0), int(y0 - cy0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))
    if rect == (0, 0, crop.shape[1], crop.shape[0]):
        return full  # GrabCut necesita píxeles de fondo fuera de la caja.
    mask = np.zeros(crop.shape[:2], np.uint8)
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, rect, background, foreground, iterations, cv2.GC_INIT_WITH_RECT)
    # 1 = primer plano seguro, 3 = primer plano probable (igual que en el ejercicio de clase)
    full[cy0:cy1, cx0:cx1] = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return largest_blob(full)


def largest_blob(mask):
    """Se queda con la mancha más grande de la máscara (quita restos sueltos) y rellena sus huecos."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return clean


# -----------------------------------------------------------------------------
# Geometría
# -----------------------------------------------------------------------------
def pen_geometry(mask):
    """Centro, eje y medidas de la silueta con el rectángulo de área mínima.

    Devuelve None si la máscara está vacía, o un diccionario con (todo en píxeles de la foto):
      center (x, y), length, width, angle_deg (dirección del eje largo, de 0 a 180, medida desde el
      eje X de la imagen hacia abajo, como las coordenadas de OpenCV), ends (los dos extremos del
      eje largo) y box_points (las 4 esquinas del rectángulo, para dibujarlo).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), _ = rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect)
    # Eje largo: el lado más largo del rectángulo (así no dependemos del convenio de ángulos de OpenCV)
    side_a, side_b = corners[1] - corners[0], corners[2] - corners[1]
    axis = side_a if np.linalg.norm(side_a) >= np.linalg.norm(side_b) else side_b
    length, width = max(w, h), min(w, h)
    unit = axis / (np.linalg.norm(axis) or 1.0)
    angle = float(np.degrees(np.arctan2(unit[1], unit[0])) % 180.0)
    center = np.array([cx, cy]) + 0.5  # +0,5: convenio de la anotación (el píxel 0 ocupa [0, 1))
    ends = [center - unit * length / 2, center + unit * length / 2]
    return {"center": center.tolist(), "length": float(length), "width": float(width), "angle_deg": angle,
            "ends": [e.tolist() for e in ends], "box_points": (corners + 0.5).tolist()}


# -----------------------------------------------------------------------------
# Color
# -----------------------------------------------------------------------------
def color_fractions(image, mask, cfg):
    """Fracción de la silueta que es azul, roja o negra (píxeles HSV que cumplen cada regla)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    inside = mask > 0
    total = int(inside.sum())
    if total == 0:
        return {name: 0.0 for name in COLOR_NAMES}
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = inside & (s >= cfg["min_saturation"]) & (v >= cfg["min_value"])
    blue = vivid & (h >= cfg["blue_hue"][0]) & (h <= cfg["blue_hue"][1])
    red = vivid & ((h <= cfg["red_hue"][0]) | (h >= cfg["red_hue"][1]))
    black = inside & (v <= cfg["black_max_value"])
    return {"azul": float(blue.sum()) / total, "rojo": float(red.sum()) / total, "negro": float(black.sum()) / total}


def pen_color(image, mask, cfg):
    """(color, fracciones): "azul", "rojo", "negro" o "desconocido", según las reglas del módulo."""
    fractions = color_fractions(image, mask, cfg)
    colored = max(("azul", "rojo"), key=lambda name: fractions[name])
    if fractions[colored] >= cfg["min_color_fraction"]:
        return colored, fractions
    if fractions["negro"] >= cfg["min_black_fraction"]:
        return "negro", fractions
    return "desconocido", fractions
