"""Preprocesado: dónde está la hoja (papel) y qué píxeles son tinta.

1. paper_mask: separa la hoja de la mesa. La hoja es la zona clara y poco saturada (en HSV:
   V alto y S bajo); la mesa suele ser más oscura o de color (madera). Se umbraliza con Otsu una
   puntuación V - S sobre una versión reducida y suavizada de la foto, se cierran los huecos del
   texto con una operación morfológica de cierre y se rellenan los contornos exteriores grandes
   (findContours + drawContours). Si no aparece ninguna zona grande de papel (por ejemplo, la
   hoja ocupa toda la foto), se usa la foto entera.

2. ink_mask: separa la tinta del papel aunque la luz no sea uniforme (degradados, sombras).
   - Fondo: un cierre morfológico con un núcleo mayor que cualquier punto o trazo "borra" la
     tinta y deja solo el color del papel en cada zona. Es un filtro paso bajo: se queda con la
     iluminación, que varía despacio. Se calcula a 1/4 de resolución, porque es muy suave.
   - Oscuridad: 1 - foto / fondo, canal a canal. Dividir por el fondo anula la iluminación (una
     sombra oscurece por igual papel y tinta). De los tres canales se toma el más oscuro, para
     que la tinta de color (roja, azul...) cuente igual que la negra.
   - Doble umbral sobre esa oscuridad (histéresis): ver ink_mask.
"""

import cv2
import numpy as np


def odd(value):
    """Entero impar >= 3 (los núcleos de OpenCV necesitan un centro)."""
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


def paper_mask(image, cfg):
    """Máscara (uint8, 0/255) de la hoja. Devuelve (máscara, polígono de la hoja o None)."""
    height, width = image.shape[:2]
    scale = min(1.0, cfg["paper_work_side"] / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)  # quita el texto pequeño y el ruido antes de umbralizar
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    score = cv2.subtract(hsv[..., 2], hsv[..., 1])  # claro y poco saturado = papel
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(0.03 * max(small.shape)),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # rellena las letras y los puntos
    # Apertura: corta las uniones finas entre la hoja y otras zonas claras (bordes de la mesa,
    # otro papel asomando), para que no formen una sola zona con la mesa en medio.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Zonas de papel: contornos exteriores grandes, rellenos con su propia forma. No se usa la
    # envolvente convexa: si la zona de papel tiene forma de U (hoja + tiras claras en los bordes
    # de la foto), la envolvente metía la mesa de en medio.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) >= cfg["paper_min_fraction"] * mask.size]
    if not big:
        return np.full((height, width), 255, np.uint8), None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, big, -1, 255, -1)
    hulls = [cv2.convexHull(c) for c in big]
    # Se encoge un poco para no tomar el borde de la hoja (y su sombra) por tinta.
    filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(cfg["paper_erode"] * max(small.shape)),) * 2))
    full = cv2.resize(filled, (width, height), interpolation=cv2.INTER_NEAREST)
    largest = max(hulls, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True) / scale
    return full, polygon.reshape(-1, 2)


def darkness(image, cfg):
    """Oscuridad de cada píxel respecto al papel que lo rodea: 0 = papel, 1 = negro."""
    height, width = image.shape[:2]
    small = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    size = odd(cfg["background_kernel"] * min(height, width) * 0.25)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), size / 4)
    background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ratio = image.astype(np.float32) / np.maximum(background, 1.0)
    return np.clip(1.0 - ratio.min(axis=2), 0.0, 1.0)


def ink_mask(dark, paper, cfg):
    """Tinta (uint8, 0/255) dentro de la hoja, con doble umbral (histéresis, como en Canny).

    Con un solo umbral alto, los trazos finos y grises (Courier, fotos desenfocadas) se rompen;
    con uno bajo, entran manchas del papel. Por eso: una mancha que supera el umbral bajo es
    tinta solo si alguno de sus píxeles supera el alto.
    """
    low, high = cfg["ink_threshold"]
    weak = cv2.bitwise_and((dark > low).astype(np.uint8) * 255, paper)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    # Oscuridad máxima de cada mancha (np.maximum.at: máximo por etiqueta)
    peak = np.zeros(count, np.float32)
    np.maximum.at(peak, labels.ravel(), dark.ravel())
    keep = peak > high
    keep[0] = False  # la etiqueta 0 es el fondo
    ink = keep[labels]
    ratio = cfg.get("ink_relative", 0)
    if ratio:
        # Umbral relativo a cada mancha: un píxel es tinta si llega a 'ratio' de la oscuridad máxima
        # de su mancha. Un número impreso en gris claro (máximo 0,3) se queda con lo que pase de
        # 0,15; un dígito negro y grueso (máximo 0,9), con lo que pase de 0,45, así que su halo
        # desenfocado no lo pega al dígito vecino.
        ink &= dark > ratio * peak[labels]
    return ink.astype(np.uint8) * 255
