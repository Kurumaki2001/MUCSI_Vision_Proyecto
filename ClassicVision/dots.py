"""Detección de puntos con contornos y reglas de forma (sin aprendizaje).

Un punto de la ficha es un círculo relleno. Para cada mancha de tinta (contorno exterior de
cv2.findContours) se miden:
  - holes: agujeros (contornos hijos, con RETR_CCOMP). Un punto no tiene; un "0", un "8" o una
    circunferencia hueca sí.
  - fill: área / área del círculo mínimo que la contiene (cv2.minEnclosingCircle). Un círculo
    relleno llena casi todo su círculo; un "1" o un "7", muy poco.
  - roundness: cociente entre los ejes menor y mayor de la elipse equivalente (momentos de
    segundo orden). 1 = redondo; un trazo alargado da valores bajos.
  - solidity: área / área de la envolvente convexa (cv2.convexHull). Un círculo es convexo.

Pasos:
  1. Candidatos: manchas que cumplen las reglas de forma.
  2. Tamaño: en una ficha todos los puntos miden casi lo mismo. Se toma la mediana del radio de
     los candidatos y se descartan los que se salen de una banda alrededor (los dos puntos de
     "Nombre:" son mucho más pequeños, por ejemplo).
  3. Puntos pegados a su número: forman una sola mancha que no es redonda. Se separan con una
     apertura morfológica (erosión + dilatación) con un disco algo menor que el punto: los
     trazos finos de los dígitos desaparecen y el punto sobrevive. Si lo que queda es un círculo
     del tamaño esperado, es un punto.
  4. Los puntos encontrados se borran de la máscara de tinta (con un pequeño margen) y lo que
     queda se pasa a la lectura de números.

Coordenadas: como en las anotaciones, el píxel (0, 0) ocupa el cuadrado [0, 1) x [0, 1), así que
el centro que da OpenCV (centrado en el píxel) se desplaza medio píxel.
"""

import math

import cv2
import numpy as np

from ClassicVision.preprocess import odd


def shape_features(mask):
    """Medidas de forma de una mancha (máscara binaria recortada a su caja, uint8 0/255).

    Devuelve un diccionario con area, holes, fill, roundness, solidity, radius y el centro (cx, cy)
    relativo a la máscara, o None si la mancha está vacía.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outer = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]
    main = max(outer, key=lambda i: cv2.contourArea(contours[i]))
    contour = contours[main]
    area = float(cv2.countNonZero(mask))  # área en píxeles (más fiable que la del polígono en manchas pequeñas)
    holes = sum(1 for i, h in enumerate(hierarchy[0])
                if h[3] == main and cv2.contourArea(contours[i]) + cv2.arcLength(contours[i], True) / 2 >= 0.04 * area)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    radius += 0.5  # el contorno pasa por los centros de los píxeles del borde
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return None
    mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
    spread = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4 * mu11 ** 2))
    major, minor = mu20 + mu02 + spread, mu20 + mu02 - spread  # proporcionales a los ejes al cuadrado
    hull_area = cv2.contourArea(cv2.convexHull(contour)) + cv2.arcLength(contour, True) / 2 + 1
    return {"area": area, "holes": holes, "radius": radius,
            "fill": area / (math.pi * radius ** 2),
            "roundness": math.sqrt(max(minor, 0.0) / major) if major > 0 else 0.0,
            "solidity": min(1.0, area / hull_area),
            "cx": moments["m10"] / moments["m00"] + 0.5, "cy": moments["m01"] / moments["m00"] + 0.5}


def looks_like_dot(f, cfg):
    """Reglas de forma de un círculo relleno."""
    return (f["holes"] == 0 and f["radius"] >= cfg["dot_min_radius_px"] and f["fill"] >= cfg["dot_min_fill"]
            and f["roundness"] >= cfg["dot_min_roundness"] and f["solidity"] >= cfg["dot_min_solidity"])


def dot_score(f):
    """Confianza del punto (0-1): lo redondo y lleno que es."""
    return round(float(min(1.0, f["fill"]) * f["roundness"] * f["solidity"]), 4)


def make_dot(f, x0, y0):
    """Punto en coordenadas de la foto a partir de sus medidas (relativas a la caja en x0, y0)."""
    x, y, r = f["cx"] + x0, f["cy"] + y0, f["radius"]
    return {"x": round(x, 2), "y": round(y, 2), "r": r, "confidence": dot_score(f),
            "box": [round(x - r, 2), round(y - r, 2), round(x + r, 2), round(y + r, 2)]}


def blobs(ink):
    """Manchas de tinta: [(x, y, w, h, máscara recortada solo con esa mancha)], con componentes conexas."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = []
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        mask = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
        out.append((int(x), int(y), int(w), int(h), mask))
    return out


def split_merged(mask, radius, cfg):
    """Busca puntos dentro de una mancha que no es un punto (un punto pegado a su número).

    Apertura con un disco de diámetro split_kernel x radio: los trazos más finos desaparecen. De
    lo que queda, se acepta como punto cada trozo redondo de radio parecido al esperado y que,
    en la mancha original, esté rodeado de papel en casi todo su contorno (un punto solo toca
    a su número por un lado; el cruce de trazos de un "4" en negrita sobrevive a la apertura,
    pero tiene tinta alrededor).
    Devuelve [(medidas, x_offset, y_offset)] en coordenadas de la máscara.
    """
    size = odd(cfg["split_kernel"] * radius)
    pad = size + round(2 * radius)  # margen para la apertura y para mirar alrededor del punto
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    opened = cv2.morphologyEx(padded, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    found = []
    for x, y, w, h, part in blobs(opened):
        f = shape_features(part)
        if not (f and f["holes"] == 0 and f["roundness"] >= cfg["dot_min_roundness"] and f["fill"] >= cfg["dot_min_fill"]
                and cfg["dot_size_band"][0] <= f["radius"] / radius <= cfg["dot_size_band"][1]):
            continue
        if touches_on_one_side(padded, f["cx"] + x - 0.5, f["cy"] + y - 0.5, f["radius"], cfg):
            found.append((f, x - pad, y - pad))
    return found


def touches_on_one_side(mask, cx, cy, r, cfg, sectors=12):
    """True si, alrededor del círculo (cx, cy, r), la tinta de la mancha está solo hacia un lado.

    Se mira un anillo de 1,3 a 1,7 radios dividido en 12 sectores de 30°. Un punto pegado a su
    número solo tiene tinta en los sectores que miran al número: unos pocos y seguidos. El
    cruce de trazos de un dígito en negrita (el centro de un "4") tiene trazos en varias
    direcciones, así que ocupa sectores separados o casi todos.
    """
    yy, xx = np.nonzero(mask)
    distance = np.hypot(xx - cx, yy - cy)
    in_ring = (distance >= 1.3 * r) & (distance <= 1.7 * r)
    angles = np.arctan2(yy[in_ring] - cy, xx[in_ring] - cx)
    counts = np.bincount(((angles + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors, minlength=sectors)
    # Píxeles de un sector completo del anillo, para decidir si el sector tiene tinta de verdad
    per_sector = np.pi * ((1.7 * r) ** 2 - (1.3 * r) ** 2) / sectors
    occupied = counts >= 0.25 * per_sector
    if not occupied.any():
        return True  # un círculo aislado que la apertura ha dejado igual
    if occupied.sum() > cfg["split_max_sectors"]:
        return False
    # ¿Forman un único tramo seguido? (en un círculo: como mucho un paso de libre a ocupado)
    starts = np.sum(occupied & ~np.roll(occupied, 1))
    return starts == 1


def detect_dots(ink, cfg):
    """Puntos de la máscara de tinta. Devuelve (puntos, tinta sin los puntos, radio típico)."""
    all_blobs = blobs(ink)
    candidates, others = [], []
    for x, y, w, h, mask in all_blobs:
        if w * h < cfg["min_blob_area"]:
            continue
        f = shape_features(mask)
        if f is None:
            continue
        (candidates if looks_like_dot(f, cfg) else others).append((f, x, y, w, h, mask))

    dots = []
    radius = float(np.median([f["radius"] for f, *_ in candidates])) if candidates else None
    if radius is not None:
        low, high = cfg["dot_size_band"]
        dots = [make_dot(f, x, y) for f, x, y, *_ in candidates if low <= f["radius"] / radius <= high]
        # Manchas más grandes que un punto que no son puntos: quizá un punto pegado a un número
        dot_area = math.pi * radius ** 2
        for f, x, y, w, h, mask in others:
            if f["area"] >= cfg["split_min_area"] * dot_area:
                dots += [make_dot(g, x + dx, y + dy) for g, dx, dy in split_merged(mask, radius, cfg)]

    # Se borran los puntos de la tinta (con margen) para que no se lean como parte de los números
    text_ink = ink.copy()
    for dot in dots:
        cv2.circle(text_ink, (round(dot["x"] - 0.5), round(dot["y"] - 0.5)),
                   round(dot["r"] * cfg["erase_margin"] + 1), 0, -1)
    return dots, text_ink, radius
