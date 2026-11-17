"""Generador de datos sintéticos: fotos cenitales de una ficha A4 de "unir los puntos".

Cada foto se construye en cuatro pasos, con OpenCV y NumPy:
  1. Hoja plana. Se dibujan, en píxeles (milímetros x px_per_mm), los puntos rellenos y junto a
     cada uno su número, sin solapes salvo un porcentaje de casos difíciles. Después se añaden
     distractores que no se etiquetan: título, letras sueltas y círculos huecos.
  2. Foto. La hoja se coloca sobre una mesa como la vería una cámara cenital, con un pequeño
     giro y algo de perspectiva. Esa transformación es una homografía, que se calcula con
     cv2.getPerspectiveTransform y se aplica con cv2.warpPerspective.
  3. Efectos de cámara: iluminación irregular, sombra, balance de blancos, desenfoque, ruido y
     compresión JPEG.
  4. Anotación. Las posiciones y cajas de la hoja se pasan a la foto con la misma homografía
     (cv2.perspectiveTransform), así que siguen siendo exactas.

Los textos se escriben con fuentes TrueType (Arial, Times...) mediante cv2.FontFace, que
OpenCV 5 incluye.

Coordenadas: igual que en las anotaciones, el píxel (0, 0) ocupa el cuadrado [0, 1) x [0, 1).
OpenCV pone el centro de ese píxel en (0, 0); por eso, al dibujar círculos, se resta medio píxel.

Salida (generator.output en config.yaml):
  {train,val,test}/images/photo_XXXXX.jpg
  {train,val,test}/annotations/photo_XXXXX.json   formato común (common/json_format.py)
  samples/photo_XXXXX.jpg                         algunas fotos de train con las cajas dibujadas
  config_used.yaml                                configuración con la que se han generado

Uso, desde la carpeta del proyecto:
    python -m data.generate           # dataset completo
    python -m data.generate --quick   # dataset mini para la prueba rápida
"""

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import json_format
from common.config import clear_generated_folder, load_config, project_path
from common.drawing import draw_annotation
from common.images import read_image, write_image
from common.matching import point_box_distance

SPLITS = ("train", "val", "test")
MM_PER_PT = 25.4 / 72   # un punto tipográfico mide 0,353 mm
AMBIGUITY_MARGIN = 1.1  # cada número debe quedar al menos un 10 % más cerca de su punto que de
                        # cualquier otro, y al revés, para que la verdad no sea ambigua
SHIFT = 4               # cv2.circle admite coordenadas con 4 bits de decimales (valores x 16)

# Dónde buscar las fuentes TrueType (Windows, Linux y macOS) y las que trae OpenCV 5.
FONT_FOLDERS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
                Path(os.environ.get("LOCALAPPDATA", "-")) / "Microsoft" / "Windows" / "Fonts",
                Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
                Path("/Library/Fonts"), Path("/System/Library/Fonts")]
BUILT_IN_FONTS = {"sans", "italic", "uni"}

# Distractores. Las letras sueltas excluyen las que se confunden con un dígito (O, o, I, l, i, j,
# J, S, s, B, b, Z, z, G, g, q, D, d): una "O" aislada no se distingue de un "0" y el modelo
# recibiría ejemplos contradictorios. Dentro de palabras sí aparecen todas.
LETTERS = "ACEFHKLMNPRTUVWXYacefhkmnprtuvwxy"
WORDS = ["sol", "casa", "flor", "pez", "gato", "nube", "luna", "tren", "mar", "árbol", "hoja", "pato"]
TITLES = ["Une los puntos", "Une los puntos en orden", "Une los puntos y colorea",
          "Completa el dibujo", "¿Qué animal se esconde?", "Ficha de repaso", "El dibujo escondido"]
FIELDS = ["Nombre:", "Fecha:", "Curso:"]

# Colores en BGR, el orden de OpenCV.
DARK_INKS = [(15, 15, 15), (30, 30, 30), (50, 45, 45), (60, 60, 60), (45, 25, 25)]
COLOR_INKS = [(30, 30, 190), (170, 60, 30), (50, 120, 20), (140, 40, 140), (0, 100, 210), (130, 120, 0)]
TABLE_COLORS = [(75, 110, 150), (55, 80, 115), (110, 150, 185), (100, 95, 95), (65, 60, 60),
                (195, 200, 200), (40, 35, 35)]


# -----------------------------------------------------------------------------
# Azar. Todas las funciones reciben el generador aleatorio (rng) de la foto que se está creando.
# -----------------------------------------------------------------------------
def random_float(rng, bounds):
    """Número real al azar en bounds = [mínimo, máximo]."""
    return float(rng.uniform(bounds[0], bounds[1]))


def random_int(rng, bounds):
    """Entero al azar en bounds = [mínimo, máximo], ambos incluidos."""
    return int(rng.integers(bounds[0], bounds[1] + 1))


def random_item(rng, items):
    """Elemento al azar de una lista."""
    return items[int(rng.integers(len(items)))]


# -----------------------------------------------------------------------------
# Fuentes y dibujo básico
# -----------------------------------------------------------------------------
def available_fonts(names):
    """Ruta de cada fuente de la lista que se encuentre en el sistema (o su nombre, si la trae OpenCV)."""
    found = []
    for name in names:
        if name in BUILT_IN_FONTS:
            found.append(name)
            continue
        path = next((p for folder in FONT_FOLDERS if folder.is_dir() for p in folder.rglob(name)), None)
        if path is None:
            print(f"Aviso: no se encuentra la fuente {name}; se omite.")
        else:
            found.append(str(path))
    if not found:
        raise RuntimeError("No se encuentra ninguna fuente de generator.numbers.fonts.")
    return found


@lru_cache(maxsize=None)
def font_face(font):
    """Carga una fuente con cv2.FontFace (con caché: cada proceso la carga una sola vez)."""
    return cv2.FontFace(font)


def text_mask(text, font, size_px, thicken=False):
    """Máscara (uint8, 0-255) de un texto, recortada justo a su tinta.

    Se escribe en blanco sobre un lienzo negro con margen de sobra y después se recorta con
    cv2.boundingRect, que da el rectángulo de los píxeles con tinta: así la caja es exacta. Con
    thicken=True, el trazo se engrosa con una dilatación.
    """
    size = max(4, round(size_px))  # tamaño de letra en píxeles
    canvas = np.zeros((3 * size, size * (len(text) + 2)), np.uint8)
    cv2.putText(canvas, text, (size, 2 * size), 255, font_face(font), size)
    if thicken:
        canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8))
    x, y, w, h = cv2.boundingRect(canvas)
    return canvas[y:y + h, x:x + w]


def paint(image, color, x, y, mask):
    """Pinta con 'color' la forma de la máscara, con su esquina superior izquierda en (x, y).

    Mezcla la tinta con lo que ya hay según la máscara: 255 es tinta pura y 0 deja el fondo.
    """
    height, width = mask.shape
    region = image[y:y + height, x:x + width].astype(np.float32)
    alpha = mask.astype(np.float32)[..., None] / 255
    mixed = region * (1 - alpha) + np.array(color, np.float32) * alpha
    image[y:y + height, x:x + width] = mixed.astype(np.uint8)


def draw_circle(image, cx, cy, radius, color, thickness=-1):
    """Círculo relleno (thickness=-1) o circunferencia, con borde suavizado y centro con decimales."""
    factor = 2 ** SHIFT
    center = (round((cx - 0.5) * factor), round((cy - 0.5) * factor))  # -0.5: convenio de OpenCV
    cv2.circle(image, center, round(radius * factor), color, thickness, cv2.LINE_AA, SHIFT)


# -----------------------------------------------------------------------------
# Colocación sin solapes ni ambigüedades
# -----------------------------------------------------------------------------
def box_gaps(box, boxes):
    """Distancia entre el rectángulo 'box' y cada uno de 'boxes' (0 si se tocan o se solapan)."""
    boxes = np.asarray(boxes, dtype=float)
    dx = np.maximum(np.maximum(boxes[:, 0] - box[2], box[0] - boxes[:, 2]), 0)
    dy = np.maximum(np.maximum(boxes[:, 1] - box[3], box[1] - boxes[:, 3]), 0)
    return np.hypot(dx, dy)


def inside(box, area):
    """True si la caja está entera dentro del área."""
    return box[0] >= area[0] and box[1] >= area[1] and box[2] <= area[2] and box[3] <= area[3]


class Layout:
    """Lo que ya está dibujado en la hoja, para comprobar que lo nuevo no se solapa ni crea ambigüedad.

    Guarda los puntos (centro y radio), la caja del número de cada punto (en el mismo orden), la
    distancia de cada punto a su número y las cajas de los distractores.
    """

    def __init__(self):
        self.dots = []        # (cx, cy, r)
        self.boxes = []       # [x0, y0, x1, y1] del número del punto i
        self.distances = []   # distancia del punto i a la caja de su número
        self.others = []      # cajas del título, las letras y los círculos huecos

    def dot_fits(self, cx, cy, r, sep):
        """El punto nuevo no toca nada ni queda casi tan cerca de un número ajeno como el punto de ese número."""
        if self.dots:
            dots = np.array(self.dots)
            if np.any(np.hypot(dots[:, 0] - cx, dots[:, 1] - cy) < dots[:, 2] + r + sep):
                return False
        if self.boxes:
            d = point_box_distance(cx, cy, self.boxes)
            if np.any(d < r + sep) or np.any(d < AMBIGUITY_MARGIN * np.array(self.distances)):
                return False
        return not self.others or bool(np.all(box_gaps([cx - r, cy - r, cx + r, cy + r], self.others) >= sep))

    def number_fits(self, box, cx, cy, distance, sep):
        """La caja del número del punto (cx, cy), a 'distance' de él, no toca nada y es inequívocamente suya.

        Comprueba que la caja no toca otros puntos ni números y que no crea ambigüedad:
          - ningún otro punto está casi tan cerca de ella como el suyo;
          - no queda casi tan cerca de otro punto como el número de ese punto;
          - su punto no tiene un número ajeno casi tan cerca como ella.
        """
        if self.dots:
            dots = np.array(self.dots)
            d = point_box_distance(dots[:, 0], dots[:, 1], box)
            if (np.any(d < dots[:, 2] + sep) or np.any(d < AMBIGUITY_MARGIN * distance)
                    or np.any(d < AMBIGUITY_MARGIN * np.array(self.distances))):
                return False
        if self.boxes:
            if np.any(box_gaps(box, self.boxes) < sep):
                return False
            if np.any(point_box_distance(cx, cy, self.boxes) < AMBIGUITY_MARGIN * distance):
                return False
        return not self.others or bool(np.all(box_gaps(box, self.others) >= sep))

    def space_is_free(self, box, sep):
        """Para los distractores: la caja queda a más de 'sep' de todo lo dibujado."""
        if self.dots:
            dots = np.array(self.dots)
            if np.any(point_box_distance(dots[:, 0], dots[:, 1], box) < dots[:, 2] + sep):
                return False
        if self.boxes and np.any(box_gaps(box, self.boxes) < sep):
            return False
        return not self.others or bool(np.all(box_gaps(box, self.others) >= sep))

    def add_pair(self, cx, cy, r, box, distance):
        self.dots.append((cx, cy, r))
        self.boxes.append(box)
        self.distances.append(distance)


def box_next_to_dot(cx, cy, r, gap, width, height, angle):
    """Caja de width x height en la dirección 'angle' desde el punto, a 'gap' píxeles de su borde.

    Primero se coloca el centro de la caja sobre el rayo que sale del punto con ese ángulo, de
    modo que el rayo corte el borde de la caja a r + gap del centro del punto. Una esquina puede
    quedar más cerca que ese corte, así que después se aleja la caja por el mismo rayo hasta que
    la distancia punto-rectángulo sea r + gap.
    """
    ux, uy = math.cos(angle), math.sin(angle)
    target = r + gap
    half_x = width / 2 / abs(ux) if abs(ux) > 1e-9 else math.inf
    half_y = height / 2 / abs(uy) if abs(uy) > 1e-9 else math.inf
    t = target + min(half_x, half_y)  # distancia del centro del punto al centro de la caja
    for _ in range(6):
        box = [cx + t * ux - width / 2, cy + t * uy - height / 2, cx + t * ux + width / 2, cy + t * uy + height / 2]
        d = float(point_box_distance(cx, cy, box))
        if d >= target - 0.01:
            break
        t += target - d
    return box


def find_number_place(rng, cfg, layout, area, cx, cy, r, mask, px_per_mm, touching, sep):
    """Prueba direcciones y huecos al azar hasta encontrar un sitio válido para el número.

    Devuelve (box, distance), o None si en 24 intentos no ha encontrado ninguno.
    """
    height, width = mask.shape
    for _ in range(24):
        gap = 0.0 if touching else random_float(rng, cfg["numbers"]["gap_mm"]) * px_per_mm
        box = box_next_to_dot(cx, cy, r, gap, width, height, rng.uniform(0, 2 * math.pi))
        x, y = round(box[0]), round(box[1])  # la máscara se pinta en píxeles enteros
        box = [x, y, x + width, y + height]
        distance = float(point_box_distance(cx, cy, box))
        if inside(box, area) and layout.number_fits(box, cx, cy, distance, sep):
            return box, distance
    return None


def shape_positions(rng, n, area):
    """n posiciones, en orden, a lo largo del contorno de una figura cerrada al azar.

    El contorno es una curva en polares, r(t) = 1 + suma de ondulaciones, estirada para ocupar
    buena parte del área útil. Al numerar los puntos en el orden del contorno, unirlos dibuja la
    figura, como en las fichas reales.
    """
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    radius = np.ones_like(t)
    for k in range(2, 7):  # la suma de amplitudes es menor que 1: el radio nunca llega a 0
        radius += rng.uniform(0, 0.5 / k) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    x, y = radius * np.cos(t), radius * np.sin(t)
    if rng.random() < 0.5:  # sentido de la numeración
        x, y = x[::-1], y[::-1]

    x0, y0, x1, y1 = area
    width, height = (x1 - x0) * rng.uniform(0.55, 0.95), (y1 - y0) * rng.uniform(0.55, 0.95)
    x = x0 + rng.uniform(0, x1 - x0 - width) + (x - x.min()) / (x.max() - x.min()) * width
    y = y0 + rng.uniform(0, y1 - y0 - height) + (y - y.min()) / (y.max() - y.min()) * height

    # n puntos a distancias casi iguales a lo largo del contorno, empezando en un sitio al azar.
    xc, yc = np.append(x, x[0]), np.append(y, y[0])
    travelled = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))])
    step = travelled[-1] / n
    s = (rng.uniform(0, travelled[-1]) + step * (np.arange(n) + rng.uniform(-0.15, 0.15, n))) % travelled[-1]
    return list(zip(np.interp(s, travelled, xc), np.interp(s, travelled, yc)))


# -----------------------------------------------------------------------------
# Paso 1: la hoja plana
# -----------------------------------------------------------------------------
def draw_title(rng, sheet, layout, margin, px_per_mm, fonts, ink):
    """Título y campos ("Nombre: ____") arriba de la hoja. Es texto sin dígitos: no se etiqueta."""
    title = text_mask(random_item(rng, TITLES), random_item(rng, fonts),
                      random_float(rng, [14, 22]) * MM_PER_PT * px_per_mm)
    x = margin if rng.random() < 0.5 else (sheet.shape[1] - title.shape[1]) // 2  # izquierda o centrado
    paint(sheet, ink, x, margin, title)
    layout.others.append([x, margin, x + title.shape[1], margin + title.shape[0]])

    if rng.random() < 0.7:  # campos con una línea para escribir
        font, size = random_item(rng, fonts), random_float(rng, [9, 12]) * MM_PER_PT * px_per_mm
        x, y = margin, margin + title.shape[0] + round(5 * px_per_mm)
        for field in FIELDS[:random_int(rng, [1, 3])]:
            text = text_mask(field, font, size)
            length = round(random_float(rng, [30, 55]) * px_per_mm)
            if x + text.shape[1] + length > sheet.shape[1] - margin:
                break
            paint(sheet, ink, x, y, text)
            base = y + text.shape[0]
            cv2.line(sheet, (x + text.shape[1] + round(2 * px_per_mm), base), (x + text.shape[1] + length, base),
                     ink, max(1, round(0.25 * px_per_mm)), cv2.LINE_AA)
            layout.others.append([x, y, x + text.shape[1] + length, base + 2])
            x += text.shape[1] + length + round(8 * px_per_mm)


def random_free_box(rng, layout, area, width, height, sep, attempts=30):
    """Busca al azar un hueco libre de width x height en el área. Devuelve su caja o None."""
    for _ in range(attempts):
        x = int(rng.integers(area[0], max(area[0] + 1, area[2] - width)))
        y = int(rng.integers(area[1], max(area[1] + 1, area[3] - height)))
        box = [x, y, x + width, y + height]
        if layout.space_is_free(box, sep):
            return box
    return None


def draw_distractors(rng, cfg, sheet, layout, area, px_per_mm, fonts, size_px, dot_radius, ink):
    """Letras, palabras y círculos huecos en huecos libres. No se etiquetan: son negativos difíciles."""
    # Lejos de puntos y números, para que no parezcan la etiqueta de un punto.
    sep = 2 * cfg["numbers"]["separation_mm"] * px_per_mm + size_px
    for _ in range(random_int(rng, cfg["distractors"]["loose_letters"])):
        text = random_item(rng, LETTERS) if rng.random() < 0.7 else random_item(rng, WORDS)
        mask = text_mask(text, random_item(rng, fonts), size_px * rng.uniform(0.8, 1.6))
        box = random_free_box(rng, layout, area, mask.shape[1], mask.shape[0], sep)
        if box is not None:
            paint(sheet, ink, box[0], box[1], mask)
            layout.others.append(box)
    for _ in range(random_int(rng, cfg["distractors"]["hollow_circles"])):
        radius = dot_radius * rng.uniform(1.3, 3.0)
        thickness = max(1, round(radius * rng.uniform(0.12, 0.3)))  # el hueco ocupa al menos el 60 % del radio
        side = math.ceil(2 * radius + thickness) + 2
        box = random_free_box(rng, layout, area, side, side, sep)
        if box is not None:
            draw_circle(sheet, box[0] + side / 2, box[1] + side / 2, radius, ink, thickness)
            layout.others.append(box)


def draw_sheet(rng, cfg, px_per_mm, fonts):
    """Dibuja la hoja plana. Devuelve (image, dots, numbers, metadata).

    dots: lista de (cx, cy, r) en píxeles de la hoja; numbers: lista de (value, box).
    El punto i va con el número i.
    """
    c_sheet, c_dots, c_numbers, c_hard = cfg["sheet"], cfg["dots"], cfg["numbers"], cfg["hard_cases"]
    width, height = round(c_sheet["size_mm"][0] * px_per_mm), round(c_sheet["size_mm"][1] * px_per_mm)
    white, tint = random_int(rng, c_sheet["paper_white"]), c_sheet["paper_tint"]
    paper = tuple(min(255, white + random_int(rng, [-tint, tint])) for _ in range(3))
    sheet = np.full((height, width, 3), paper, np.uint8)
    layout = Layout()

    # Tinta: casi siempre negra o gris oscura; a veces de color, y entonces los puntos pueden ir
    # de otro color que los números.
    number_ink = dot_ink = random_item(rng, DARK_INKS)
    if rng.random() < cfg["ink"]["color_probability"]:
        number_ink = dot_ink = random_item(rng, COLOR_INKS)
        if rng.random() < cfg["ink"]["different_dot_color_probability"]:
            dot_ink = random_item(rng, COLOR_INKS)

    # Área útil: dentro de los márgenes y, si hay título, por debajo de él.
    margin = round(random_float(rng, c_sheet["margin_mm"]) * px_per_mm)
    has_title = rng.random() < cfg["distractors"]["title_probability"]
    band = round(28 * px_per_mm) if has_title else 0
    area = (margin, margin + band, width - margin, height - margin)
    if has_title:
        draw_title(rng, sheet, layout, margin, px_per_mm, fonts, number_ink)

    # Aspecto de los puntos y números de esta hoja (una sola fuente, como en una ficha real).
    font = random_item(rng, fonts)
    size_pt = random_float(rng, c_numbers["size_pt"])
    size_px = size_pt * MM_PER_PT * px_per_mm
    thicken = rng.random() < c_numbers["thicken_probability"]
    base_radius = random_float(rng, c_dots["diameter_mm"]) * px_per_mm / 2
    sep = c_numbers["separation_mm"] * px_per_mm

    # Posiciones candidatas, en el orden en que se numeran.
    n = random_int(rng, c_dots["count"])
    use_shape = rng.random() < c_dots["shape_probability"]
    if use_shape:
        candidates = shape_positions(rng, n, area)
    else:
        candidates = ((random_float(rng, [area[0], area[2]]), random_float(rng, [area[1], area[3]]))
                      for _ in range(40 * n))

    dots, numbers = [], []
    for cx, cy in candidates:
        if len(dots) == n:
            break
        r = base_radius * (1 + rng.uniform(-1, 1) * c_dots["diameter_variation"])
        if not inside([cx - r, cy - r, cx + r, cy + r], area) or not layout.dot_fits(cx, cy, r, sep):
            continue

        value = len(dots) + 1  # los puntos que no caben se saltan: la numeración sigue seguida
        mask = text_mask(str(value), font, size_px, thicken)
        touching = rng.random() < c_hard["touching_probability"]
        crowded = rng.random() < c_hard["crowded_probability"]
        number_sep = c_hard["crowded_separation_mm"] * px_per_mm if crowded else sep
        place = find_number_place(rng, cfg, layout, area, cx, cy, r, mask, px_per_mm, touching, number_sep)
        if place is None:
            continue
        box, distance = place
        layout.add_pair(cx, cy, r, box, distance)
        draw_circle(sheet, cx, cy, r, dot_ink)
        paint(sheet, number_ink, box[0], box[1], mask)
        dots.append((cx, cy, r))
        numbers.append((value, box))

    draw_distractors(rng, cfg, sheet, layout, area, px_per_mm, fonts, size_px, base_radius, number_ink)
    metadata = {"layout": "shape" if use_shape else "random", "requested_dots": n,
                "font": Path(font).name, "size_pt": round(size_pt, 1), "thicken": thicken,
                "dot_diameter_mm": round(2 * base_radius / px_per_mm, 2), "title": has_title}
    return sheet, dots, numbers, metadata


# -----------------------------------------------------------------------------
# Paso 2: la hoja en la foto (homografía)
# -----------------------------------------------------------------------------
def table_background(rng, width, height):
    """Mesa de un color al azar con una textura suave (vetas) y algo de grano."""
    color = np.array(random_item(rng, TABLE_COLORS), np.float32) * rng.uniform(0.85, 1.15)
    grain = rng.standard_normal((height // 6 + 2, width // 24 + 2)).astype(np.float32)
    texture = cv2.resize(grain, (width, height), interpolation=cv2.INTER_CUBIC)
    table = color + texture[..., None] * rng.uniform(3, 12)
    return np.clip(table, 0, 255).astype(np.uint8)


def compose_photo(rng, cfg, sheet):
    """Coloca la hoja sobre la mesa como la vería una cámara cenital.

    Devuelve la foto (todavía sin efectos), la homografía hoja -> foto y la escala aplicada para
    que la hoja quepa (1 si cabía sin reducirla).
    """
    c = cfg["camera"]
    sheet_height, sheet_width = sheet.shape[:2]
    height = round(sheet_height / random_float(rng, c["sheet_fill"]))
    width = round(height * c["aspect_ratio"])

    # Esquinas de la hoja respecto a su centro, con un giro pequeño y un desplazamiento distinto
    # en cada esquina (la perspectiva de una cámara que no está del todo perpendicular).
    corners = np.array([[0, 0], [sheet_width, 0], [sheet_width, sheet_height], [0, sheet_height]], np.float32)
    angle = math.radians(rng.uniform(-c["max_rotation_deg"], c["max_rotation_deg"]))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    target = (corners - [sheet_width / 2, sheet_height / 2]) @ rotation.T
    target += rng.uniform(-1, 1, (4, 2)) * c["perspective"] * np.array([sheet_width, sheet_height])
    # Si no cabe en la foto (dejando 2 px por lado), se reduce.
    size = target.max(axis=0) - target.min(axis=0)
    scale = min(1.0, (width - 4) / size[0], (height - 4) / size[1])
    target *= scale
    # Desplazamiento al azar dentro del hueco que queda.
    room = np.array([width, height]) - (target.max(axis=0) - target.min(axis=0))
    target += -target.min(axis=0) + [rng.uniform(2, room[0] - 2), rng.uniform(2, room[1] - 2)]
    homography = cv2.getPerspectiveTransform(corners, target.astype(np.float32))

    # La hoja vista por la cámara y su silueta (255 donde hay hoja). OpenCV mide desde el centro
    # del píxel y las anotaciones desde su esquina; con giros de pocos grados y sin cambio de
    # tamaño, eso altera la homografía en centésimas de píxel, así que se usa la misma matriz
    # para la imagen y para las anotaciones.
    paper = tuple(int(v) for v in sheet[0, 0])
    sheet_in_photo = cv2.warpPerspective(sheet, homography, (width, height), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=paper)
    silhouette = cv2.warpPerspective(np.full((sheet_height, sheet_width), 255, np.uint8), homography,
                                     (width, height), flags=cv2.INTER_LINEAR)

    # Mesa con la sombra que proyecta la hoja: su silueta desplazada y difuminada.
    offset = np.float32([[1, 0, rng.integers(-6, 7)], [0, 1, rng.integers(2, 10)]])
    shadow = cv2.warpAffine(silhouette, offset, (width, height))
    shadow = cv2.GaussianBlur(shadow, (0, 0), rng.uniform(4, 12)).astype(np.float32) / 255
    table = table_background(rng, width, height).astype(np.float32)
    table *= (1 - rng.uniform(0.2, 0.45) * shadow)[..., None]

    # Mezcla: donde la silueta vale 255 se ve la hoja; donde vale 0, la mesa.
    alpha = silhouette.astype(np.float32)[..., None] / 255
    photo = table * (1 - alpha) + sheet_in_photo.astype(np.float32) * alpha
    return np.clip(photo, 0, 255).astype(np.uint8), homography, scale


# -----------------------------------------------------------------------------
# Paso 3: efectos de cámara
# -----------------------------------------------------------------------------
def lighting_field(rng, width, height, minimum):
    """Factor de luz entre 'minimum' y 1: un degradado en una dirección al azar mezclado con un
    viñeteado (esquinas más oscuras)."""
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    angle = rng.uniform(0, 2 * math.pi)
    gradient = (x * math.cos(angle) + y * math.sin(angle) + math.sqrt(2)) / (2 * math.sqrt(2))
    vignette = (x ** 2 + y ** 2) / 2
    weight = rng.uniform(0.3, 1.0)
    return 1 - (1 - minimum) * (weight * gradient + (1 - weight) * vignette)


def soft_shadow(rng, width, height):
    """Factor menor que 1 en una mancha grande de bordes muy difuminados (sombra del brazo de la
    cámara, de una mano...). Se dibuja a 1/8 de resolución porque es muy suave."""
    small = np.zeros((height // 8 + 1, width // 8 + 1), np.uint8)
    h, w = small.shape
    center = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
    axes = (int(rng.uniform(0.15, 0.45) * w), int(rng.uniform(0.15, 0.45) * h))
    cv2.ellipse(small, center, axes, 0, 0, 360, 255, -1)
    small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.05, 0.15) * min(w, h))
    stain = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return 1 - rng.uniform(0.1, 0.3) * stain


def camera_effects(rng, effects, photo):
    """Iluminación irregular, sombra opcional, balance de blancos, desenfoque y ruido del sensor."""
    height, width = photo.shape[:2]
    light = lighting_field(rng, width, height, random_float(rng, effects["min_lighting"]))
    if rng.random() < effects["shadow_probability"]:
        light = light * soft_shadow(rng, width, height)
    balance = rng.uniform(0.96, 1.04, size=3).astype(np.float32)  # tono de la luz por canal
    photo = np.clip(photo.astype(np.float32) * light[..., None] * balance, 0, 255).astype(np.uint8)

    sigma = random_float(rng, effects["blur_sigma"])
    if sigma > 0.05:
        photo = cv2.GaussianBlur(photo, (0, 0), sigma)
    pixels = photo.astype(np.float32)
    pixels += rng.standard_normal(pixels.shape, dtype=np.float32) * random_float(rng, effects["noise_sigma"])
    return np.clip(pixels, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Paso 4: anotación en coordenadas de la foto
# -----------------------------------------------------------------------------
def transform_points(homography, points):
    """Pasa una lista de puntos (x, y) de la hoja a la foto con la homografía."""
    points = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def annotate(homography, dots, numbers, width, height):
    """Pasa puntos y números de la hoja a la foto y construye las listas del formato común.

    Caja de cada punto: la de 16 puntos de su borde ya transformados. Caja de cada número: la de
    sus 4 esquinas transformadas (con giros de pocos grados, apenas más holgada que la tinta).
    """
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    def box_in_photo(outline):
        p = transform_points(homography, outline)
        x0, x1 = np.clip([p[:, 0].min(), p[:, 0].max()], 0, width)
        y0, y1 = np.clip([p[:, 1].min(), p[:, 1].max()], 0, height)
        return [round(float(v), 2) for v in (x0, y0, x1, y1)]

    dot_list, number_list, pairs = [], [], []
    for i, ((cx, cy, r), (value, box)) in enumerate(zip(dots, numbers)):
        x, y = (round(float(v), 2) for v in transform_points(homography, [(cx, cy)])[0])
        dot_box = box_in_photo(np.c_[cx + r * np.cos(angles), cy + r * np.sin(angles)])
        number_box = box_in_photo([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
        dot_list.append({"id": i, "x": x, "y": y, "box": dot_box})
        number_list.append({"id": i, "value": value, "box": number_box})
        # Distancia calculada con los mismos valores redondeados que se guardan en el JSON.
        distance = float(point_box_distance(x, y, number_box))
        pairs.append({"dot": i, "number": i, "distance": round(distance, 2)})
    return dot_list, number_list, pairs


def generate_photo(cfg, split, index, fonts):
    """Genera una foto y su anotación. Devuelve (photo, annotation, jpeg_quality).

    Cada foto tiene su propio generador aleatorio, creado a partir de (semilla, partición,
    índice): cualquier foto se puede regenerar sola y el resultado no depende del orden ni del
    número de procesos.
    """
    rng = np.random.default_rng([cfg["seed"], SPLITS.index(split), index])
    px_per_mm = random_float(rng, cfg["camera"]["px_per_mm"])
    sheet, dots, numbers, metadata = draw_sheet(rng, cfg, px_per_mm, fonts)
    photo, homography, scale = compose_photo(rng, cfg, sheet)
    photo = camera_effects(rng, cfg["effects"], photo)
    jpeg_quality = random_int(rng, cfg["effects"]["jpeg_quality"])
    height, width = photo.shape[:2]
    dot_list, number_list, pairs = annotate(homography, dots, numbers, width, height)
    metadata.update(px_per_mm=round(px_per_mm * scale, 3), jpeg_quality=jpeg_quality)
    annotation = json_format.create(f"photo_{index:05d}.jpg", width, height, dot_list, number_list,
                                    pairs, metadata=metadata)
    return photo, annotation, jpeg_quality


def _generate_and_save(task):
    """Trabajo de cada proceso: genera una foto y guarda la imagen y su anotación."""
    cfg, split, index, fonts, folder = task
    photo, annotation, jpeg_quality = generate_photo(cfg, split, index, fonts)
    name = annotation["image"]
    write_image(folder / split / "images" / name, photo, jpeg_quality)
    json_format.save(annotation, folder / split / "annotations" / name.replace(".jpg", ".json"))
    return len(annotation["dots"])


def main():
    parser = argparse.ArgumentParser(description="Genera el dataset sintético (fotos y anotaciones).")
    parser.add_argument("--quick", action="store_true", help="dataset mini para la prueba rápida")
    args = parser.parse_args()

    config = load_config()
    cfg = config["generator"]
    splits, folder = cfg["splits"], project_path(cfg["output"])
    if args.quick:
        splits, folder = config["quick"]["splits"], project_path(config["quick"]["data_output"])

    fonts = available_fonts(cfg["numbers"]["fonts"])
    clear_generated_folder(folder, "config_used.yaml")
    for split in SPLITS:
        (folder / split / "images").mkdir(parents=True)
        (folder / split / "annotations").mkdir(parents=True)
    used = {"generator": cfg, "splits": splits, "fonts_used": fonts}
    (folder / "config_used.yaml").write_text(yaml.safe_dump(used, allow_unicode=True, sort_keys=False),
                                             encoding="utf-8")

    tasks = [(cfg, split, i, fonts, folder) for split in SPLITS for i in range(splits[split])]
    print(f"Generando {len(tasks)} fotos en {folder} con {cfg['processes']} procesos...")
    start, total_dots = time.time(), 0
    with ProcessPoolExecutor(max_workers=cfg["processes"]) as executor:
        for done, n in enumerate(executor.map(_generate_and_save, tasks, chunksize=2), start=1):
            total_dots += n
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} fotos ({time.time() - start:.0f} s)")

    # Algunas fotos de train con las cajas dibujadas, para comprobar a ojo que cuadran.
    (folder / "samples").mkdir()
    for i in range(min(cfg["samples_with_boxes"], splits["train"])):
        name = f"photo_{i:05d}"
        annotation = json_format.load(folder / "train" / "annotations" / f"{name}.json")
        photo = read_image(folder / "train" / "images" / f"{name}.jpg")
        write_image(folder / "samples" / f"{name}.jpg", draw_annotation(photo, annotation, join_in_order=True), 90)

    print(f"Hecho: {len(tasks)} fotos y {total_dots} puntos "
          f"({total_dots / len(tasks):.1f} por foto) en {time.time() - start:.0f} s.")


if __name__ == "__main__":
    main()
