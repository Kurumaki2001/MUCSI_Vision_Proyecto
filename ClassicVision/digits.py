"""Lectura de números sin aprendizaje: segmentación de caracteres y comparación con plantillas.

1. Caracteres. Cada mancha de tinta que queda tras borrar los puntos es, casi siempre, un
   carácter. Se descartan las que no tienen tamaño de carácter (comparando con la altura mediana
   de todas) y las líneas largas (la raya de "Nombre: ____"). Los trozos de un mismo carácter
   (un "5" que se ha partido al umbralizar) se juntan si están uno encima del otro. Si una mancha
   es demasiado ancha para ser un dígito, son dos o tres dígitos pegados: se cortan por las
   columnas con menos tinta (proyección vertical).

2. Números. Los caracteres que están a la misma altura y casi tocándose forman un número (unión
   de vecinos, como en una componente conexa). Un grupo de más de max_digits caracteres es una
   palabra y se descarta.

3. Reconocimiento por plantillas (template matching). Las plantillas son los dígitos 0-9
   escritos con varias fuentes (cv2.putText con las fuentes TrueType de classic.template_fonts;
   con OpenCV 4, las fuentes Hershey que trae OpenCV), normales y engrosados. Cada carácter y
   cada plantilla se normalizan igual: se escalan a 28 px conservando la proporción, se centran
   en un cuadro de 32 x 32 y se suavizan. La similitud es la correlación normalizada (lo mismo
   que cv2.matchTemplate con TM_CCOEFF_NORMED cuando las dos imágenes miden lo mismo): 1 =
   idénticas. El dígito leído es el de la plantilla más parecida.

4. Decisión. Un grupo cuya similitud media es baja no es un número (una letra suelta, una
   circunferencia...): se descarta. Un número que empieza por 0 tampoco existe en las fichas (una
   "O" o una circunferencia pequeña se leería "0"). Si algún dígito de un número tiene una
   similitud baja, el número se queda con value None: sigue en el emparejamiento, para que su
   punto no se quede con un número ajeno.
"""

from functools import lru_cache

import cv2
import numpy as np

from ClassicVision.dots import blobs

NORM_SIZE, NORM_INNER = 32, 28  # cuadro normalizado y tamaño máximo del carácter dentro
HERSHEY_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
                 cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN]


# -----------------------------------------------------------------------------
# Normalización y plantillas
# -----------------------------------------------------------------------------
def normalize(glyph):
    """Vector (1024,) de media 0 y norma 1 de un carácter (imagen recortada a su tinta, 0 = papel)."""
    h, w = glyph.shape[:2]
    if w == 0 or h == 0:
        return None
    glyph = glyph.astype(np.float32)
    scale = NORM_INNER / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    glyph = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((NORM_SIZE, NORM_SIZE), np.float32)
    x0, y0 = (NORM_SIZE - new_w) // 2, (NORM_SIZE - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = glyph
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.0).ravel()
    canvas -= canvas.mean()
    norm = np.linalg.norm(canvas)
    return canvas / norm if norm > 0 else None


def render_digit(digit, font, size=64):
    """Máscara de un dígito escrito con una fuente TrueType (ruta) o Hershey (entero de OpenCV)."""
    canvas = np.zeros((3 * size, 3 * size), np.uint8)
    if isinstance(font, int):
        cv2.putText(canvas, digit, (size, 2 * size), font, size / 30, 255, max(1, size // 20), cv2.LINE_AA)
    else:
        cv2.putText(canvas, digit, (size, 2 * size), 255, cv2.FontFace(font), size)
    return canvas


@lru_cache(maxsize=None)
def build_templates(fonts):
    """Plantillas: (matriz (n, 1024) de vectores normalizados, dígito de cada fila)."""
    vectors, labels = [], []
    for font in fonts:
        for digit in "0123456789":
            mask = render_digit(digit, font)
            for variant in (mask, cv2.dilate(mask, np.ones((5, 5), np.uint8))):  # normal y engrosada
                x, y, w, h = cv2.boundingRect(variant)
                vector = normalize(variant[y:y + h, x:x + w])
                if vector is not None:
                    vectors.append(vector)
                    labels.append(digit)
    return np.array(vectors, np.float32), np.array(labels)


def template_fonts(names):
    """Fuentes para las plantillas: las TrueType de la lista si OpenCV las admite (OpenCV 5).

    El nombre "hershey" añade las fuentes Hershey que trae OpenCV. Sin cv2.FontFace (OpenCV 4)
    solo se pueden usar esas.
    """
    hershey = tuple(HERSHEY_FONTS) if "hershey" in names or not hasattr(cv2, "FontFace") else ()
    truetype = [name for name in names if name != "hershey"]
    if not truetype or not hasattr(cv2, "FontFace"):
        return hershey
    from data.generate import available_fonts  # busca las fuentes en las carpetas del sistema
    return hershey + tuple(available_fonts(truetype))


def glyph_image(char, dark):
    """Imagen en grises del carácter: la oscuridad de la foto dentro de su máscara (algo dilatada).

    Se usa la oscuridad (0-1) y no la máscara binaria porque, en dígitos pequeños en negrita, el
    umbral cierra los huecos interiores (un "8" queda como una mancha); en grises, el hueco
    sigue siendo más claro que el trazo. La máscara dilatada deja fuera la tinta de los vecinos.
    """
    region = cv2.dilate(char.mask, np.ones((3, 3), np.uint8)) > 0
    crop = dark[char.y0:char.y1, char.x0:char.x1] * region
    return crop * (255.0 / max(float(crop.max()), 1e-6))


def recognize(glyph, templates):
    """(dígito, similitud) del carácter: la plantilla con mayor correlación normalizada."""
    vector = normalize(glyph)
    if vector is None:
        return None, 0.0
    vectors, labels = templates
    scores = vectors @ vector
    best = int(np.argmax(scores))
    return str(labels[best]), float(scores[best])


# -----------------------------------------------------------------------------
# Caracteres y números
# -----------------------------------------------------------------------------
class Char:
    """Un carácter: su caja [x0, y0, x1, y1) en la foto y su máscara recortada a esa caja."""

    def __init__(self, x0, y0, x1, y1, mask):
        self.x0, self.y0, self.x1, self.y1, self.mask = x0, y0, x1, y1, mask

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    def merge(self, other):
        """Junta dos trozos de un mismo carácter en uno."""
        x0, y0 = min(self.x0, other.x0), min(self.y0, other.y0)
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for c in (self, other):
            region = mask[c.y0 - y0:c.y1 - y0, c.x0 - x0:c.x1 - x0]
            np.maximum(region, c.mask, out=region)
        return Char(x0, y0, x1, y1, mask)


def find_chars(text_ink, cfg):
    """Manchas con tamaño de carácter, con los trozos juntados y los dígitos pegados separados."""
    chars = [Char(x, y, x + w, y + h, mask) for x, y, w, h, mask in blobs(text_ink)
             if h >= cfg["char_min_height_px"] and cv2.countNonZero(mask) >= cfg["min_blob_area"]]
    chars = [c for c in chars if c.width <= cfg["char_max_aspect"] * c.height]  # fuera rayas largas
    if not chars:
        return [], None

    # Altura típica de un carácter: mediana de las manchas con proporción de carácter
    def typical_height():
        typical = [c.height for c in chars if 0.15 * c.height <= c.width <= 1.2 * c.height]
        return float(np.median(typical)) if typical else float(np.median([c.height for c in chars]))

    # Trozos de un mismo carácter: se solapan mucho en horizontal, están uno encima del otro y,
    # juntos, miden lo que un carácter (si no, son dos números, uno encima del otro)
    height = typical_height()
    chars.sort(key=lambda c: c.x0)
    merged = True
    while merged:
        merged = False
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                a, b = chars[i], chars[j]
                overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
                vertical_gap = max(a.y0, b.y0) - min(a.y1, b.y1)
                joined_height = max(a.y1, b.y1) - min(a.y0, b.y0)
                if (overlap >= cfg["piece_overlap"] * min(a.width, b.width)
                        and vertical_gap <= 0.3 * max(a.height, b.height)
                        and joined_height <= cfg["char_height_band"][1] * height):
                    chars[i] = a.merge(b)
                    del chars[j]
                    merged = True
                    break
            if merged:
                break

    height = typical_height()  # otra vez, ya con los caracteres completos
    low, high = cfg["char_height_band"]
    chars = [c for c in chars if low * height <= c.height <= high * height]

    # Dígitos pegados: más anchos que un dígito -> se cortan por las columnas con menos tinta. El
    # número de trozos sale del ancho típico de un dígito en esta foto (ancho / alto de los
    # caracteres sueltos, sin contar los "1", que son estrechos): Verdana tiene dígitos anchos y
    # Arial Narrow estrechos.
    ratios = [c.width / c.height for c in chars if 0.4 <= c.width / c.height <= cfg["split_aspect"]]
    digit_aspect = float(np.median(ratios)) if len(ratios) >= 5 else cfg["digit_aspect"]
    result = []
    for c in chars:
        if c.width <= cfg["split_aspect"] * c.height:
            result.append(c)
            continue
        pieces = int(np.clip(round(c.width / (digit_aspect * c.height)), 2, cfg["max_digits"]))
        profile = (c.mask > 0).sum(axis=0)
        cuts, window = [0], max(1, round(0.15 * c.width))
        for k in range(1, pieces):
            ideal = round(k * c.width / pieces)
            lo, hi = max(1, ideal - window), min(c.width - 1, ideal + window)
            cuts.append(lo + int(np.argmin(profile[lo:hi])) if hi > lo else ideal)
        cuts.append(c.width)
        for a, b in zip(cuts, cuts[1:]):
            part = c.mask[:, a:b]
            if cv2.countNonZero(part):
                y, h = cv2.boundingRect(part)[1], cv2.boundingRect(part)[3]
                result.append(Char(c.x0 + a, c.y0 + y, c.x0 + b, c.y0 + y + h, part[y:y + h]))
    return result, height


def aligned(a, b, cfg):
    """True si dos caracteres están en la misma línea y tienen alturas parecidas."""
    vertical_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    ratio = max(a.height, b.height) / min(a.height, b.height)
    return vertical_overlap >= 0.5 * min(a.height, b.height) and ratio <= cfg["max_height_ratio"]


def center_x(c):
    return (c.x0 + c.x1) / 2


def digit_pitch(chars, height, cfg):
    """Paso entre dígitos (distancia entre centros de dos dígitos seguidos) en esta foto, en píxeles.

    Se usa la distancia entre centros y no el hueco entre cajas porque en las fuentes
    monoespaciadas (Courier, Consolas) un "1" deja mucho hueco a los lados: en "14" el hueco es
    tan grande como el que separa dos números distintos, pero los centros siguen a un paso fijo.
    Para cada carácter se mide la distancia al centro de su vecino alineado más cercano. En una
    ficha casi todos los números tienen dos o más dígitos, así que la mediana de esas distancias
    es el paso de la fuente.
    """
    distances = []
    for a in chars:
        near = [abs(center_x(b) - center_x(a)) for b in chars
                if b is not a and aligned(a, b, cfg) and abs(center_x(b) - center_x(a)) <= 1.6 * height]
        if near:
            distances.append(min(near))
    if len(distances) < 5:
        return (cfg["digit_aspect"] + 0.1) * height
    return float(np.median(distances))


def group_numbers(chars, height, cfg):
    """Agrupa los caracteres en números: misma línea y a un paso de dígito. Devuelve listas ordenadas por x."""
    max_distance = cfg["pitch_factor"] * digit_pitch(chars, height, cfg)
    parent = list(range(len(chars)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(chars):
        for j in range(i + 1, len(chars)):
            b = chars[j]
            gap = max(a.x0, b.x0) - min(a.x1, b.x1)
            if (aligned(a, b, cfg) and abs(center_x(a) - center_x(b)) <= max_distance
                    and gap <= cfg["max_digit_gap"] * height):
                parent[root(i)] = root(j)
    groups = {}
    for i, c in enumerate(chars):
        groups.setdefault(root(i), []).append(c)
    return [sorted(g, key=lambda c: c.x0) for g in groups.values()]


def read_numbers(text_ink, dark, templates, cfg):
    """Números de la foto: [{"box", "value" (int o None), "confidence"}].

    text_ink: tinta sin los puntos (para separar caracteres); dark: oscuridad (para reconocerlos).
    """
    chars, height = find_chars(text_ink, cfg)
    numbers = []
    if not chars:
        return numbers
    for group in group_numbers(chars, height, cfg):
        if len(group) > cfg["max_digits"]:
            continue  # palabra
        readings = [recognize(glyph_image(c, dark), templates) for c in group]
        digits = "".join(d or "?" for d, _ in readings)
        scores = [s for _, s in readings]
        if np.mean(scores) < cfg["number_min_score"] or digits.startswith("0"):
            continue  # no parece un número
        value = int(digits) if min(scores) >= cfg["ocr_min_confidence"] and digits.isdigit() else None
        box = [min(c.x0 for c in group), min(c.y0 for c in group), max(c.x1 for c in group), max(c.y1 for c in group)]
        numbers.append({"box": [float(v) for v in box], "value": value, "confidence": round(min(scores), 4)})
    return numbers
