"""Etiquetado de fotos reales con SAM 2.1: unos clics por punto y por número -> JSON en el formato común.

Adaptado de etiquetar_sam.py (prácticas 2.2 y 2.3 de clase). Se mantiene la idea: el usuario
indica con clics o con una caja dónde está el objeto, SAM saca su contorno y de ahí sale la caja.
Lo que cambia para este proyecto:
  - Se etiqueta por parejas: el punto, su número y el valor del número tecleado. Se guarda en el
    formato común (common/json_format.py), con cajas, así que la foto sirve para evaluar todo
    (puntos, lectura de números y parejas) y para entrenar YOLO con prepare_yolo.py.
  - SAM recibe un recorte de annotation.crop_px alrededor del clic, no la foto entera: amplía su
    entrada a 1024 px, así que un punto de pocos píxeles se ve varias veces más grande y la caja
    sale mejor (medido en config.yaml).
  - Zoom con la rueda del ratón: a 1920 px en una ventana normal, un punto mide 2-3 px en pantalla.
  - Si SAM devuelve una caja absurda (mayor que annotation.max_box_px), se descarta.
  - Con --prelabel, el pipeline actual (YOLO + EasyOCR) propone las parejas y solo hay que
    corregir los errores. El JSON lo indica en metadata.prelabel. OJO: las fotos de TEST conviene
    etiquetarlas desde cero; si no, la evaluación sale sesgada a favor del modelo que las propuso.

Uso, desde la carpeta del proyecto:
    python -m tools.annotate_sam                       # fotos de real_data.images, desde la primera sin anotar
    python -m tools.annotate_sam --prelabel            # el modelo propone, el usuario corrige
    python -m tools.annotate_sam data/real/images --start 25

Ratón:
    clic izquierdo         "esto si" del objeto que se está marcando (punto o número)
    clic derecho           "esto no" (para quitar una sombra, el número de al lado...)
    arrastrar (izquierdo)  caja alrededor del objeto. Útil con números de varias cifras. Si SAM
                           falla, se usa la propia caja dibujada
    rueda                  zoom en el sitio del ratón
    arrastrar (central)    mover la foto
Teclado:
    ESPACIO     fijar la caja propuesta: tras el punto se pasa al número
    0-9         valor del número;  Retroceso borra la última cifra (o los clics, si no hay cifras)
    Intro       guardar la pareja (punto + número + valor) y empezar otra
    Tab         cambiar entre punto y número en la pareja actual
    e / x       editar / borrar la pareja que está bajo el ratón
    z           deshacer la última pareja guardada
    c           cancelar la pareja actual (si se estaba editando, vuelve a como estaba)
    f           ver la foto entera
    a / d       foto anterior / siguiente (guarda antes)
    q / Esc     guardar y salir

Los textos de la ventana van sin tildes porque las fuentes integradas de OpenCV solo tienen ASCII.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import json_format
from common.config import load_config, project_path
from common.images import read_image
from common.matching import point_box_distance

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
KEY_TAB, KEY_ENTER, KEY_ESC, KEY_BACKSPACE, KEY_SPACE = 9, 13, 27, 8, 32
WINDOW = "Etiquetar con SAM"
BACKGROUND = (40, 40, 40)
COLORS = {"dot": (0, 170, 0), "number": (230, 120, 0)}   # BGR: puntos en verde, números en azul
CURRENT, PROPOSAL = (0, 220, 255), (255, 0, 255)          # pareja actual en amarillo, propuesta de SAM en magenta


# -----------------------------------------------------------------------------
# Geometría (sin ventanas ni SAM, para poder probarla)
# -----------------------------------------------------------------------------
def crop_window(points, labels, hint_box, crop_px, width, height):
    """Recorte [x0, y0, x1, y1] (enteros) que se pasa a SAM, centrado en los clics "sí" o en la caja.

    Mide crop_px de lado o, si la caja dibujada es mayor, 1,5 veces la caja. Se desplaza para no
    salirse de la foto y, si la foto es más pequeña que el recorte, es la foto entera.
    """
    positives = [p for p, label in zip(points, labels) if label == 1]
    if hint_box is not None:
        cx, cy = (hint_box[0] + hint_box[2]) / 2, (hint_box[1] + hint_box[3]) / 2
        side = max(crop_px, 1.5 * max(hint_box[2] - hint_box[0], hint_box[3] - hint_box[1]))
    else:
        cx, cy = np.mean(positives, axis=0)
        side = crop_px
    side_x, side_y = min(round(side), width), min(round(side), height)
    x0 = int(min(max(round(cx - side_x / 2), 0), width - side_x))
    y0 = int(min(max(round(cy - side_y / 2), 0), height - side_y))
    return x0, y0, x0 + side_x, y0 + side_y


def polygon_box(polygon, x0=0, y0=0):
    """Caja [x0, y0, x1, y1] de un contorno de SAM dentro de un recorte que empieza en (x0, y0).

    El contorno da índices de píxel (el píxel i ocupa [i, i + 1)), así que el borde derecho e
    inferior de la caja es el último píxel + 1.
    """
    polygon = np.asarray(polygon, dtype=float)
    return [round(float(polygon[:, 0].min()) + x0, 2), round(float(polygon[:, 1].min()) + y0, 2),
            round(float(polygon[:, 0].max()) + 1 + x0, 2), round(float(polygon[:, 1].max()) + 1 + y0, 2)]


def plausible_box(box, max_side, points=(), labels=()):
    """¿Es creíble la caja de SAM? No puede ser mayor que max_side ni dejar fuera un clic "sí"."""
    if box is None or max(box[2] - box[0], box[3] - box[1]) > max_side:
        return False
    return all(box[0] - 1 <= x <= box[2] + 1 and box[1] - 1 <= y <= box[3] + 1
               for (x, y), label in zip(points, labels) if label == 1)


def box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class View:
    """Zoom y desplazamiento de la foto en la ventana.

    Coordenadas de la foto y de la ventana con el mismo convenio que las anotaciones: el píxel
    (0, 0) ocupa el cuadrado [0, 1) x [0, 1). Con zoom 1 la foto entera cabe en la ventana.
    """

    MAX_ZOOM = 40.0

    def __init__(self, width, height, window=(1280, 720)):
        self.width, self.height = width, height
        self.window = window
        self.reset()

    def reset(self):
        self.zoom = 1.0
        self.cx, self.cy = self.width / 2, self.height / 2   # punto de la foto que queda en el centro

    def scale(self):
        """Píxeles de ventana por píxel de foto."""
        return min(self.window[0] / self.width, self.window[1] / self.height) * self.zoom

    def to_image(self, wx, wy):
        s = self.scale()
        return self.cx + (wx - self.window[0] / 2) / s, self.cy + (wy - self.window[1] / 2) / s

    def to_window(self, x, y):
        s = self.scale()
        return (x - self.cx) * s + self.window[0] / 2, (y - self.cy) * s + self.window[1] / 2

    def zoom_at(self, wx, wy, factor):
        """Cambia el zoom dejando quieto el punto de la foto que está bajo el ratón."""
        x, y = self.to_image(wx, wy)
        self.zoom = min(max(self.zoom * factor, 1.0), self.MAX_ZOOM)
        s = self.scale()
        self.cx, self.cy = x - (wx - self.window[0] / 2) / s, y - (wy - self.window[1] / 2) / s
        self.clamp()

    def pan(self, dwx, dwy):
        """Mueve la foto lo que se ha movido el ratón (en píxeles de ventana)."""
        s = self.scale()
        self.cx, self.cy = self.cx - dwx / s, self.cy - dwy / s
        self.clamp()

    def clamp(self):
        """El centro de la vista no puede salir de la foto."""
        self.cx = min(max(self.cx, 0.0), float(self.width))
        self.cy = min(max(self.cy, 0.0), float(self.height))

    def matrix(self):
        """Matriz de cv2.warpAffine que dibuja la foto en la ventana.

        cv2.warpAffine pone el centro del píxel (0, 0) en (0, 0), es decir, usa x - 0.5 respecto a
        nuestro convenio. Al pasar to_window a ese convenio, la traslación queda así.
        """
        s = self.scale()
        tx = (0.5 - self.cx) * s + self.window[0] / 2 - 0.5
        ty = (0.5 - self.cy) * s + self.window[1] / 2 - 0.5
        return np.float32([[s, 0, tx], [0, s, ty]])


# -----------------------------------------------------------------------------
# Estado del etiquetado de una foto (sin ventanas ni SAM, para poder probarlo)
# -----------------------------------------------------------------------------
class LabelState:
    """Parejas guardadas de una foto y la pareja que se está marcando.

    Cada pareja es {"dot": caja o None, "number": caja o None, "text": cifras del valor}. Una pareja
    solo con punto o solo con número se guarda como punto o número sin pareja.
    """

    def __init__(self, entries=None, prelabeled=False):
        self.entries = list(entries or [])
        self.prelabeled = prelabeled
        self.new_entry()

    # ---------- pareja actual ----------
    def new_entry(self):
        self.current = {"dot": None, "number": None, "text": ""}
        self.active = "dot"            # objeto al que van los clics: "dot" o "number"
        self.backup = None             # (posición, pareja original) mientras se edita una pareja
        self.clear_prompts()

    def clear_prompts(self):
        """Borra los clics, la caja dibujada y la propuesta de SAM del objeto activo."""
        self.points, self.labels, self.hint_box, self.proposal = [], [], None, None

    def has_current(self):
        return bool(self.current["dot"] or self.current["number"] or self.current["text"]
                    or self.proposal or self.points)

    def add_point(self, x, y, positive):
        self.points.append((float(x), float(y)))
        self.labels.append(1 if positive else 0)

    def set_hint_box(self, x0, y0, x1, y1):
        self.hint_box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    def can_segment(self):
        """SAM necesita al menos un clic "sí" o una caja."""
        return self.hint_box is not None or 1 in self.labels

    def confirm(self):
        """ESPACIO: la propuesta pasa a ser la caja del objeto activo. Tras el punto, toca el número."""
        if self.proposal is not None:
            self.current[self.active] = self.proposal
        self.clear_prompts()
        if self.active == "dot" and self.current["dot"] is not None:
            self.active = "number"

    def switch(self):
        """Tab: cambia entre punto y número (por ejemplo, para rehacer el punto de una pareja)."""
        self.active = "number" if self.active == "dot" else "dot"
        self.clear_prompts()

    def type_digit(self, digit):
        self.current["text"] += digit

    def backspace(self):
        """Borra la última cifra; si no hay cifras, los clics del objeto activo."""
        if self.current["text"]:
            self.current["text"] = self.current["text"][:-1]
        else:
            self.clear_prompts()

    def accept(self):
        """Intro: guarda la pareja actual (con la propuesta pendiente, si la hay). Devuelve si ha guardado."""
        if self.proposal is not None:
            self.current[self.active] = self.proposal
        saved = self.current["dot"] is not None or self.current["number"] is not None
        if saved:
            position = self.backup[0] if self.backup else len(self.entries)
            self.entries.insert(position, self.current)
        elif self.backup:                       # se ha vaciado una pareja en edición: se recupera
            self.entries.insert(*self.backup)
        self.new_entry()
        return saved

    def cancel(self):
        """Descarta la pareja actual; si se estaba editando una, vuelve a como estaba."""
        if self.backup:
            self.entries.insert(*self.backup)
        self.new_entry()

    # ---------- parejas guardadas ----------
    def entry_at(self, x, y, margin=2.0):
        """Posición de la pareja con una caja (punto o número) que contiene (x, y); la más pequeña si hay varias."""
        best, best_area = None, None
        for i, entry in enumerate(self.entries):
            for box in (entry["dot"], entry["number"]):
                if box and box[0] - margin <= x <= box[2] + margin and box[1] - margin <= y <= box[3] + margin:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if best_area is None or area < best_area:
                        best, best_area = i, area
        return best

    def edit_at(self, x, y):
        """Saca la pareja bajo (x, y) a la pareja actual para corregirla. Devuelve si había alguna."""
        i = self.entry_at(x, y)
        if i is None:
            return False
        self.cancel()                           # la pareja que se estuviera marcando se descarta
        i = self.entry_at(x, y)                 # cancel puede haber devuelto una pareja a la lista
        entry = self.entries.pop(i)
        self.current = dict(entry)
        self.backup = (i, entry)
        self.active = "number" if entry["dot"] is not None else "dot"
        self.clear_prompts()
        return True

    def delete_at(self, x, y):
        i = self.entry_at(x, y)
        if i is not None:
            self.entries.pop(i)
        return i is not None

    def undo(self):
        if self.entries:
            self.entries.pop()

    def missing_values(self):
        """Números sin valor tecleado."""
        return sum(1 for entry in self.entries if entry["number"] is not None and not entry["text"])

    # ---------- formato común ----------
    def to_annotation(self, image_name, width, height):
        """Anotación en el formato común, con las cajas de puntos y números."""
        dots, numbers, pairs, unpaired_dots, unpaired_numbers = [], [], [], [], []
        for entry in self.entries:
            if entry["dot"] is not None:
                x, y = box_center(entry["dot"])
                dots.append({"id": len(dots), "x": round(x, 2), "y": round(y, 2), "box": entry["dot"]})
            if entry["number"] is not None:
                value = int(entry["text"]) if entry["text"] else None
                numbers.append({"id": len(numbers), "value": value, "box": entry["number"]})
            if entry["dot"] is not None and entry["number"] is not None:
                distance = float(point_box_distance(x, y, entry["number"]))
                pairs.append({"dot": dots[-1]["id"], "number": numbers[-1]["id"], "distance": round(distance, 2)})
            elif entry["dot"] is not None:
                unpaired_dots.append(dots[-1]["id"])
            else:
                unpaired_numbers.append(numbers[-1]["id"])
        metadata = {"tool": "tools/annotate_sam.py", "prelabel": self.prelabeled}
        return json_format.create(image_name, width, height, dots, numbers, pairs, unpaired_dots,
                                  unpaired_numbers, metadata=metadata)

    @classmethod
    def from_annotation(cls, data, prelabeled=None):
        """Parejas de una anotación guardada o de un resultado del pipeline (mismo formato)."""
        if any(not dot.get("box") for dot in data["dots"]):
            raise ValueError(f"{data['image']}: hay puntos sin caja (anotación de tools/annotate.py). "
                             "Esta herramienta necesita las cajas.")
        dots = {dot["id"]: dot for dot in data["dots"]}
        numbers = {number["id"]: number for number in data["numbers"]}

        def text(number):
            return "" if number.get("value") is None else str(number["value"])

        entries = []
        for pair in data["pairs"]:
            number = numbers.pop(pair["number"])
            entries.append({"dot": list(dots.pop(pair["dot"])["box"]), "number": list(number["box"]),
                            "text": text(number)})
        entries += [{"dot": list(dot["box"]), "number": None, "text": ""} for dot in dots.values()]
        entries += [{"dot": None, "number": list(number["box"]), "text": text(number)}
                    for number in numbers.values()]
        if prelabeled is None:
            prelabeled = bool(data.get("metadata", {}).get("prelabel", False))
        return cls(entries, prelabeled)


# -----------------------------------------------------------------------------
# SAM y propuestas del modelo
# -----------------------------------------------------------------------------
class SamSegmenter:
    """SAM 2.1 de Ultralytics, que recibe solo un recorte alrededor de los clics."""

    def __init__(self, model_path, crop_px):
        from ultralytics import SAM   # se importa aquí porque tarda en cargar
        self.sam = SAM(str(model_path))
        self.crop_px = crop_px

    def segment(self, image, points, labels, hint_box):
        """Caja del objeto indicado (coordenadas de la foto), o None si SAM no encuentra nada."""
        height, width = image.shape[:2]
        x0, y0, x1, y1 = crop_window(points, labels, hint_box, self.crop_px, width, height)
        crop = image[y0:y1, x0:x1]
        prompts = {}
        if hint_box is not None:
            prompts["bboxes"] = [[hint_box[0] - x0, hint_box[1] - y0, hint_box[2] - x0, hint_box[3] - y0]]
        if points:
            # Varios clics para UN objeto: points=[[...]], labels=[[...]] (formato de Ultralytics).
            # -0.5: SAM, como OpenCV, pone el centro del píxel (0, 0) en (0, 0).
            prompts["points"] = [[[x - x0 - 0.5, y - y0 - 0.5] for x, y in points]]
            prompts["labels"] = [list(labels)]
        try:
            result = self.sam(crop, verbose=False, **prompts)[0]
        except Exception as error:   # por si esta versión no admite caja y puntos a la vez (como en clase)
            if "bboxes" not in prompts or "points" not in prompts:
                raise
            print(f"No se pudo combinar caja y clics ({error}); uso solo la caja")
            result = self.sam(crop, bboxes=prompts["bboxes"], verbose=False)[0]
        if result.masks is None or len(result.masks.xy) == 0 or len(result.masks.xy[0]) < 3:
            return None
        return polygon_box(result.masks.xy[0], x0, y0)


class Prelabeler:
    """Propuestas del pipeline actual (YOLO + EasyOCR). Se carga la primera vez que se usa."""

    def __init__(self, config, ocr_mode=None):
        self.config = config
        self.ocr_mode = ocr_mode or config["inference"]["ocr_mode"]
        self.pipeline = None

    def __call__(self, image, image_name):
        if self.pipeline is None:
            from ArtificialVision.pipeline import Pipeline
            print("Cargando YOLO y EasyOCR para las propuestas...")
            self.pipeline = Pipeline(self.config)
        return self.pipeline.process(image, image_name, self.ocr_mode)


# -----------------------------------------------------------------------------
# Ventana
# -----------------------------------------------------------------------------
class SamAnnotator:
    """Une la ventana de OpenCV, el estado de la foto, el zoom y SAM."""

    def __init__(self, photos, annotations, segmenter, max_box_px, prelabeler=None):
        self.photos, self.annotations = photos, annotations
        self.segmenter, self.max_box_px, self.prelabeler = segmenter, max_box_px, prelabeler
        self.window_size = (1280, 720)
        self.mouse = (0.0, 0.0)          # posición del ratón en la ventana
        self.press = self.drag = self.pan_from = None

    def annotation_path(self, i):
        return self.annotations / f"{self.photos[i].stem}.json"

    def open(self, i):
        self.i = i
        self.image = read_image(self.photos[i])
        self.height, self.width = self.image.shape[:2]
        self.view = View(self.width, self.height, self.window_size)
        path = self.annotation_path(i)
        if path.exists():
            self.state = LabelState.from_annotation(json_format.load(path))
        elif self.prelabeler is not None:
            self.state = LabelState.from_annotation(self.prelabeler(self.image, self.photos[i].name), prelabeled=True)
            print(f"{self.photos[i].name}: {len(self.state.entries)} propuestas del modelo; revisalas antes de pasar de foto")
        else:
            self.state = LabelState()

    def save(self):
        """Guarda la foto actual. Una foto que solo se ha mirado no se guarda (no tiene parejas)."""
        path = self.annotation_path(self.i)
        if self.state.entries or path.exists():
            json_format.save(self.state.to_annotation(self.photos[self.i].name, self.width, self.height), path)

    def leave(self):
        """Antes de cambiar de foto o salir: descarta la pareja a medias y guarda."""
        if self.state.has_current() and not self.state.backup:
            print("Pareja sin guardar (falta Intro): descartada")
        self.state.cancel()
        self.save()
        missing = self.state.missing_values()
        if missing:
            print(f"Aviso: {self.photos[self.i].name} tiene {missing} números sin valor")

    # ---------- SAM ----------
    def segment(self):
        state = self.state
        if not state.can_segment():
            return
        box = self.segmenter.segment(self.image, state.points, state.labels, state.hint_box)
        if plausible_box(box, self.max_box_px[state.active], state.points, state.labels):
            state.proposal = box
        elif state.hint_box is not None:
            state.proposal = [round(v, 2) for v in state.hint_box]
            print("SAM no ha dado una caja creible: uso la caja dibujada")
        else:
            state.proposal = None
            print("SAM no ha dado una caja creible: prueba otro clic o arrastra una caja")

    # ---------- dibujo ----------
    def draw_box(self, canvas, box, color, thickness=1):
        (x0, y0), (x1, y1) = self.view.to_window(box[0], box[1]), self.view.to_window(box[2], box[3])
        if x1 - x0 < 4:                                 # muy pequeña en pantalla: se agranda para que se vea
            x0, x1 = (x0 + x1) / 2 - 2, (x0 + x1) / 2 + 2
        if y1 - y0 < 4:
            y0, y1 = (y0 + y1) / 2 - 2, (y0 + y1) / 2 + 2
        cv2.rectangle(canvas, (round(x0), round(y0)), (round(x1), round(y1)), color, thickness)
        return round(x0), round(y0)

    def draw_entry(self, canvas, entry, color=None, thickness=1):
        if entry["dot"] and entry["number"]:
            a = self.view.to_window(*box_center(entry["dot"]))
            b = self.view.to_window(*box_center(entry["number"]))
            cv2.line(canvas, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color or (160, 160, 160), 1)
        if entry["dot"]:
            self.draw_box(canvas, entry["dot"], color or COLORS["dot"], thickness)
        if entry["number"]:
            x, y = self.draw_box(canvas, entry["number"], color or COLORS["number"], thickness)
            on_screen = (entry["number"][3] - entry["number"][1]) * self.view.scale()
        elif entry["dot"] and entry["text"]:           # valor tecleado antes de marcar el número
            x, y = self.view.to_window(*box_center(entry["dot"]))
            x, y, on_screen = round(x) + 8, round(y), 99
        else:
            return                                     # sin número no hay valor que dibujar
        # El valor solo se escribe si el número se ve lo bastante grande para compararlo con el
        # impreso; si no, tapa la ficha. Los números sin valor ("?" en rojo) se marcan siempre.
        if not entry["text"]:
            write(canvas, "?", (x, y - 4), (0, 0, 255), 0.5)
        elif on_screen >= 12 or color is not None:
            write(canvas, entry["text"], (x, y - 4), color or COLORS["number"], 0.5)

    def render(self):
        view, state = self.view, self.state
        view.window = self.window_size
        flags = cv2.INTER_NEAREST if view.scale() >= 2 else cv2.INTER_LINEAR   # con zoom, píxeles nítidos
        canvas = cv2.warpAffine(self.image, view.matrix(), self.window_size, flags=flags,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND)
        for entry in state.entries:
            self.draw_entry(canvas, entry)
        self.draw_entry(canvas, state.current, CURRENT, 2)
        if state.hint_box is not None:
            self.draw_box(canvas, state.hint_box, (0, 140, 255))
        if state.proposal is not None:
            self.draw_box(canvas, state.proposal, PROPOSAL, 2)
        for (x, y), label in zip(state.points, state.labels):
            wx, wy = view.to_window(x, y)
            cv2.circle(canvas, (round(wx), round(wy)), 4, (0, 200, 0) if label else (0, 0, 230), -1)
        if self.drag is not None:
            cv2.rectangle(canvas, self.drag[:2], self.drag[2:], (0, 140, 255), 1)

        name = self.photos[self.i].name
        header = (f"{self.i + 1}/{len(self.photos)} {name} | parejas: {len(state.entries)} | "
                  f"sin valor: {state.missing_values()} | marcando: {'PUNTO' if state.active == 'dot' else 'NUMERO'}"
                  f" {state.current['text']} | zoom {round(view.zoom * 100)} %" + (" | PRELABEL" if state.prelabeled else ""))
        width, height = self.window_size
        cv2.rectangle(canvas, (0, 0), (width, 30), BACKGROUND, -1)            # bandas para que el texto se lea
        cv2.rectangle(canvas, (0, height - 44), (width, height), BACKGROUND, -1)
        write(canvas, header, (10, 21), (255, 255, 255), 0.5)
        help_lines = ["izq: si  der: no  arrastrar: caja  ESPACIO: fijar  0-9: valor  Intro: guardar pareja  Tab: punto/numero",
                      "e: editar  x: borrar  z: deshacer  c: cancelar  rueda: zoom  central: mover  f: entera  a/d: foto  q: salir"]
        for k, line in enumerate(help_lines):
            write(canvas, line, (10, height - 26 + 18 * k), (255, 255, 255), 0.42)
        return canvas

    # ---------- eventos ----------
    def on_mouse(self, event, x, y, flags, _):
        wx, wy = x + 0.5, y + 0.5                        # centro del píxel de la ventana
        if event == cv2.EVENT_MOUSEMOVE:
            if self.pan_from is not None:
                self.view.pan(wx - self.pan_from[0], wy - self.pan_from[1])
                self.pan_from = (wx, wy)
            elif self.press is not None and (abs(x - self.press[0]) > 6 or abs(y - self.press[1]) > 6):
                self.drag = (self.press[0], self.press[1], x, y)
            self.mouse = (wx, wy)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.press = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.press is not None:
            if self.drag is not None:                     # se ha arrastrado: caja
                x0, y0 = self.view.to_image(self.drag[0] + 0.5, self.drag[1] + 0.5)
                x1, y1 = self.view.to_image(wx, wy)
                x0, x1 = (min(max(v, 0), self.width) for v in (x0, x1))     # la caja no sale de la foto
                y0, y1 = (min(max(v, 0), self.height) for v in (y0, y1))
                self.state.set_hint_box(x0, y0, x1, y1)
            else:                                         # clic: "esto sí"
                self.state.add_point(*self.view.to_image(wx, wy), positive=True)
            self.press = self.drag = None
            self.segment()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.state.add_point(*self.view.to_image(wx, wy), positive=False)
            self.segment()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.pan_from = (wx, wy)
        elif event == cv2.EVENT_MBUTTONUP:
            self.pan_from = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.view.zoom_at(wx, wy, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)

    def on_key(self, key):
        """Devuelve False para salir."""
        state = self.state
        if key in (ord("q"), KEY_ESC):
            self.leave()
            return False
        if ord("0") <= key <= ord("9"):
            state.type_digit(chr(key))
        elif key == KEY_BACKSPACE:
            state.backspace()
        elif key == KEY_SPACE:
            state.confirm()
        elif key in (KEY_ENTER, 10):
            if state.accept():
                self.save()
        elif key == KEY_TAB:
            state.switch()
        elif key == ord("c"):
            state.cancel()
        elif key == ord("e"):
            state.edit_at(*self.view.to_image(*self.mouse))
        elif key == ord("x"):
            if state.delete_at(*self.view.to_image(*self.mouse)):
                self.save()
        elif key == ord("z"):
            state.undo()
            self.save()
        elif key == ord("f"):
            self.view.reset()
        elif key in (ord("d"), ord("a")):
            step = 1 if key == ord("d") else -1
            if 0 <= self.i + step < len(self.photos):
                self.leave()
                self.open(self.i + step)
        return True


def write(canvas, text, origin, color, size):
    """Escribe una línea de texto.

    Sin el borde negro de la herramienta de clase: con la fuente de OpenCV 5, el texto grueso y el
    fino no avanzan igual y el borde se va separando de las letras. Por eso las barras de arriba y
    abajo llevan una banda oscura detrás.
    """
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Etiqueta puntos y números de fotos reales con SAM 2.1.")
    parser.add_argument("folder", type=Path, nargs="?", default=project_path(config["real_data"]["images"]),
                        help="carpeta con las fotos")
    parser.add_argument("--annotations", type=Path, default=project_path(config["real_data"]["annotations"]),
                        help="carpeta de los JSON")
    parser.add_argument("--prelabel", action="store_true", help="el pipeline actual propone las parejas")
    parser.add_argument("--mode", choices=["page", "crops"], help="modo de OCR de las propuestas (por defecto, el de config.yaml)")
    parser.add_argument("--start", type=int, help="empezar en la foto número N (por defecto, la primera sin anotar)")
    args = parser.parse_args()

    photos = sorted(path for path in args.folder.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS) \
        if args.folder.exists() else []
    if not photos:
        raise SystemExit(f"No hay fotos en {args.folder}. Sácalas antes con: python -m tools.extract_frames")
    args.annotations.mkdir(parents=True, exist_ok=True)

    c_annotation = config["annotation"]
    print(f"{len(photos)} fotos. Cargando SAM ({c_annotation['sam_model']}; la primera vez se descarga)...")
    segmenter = SamSegmenter(project_path(c_annotation["sam_model"]), c_annotation["crop_px"])
    prelabeler = Prelabeler(config, args.mode) if args.prelabel else None
    annotator = SamAnnotator(photos, args.annotations, segmenter, c_annotation["max_box_px"], prelabeler)

    if args.start:
        first = min(max(args.start, 1), len(photos)) - 1
    else:
        first = next((i for i in range(len(photos)) if not annotator.annotation_path(i).exists()), 0)
    annotator.open(first)

    # Ventana redimensionable: se le da cada vez una imagen del tamaño exacto de la ventana
    # (como en la herramienta de clase), así OpenCV no la estira y los clics caen donde se ve.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fit = min(1.0, 1400 / annotator.width, 850 / annotator.height)
    annotator.window_size = (max(640, round(annotator.width * fit)), max(400, round(annotator.height * fit)))
    cv2.resizeWindow(WINDOW, *annotator.window_size)
    cv2.setMouseCallback(WINDOW, annotator.on_mouse)
    while True:
        try:
            _, _, width, height = cv2.getWindowImageRect(WINDOW)
            if width > 0 and height > 0:
                annotator.window_size = (width, height)
        except cv2.error:
            pass                                     # si no se puede leer, se mantiene el último tamaño
        cv2.imshow(WINDOW, annotator.render())
        key = cv2.waitKey(30)
        if key != -1 and not annotator.on_key(key & 0xFF):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # ventana cerrada con la X
            annotator.leave()
            break
    cv2.destroyAllWindows()
    done = sum(1 for i in range(len(photos)) if annotator.annotation_path(i).exists())
    print(f"Fotos anotadas: {done}/{len(photos)} en {args.annotations}")


if __name__ == "__main__":
    main()
