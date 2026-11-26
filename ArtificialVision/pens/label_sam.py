"""Etiquetado de los rotuladores con SAM 2.1: un clic por rotulador -> etiquetas YOLO.

Es la herramienta de etiquetado de las prácticas de SAM, con los valores por defecto del
proyecto: una sola clase (0 = rotulador; el color lo decide después el HSV, no YOLO), las fotos de
data/pens/frames y las etiquetas en data/pens/labels.

Uso, desde la carpeta del proyecto:
  python -m ArtificialVision.pens.label_sam                       # fotos y etiquetas por defecto
  python -m ArtificialVision.pens.label_sam otra/carpeta --out otras/etiquetas

Ratón:
  clic izquierdo   punto "esto sí" del rotulador que estás marcando
  clic derecho     punto "esto no" (para quitar sombras, mesa, el rotulador de al lado...)
  arrastrar        (botón izquierdo) dibujar una caja alrededor del rotulador; se puede
                   combinar con puntos "sí" / "no" para afinar
Teclado:
  ESPACIO / ENTER  aceptar el rotulador (se guarda su caja)
  c                cancelar el rotulador que estás marcando
  z                borrar la última caja aceptada de esta foto
  d                guardar y pasar a la foto siguiente
  a                guardar y volver a la anterior
  (ventana)        estírala o maximízala: la foto se adapta sin deformarse
  q / ESC          guardar y salir

Consejo: marca TODOS los rotuladores de cada foto, también los cortados por el borde o tapados a
medias. Un rotulador sin marcar le enseña a YOLO que "eso no es un rotulador". No marques los
capuchones sueltos ni otros objetos (ver Documentacion_Rotuladores/video.md).

Cada foto data/pens/frames/v01_00030.jpg genera data/pens/labels/v01_00030.txt con una línea por
rotulador:  clase  centro_x  centro_y  ancho  alto   (todo dividido por el tamaño de la foto)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import SAM

COLORES = [(170, 95, 0), (60, 170, 60), (40, 130, 230), (180, 60, 180), (30, 200, 220),
           (200, 200, 60), (90, 90, 200), (150, 150, 150), (60, 60, 60), (0, 0, 200)]
EXTS = (".jpg", ".jpeg", ".png")


def leer_etiquetas(ruta, w, h):
    """Lee un .txt YOLO y devuelve [(clase, x1, y1, x2, y2)] en píxeles."""
    cajas = []
    if os.path.exists(ruta):
        for linea in open(ruta, encoding="utf-8"):
            p = linea.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                cajas.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return cajas


def escribir_etiquetas(ruta, cajas, w, h):
    """Guarda las cajas en formato YOLO: clase cx cy ancho alto, normalizado entre 0 y 1."""
    with open(ruta, "w", encoding="utf-8") as f:
        for c, x1, y1, x2, y2 in cajas:
            f.write(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


class Etiquetador:
    def __init__(self, fotos, salida, clases, modelo, ancho_max):
        self.fotos, self.salida, self.clases = fotos, salida, clases
        self.sam = SAM(modelo)
        self.ancho_max = ancho_max
        self.i, self.clase = 0, 0
        self.abrir(0)

    # ---------- foto actual ----------
    def abrir(self, i):
        self.i = i
        self.img = cv2.imread(self.fotos[i])
        self.h, self.w = self.img.shape[:2]
        self.escala, self.dx, self.dy = 1.0, 0, 0  # se recalculan al dibujar, según el tamaño de la ventana
        self.txt = os.path.join(self.salida, os.path.splitext(os.path.basename(self.fotos[i]))[0] + ".txt")
        self.cajas = leer_etiquetas(self.txt, self.w, self.h)
        self.cancelar()

    def guardar(self):
        # Una foto que solo has mirado no se guarda: un .txt vacío le diría a YOLO "aquí no hay rotuladores".
        if self.cajas or os.path.exists(self.txt):
            escribir_etiquetas(self.txt, self.cajas, self.w, self.h)

    # ---------- rotulador que se está marcando ----------
    def cancelar(self):
        self.puntos, self.signos, self.contorno, self.caja = [], [], None, None
        self.caja_pista = None                # caja dibujada arrastrando el ratón
        self.arrastre = None                  # (x0, y0, x1, y1) mientras se arrastra, en la ventana

    def anadir_punto(self, x, y, signo):
        self.puntos.append([int(x), int(y)])
        self.signos.append(signo)
        self.segmentar()

    def poner_caja(self, x1, y1, x2, y2):
        self.caja_pista = [int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))]
        self.segmentar()

    def segmentar(self):
        """Pide a SAM la máscara con las pistas que haya: caja, puntos o ambas."""
        if self.caja_pista is None and 1 not in self.signos:   # sin caja ni ningún "sí" no hay rotulador
            return
        pistas = {}
        if self.caja_pista is not None:
            pistas["bboxes"] = [self.caja_pista]
        if self.puntos:
            # Varios puntos para UN rotulador: points=[[...]], labels=[[...]] (formato de Ultralytics)
            pistas["points"], pistas["labels"] = [self.puntos], [self.signos]
        try:
            r = self.sam(self.img, verbose=False, **pistas)[0]
        except Exception as e:
            if "bboxes" in pistas and "points" in pistas:     # por si esta versión no admite ambas a la vez
                print(f"No se pudo combinar caja y puntos ({e}); uso solo la caja")
                r = self.sam(self.img, bboxes=[self.caja_pista], verbose=False)[0]
            else:
                raise
        if r.masks is None or len(r.masks.xy) == 0 or len(r.masks.xy[0]) < 3:
            self.contorno, self.caja = None, None
            print("SAM no ha encontrado nada ahí; prueba otro punto")
            return
        pol = r.masks.xy[0]               # contorno en píxeles de la foto original
        self.contorno = pol.astype(np.int32)
        (x1, y1), (x2, y2) = pol.min(0), pol.max(0)
        self.caja = (float(x1), float(y1), float(x2), float(y2))

    def aceptar(self):
        if self.caja is not None:
            self.cajas.append((self.clase, *self.caja))
            self.guardar()
        self.cancelar()

    # ---------- dibujo ----------
    def nombre(self, c):
        return self.clases[c] if c < len(self.clases) else f"clase {c}"

    def dibujar(self, ancho_v, alto_v):
        """Devuelve una imagen del tamaño exacto de la ventana, con la foto escalada sin deformar."""
        vis = self.img.copy()
        for c, x1, y1, x2, y2 in self.cajas:
            col = COLORES[c % len(COLORES)]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, self.nombre(c), (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if self.contorno is not None:
            capa = vis.copy()
            cv2.fillPoly(capa, [self.contorno], COLORES[self.clase % len(COLORES)])
            vis = cv2.addWeighted(capa, 0.4, vis, 0.6, 0)
            x1, y1, x2, y2 = map(int, self.caja)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
        if self.caja_pista is not None:
            x1, y1, x2, y2 = self.caja_pista
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
        for (x, y), s in zip(self.puntos, self.signos):
            cv2.circle(vis, (x, y), 6, (0, 200, 0) if s else (0, 0, 230), -1)
            cv2.circle(vis, (x, y), 6, (255, 255, 255), 1)
        cab = (f"{self.i + 1}/{len(self.fotos)} {os.path.basename(self.fotos[self.i])}  |  "
               f"clase {self.clase}: {self.nombre(self.clase)}  |  rotuladores: {len(self.cajas)}")
        # Misma escala en ancho y alto (la menor de las dos): la foto cabe entera y no se deforma.
        # Lo que sobra se rellena con bandas grises, y el lienzo mide justo lo que la ventana.
        self.escala = min(ancho_v / self.w, alto_v / self.h)
        nw, nh = max(1, round(self.w * self.escala)), max(1, round(self.h * self.escala))
        interp = cv2.INTER_AREA if self.escala < 1 else cv2.INTER_LINEAR
        lienzo = np.full((alto_v, ancho_v, 3), 40, np.uint8)
        self.dx, self.dy = (ancho_v - nw) // 2, (alto_v - nh) // 2
        lienzo[self.dy:self.dy + nh, self.dx:self.dx + nw] = cv2.resize(vis, (nw, nh), interpolation=interp)
        if self.arrastre is not None:                     # caja que se está dibujando (coordenadas de ventana)
            ax0, ay0, ax1, ay1 = self.arrastre
            cv2.rectangle(lienzo, (ax0, ay0), (ax1, ay1), (0, 220, 255), 1)
        cab += f"  |  {round(self.escala * 100)} %"
        for grosor, color in ((4, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(lienzo, cab, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, grosor)
            cv2.putText(lienzo, "clic izq: si  der: no  arrastrar: caja  ESPACIO: aceptar  c: cancelar  z: deshacer  a/d: foto  q: salir",
                        (10, alto_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, grosor)
        return lienzo

    def a_foto(self, x, y):
        """De coordenadas de la ventana a píxeles de la foto original (quitando bandas y escala)."""
        return (x - self.dx) / self.escala, (y - self.dy) / self.escala

    def dentro(self, x, y):
        fx, fy = self.a_foto(x, y)
        return 0 <= fx < self.w and 0 <= fy < self.h

    # ---------- eventos ----------
    def raton(self, ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:                 # puede ser un clic o el inicio de una caja
            self.inicio = (x, y) if self.dentro(x, y) else None
        elif ev == cv2.EVENT_MOUSEMOVE and getattr(self, "inicio", None) is not None:
            x0, y0 = self.inicio
            if abs(x - x0) > 8 or abs(y - y0) > 8:      # se ha movido: es una caja
                self.arrastre = (x0, y0, x, y)
        elif ev == cv2.EVENT_LBUTTONUP and getattr(self, "inicio", None) is not None:
            x0, y0 = self.inicio
            self.inicio = None
            fx0, fy0 = self.a_foto(x0, y0)              # de la ventana a la foto original
            fx, fy = self.a_foto(x, y)
            fx, fy = min(max(fx, 0), self.w - 1), min(max(fy, 0), self.h - 1)
            if self.arrastre is not None:
                self.arrastre = None
                self.poner_caja(fx0, fy0, fx, fy)
            else:
                self.anadir_punto(fx, fy, 1)
        elif ev == cv2.EVENT_RBUTTONDOWN and self.dentro(x, y):
            self.anadir_punto(*self.a_foto(x, y), 0)

    def tecla(self, k):
        """Devuelve False para salir."""
        if k in (ord("q"), 27):
            self.guardar()
            return False
        if ord("0") <= k <= ord("9"):
            self.clase = k - ord("0")
        elif k in (ord(" "), 13, 10):
            self.aceptar()
        elif k == ord("c"):
            self.cancelar()
        elif k == ord("z") and self.cajas:
            self.cajas.pop()
            self.guardar()
        elif k == ord("d") and self.i < len(self.fotos) - 1:
            self.guardar()
            self.abrir(self.i + 1)
        elif k == ord("a") and self.i > 0:
            self.guardar()
            self.abrir(self.i - 1)
        return True


def main():
    from common.config import load_config, project_path
    cfg = load_config()["pens"]
    p = argparse.ArgumentParser(description="Etiqueta los rotuladores con SAM (clase 0 = rotulador).")
    p.add_argument("fotos", nargs="?", default=str(project_path(cfg["frames"])), help="carpeta con las fotos")
    p.add_argument("--out", default=str(project_path(cfg["labels"])), help="carpeta para los .txt")
    p.add_argument("--modelo", default=str(project_path(cfg["sam_model"])), help="pesos de SAM 2.1")
    p.add_argument("--ancho", type=int, default=1280, help="ancho máximo inicial de la ventana")
    p.add_argument("--alto", type=int, default=760, help="alto máximo inicial de la ventana")
    p.add_argument("--desde", type=int, default=1, help="empezar en la foto número N")
    args = p.parse_args()

    fotos = sorted(f for f in glob.glob(os.path.join(args.fotos, "*")) if f.lower().endswith(EXTS))
    if not fotos:
        sys.exit(f"No hay fotos .jpg/.png en {args.fotos}. Sácalas antes con python -m ArtificialVision.pens.frames")
    os.makedirs(args.out, exist_ok=True)
    clases = list(cfg["classes"])
    with open(os.path.join(args.out, "clases.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(clases) + "\n")
    print("Clases:", ", ".join(f"{i}={c}" for i, c in enumerate(clases)))
    print(f"{len(fotos)} fotos. Cargando {args.modelo} (la primera vez se descarga)...")

    et = Etiquetador(fotos, args.out, clases, args.modelo, args.ancho)
    et.abrir(max(0, min(len(fotos) - 1, args.desde - 1)))
    # Ventana redimensionable. Le damos cada vez una imagen del tamaño exacto de la ventana,
    # con la foto escalada por igual en ancho y alto: así OpenCV no tiene que estirarla.
    cv2.namedWindow("Etiquetar", cv2.WINDOW_NORMAL)
    s0 = min(1.0, args.ancho / et.w, args.alto / et.h)
    tam = (max(320, round(et.w * s0)), max(240, round(et.h * s0)))
    cv2.resizeWindow("Etiquetar", *tam)
    cv2.setMouseCallback("Etiquetar", et.raton)
    while True:
        try:
            _, _, aw, ah = cv2.getWindowImageRect("Etiquetar")
            if aw > 0 and ah > 0:
                tam = (aw, ah)
        except cv2.error:
            pass                                    # si no se puede leer, mantenemos el último tamaño
        cv2.imshow("Etiquetar", et.dibujar(*tam))
        k = cv2.waitKey(30) & 0xFF
        if k != 255 and not et.tecla(k):
            break
        if cv2.getWindowProperty("Etiquetar", cv2.WND_PROP_VISIBLE) < 1:   # ventana cerrada con la X
            et.guardar()
            break
    cv2.destroyAllWindows()
    hechas = sum(1 for f in fotos if os.path.exists(os.path.join(
        args.out, os.path.splitext(os.path.basename(f))[0] + ".txt")))
    print(f"Fotos con etiquetas: {hechas}/{len(fotos)} en {args.out}/")


if __name__ == "__main__":
    main()
