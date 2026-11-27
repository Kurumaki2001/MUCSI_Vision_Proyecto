"""Lectura y escritura de imágenes con OpenCV.

cv2.imread y cv2.imwrite pueden fallar en Windows si la ruta tiene caracteres no ASCII (tildes,
eñes). Para evitarlo, se leen los bytes del fichero con NumPy y se decodifican con cv2.imdecode;
al guardar, se codifican con cv2.imencode y se escriben con NumPy.
"""

from pathlib import Path

import cv2
import numpy as np


def check_unique_image_stems(paths):
    """Evita que dos imágenes escriban sobre los mismos archivos de salida."""
    names = {}
    for path in paths:
        key = path.stem.casefold()
        if key in names:
            raise SystemExit(f"{names[key]} y {path.name} tienen el mismo nombre de salida. Renombra una de las fotos.")
        names[key] = path.name


def read_image(path):
    """Lee una imagen en color. Como todo en OpenCV, los canales van en orden BGR."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se puede leer la imagen {path}")
    return image


def write_image(path, image, jpeg_quality=95):
    """Guarda una imagen; si es JPEG, con la calidad indicada (0-100)."""
    suffix = Path(path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if suffix in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise OSError(f"No se puede codificar la imagen {path}")
    encoded.tofile(str(path))
