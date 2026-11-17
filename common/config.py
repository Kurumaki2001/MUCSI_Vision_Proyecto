"""Lectura de config.yaml y rutas del proyecto."""

import shutil
from pathlib import Path

import yaml

# Carpeta del proyecto: la que contiene config.yaml (una por encima de common/).
ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Lee config.yaml (o el fichero indicado) y lo devuelve como diccionario."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path):
    """Convierte una ruta de config.yaml, relativa a la carpeta del proyecto, en absoluta."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def clear_generated_folder(folder, marker):
    """Borra la carpeta de salida de una ejecución anterior para no mezclar datos de dos ejecuciones.

    Por seguridad, solo borra carpetas que contengan el fichero 'marker', que el script que las
    genera escribe siempre. Así, una ruta mal puesta en config.yaml no borra nada importante.
    """
    folder = Path(folder)
    if not folder.exists():
        return
    if not (folder / marker).exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} ya existe y no la ha creado este script (falta {marker}). "
                         "Bórrala a mano o cambia la ruta de salida en config.yaml.")
    shutil.rmtree(folder)
