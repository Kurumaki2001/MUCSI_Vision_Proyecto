"""Monta el dataset de YOLO de los rotuladores con las fotos etiquetadas (como en las prácticas de SAM).

La validación se separa POR VÍDEO: todas las fotos de los vídeos indicados en --val van a
validación y el resto a entrenamiento. Así no hay fotos casi iguales en los dos lados y la
validación mide de verdad cómo funciona con fotos nuevas.
Si hay mapa de etiquetas, cada pareja de archivos debe referir al mismo vídeo en ambos mapas.
Si discrepan, comprueba el origen de las etiquetas y vuelve a extraer los vídeos antes de montar el dataset.

Uso, desde la carpeta del proyecto:
    python -m ArtificialVision.pens.prepare_dataset                 # validación = el último vídeo
    python -m ArtificialVision.pens.prepare_dataset --val v03 v04

Resultado (pens.dataset en config.yaml): images/{train,val}, labels/{train,val} y data.yaml.
"""

import argparse
import collections
import re
import shutil
import tempfile
from pathlib import Path

import yaml

from ArtificialVision.pens.frames import video_mapping
from common.config import load_config, project_path
from tools.extract_frames import MAPPING_FILE, read_mapping

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DATASET_MARKER = ".pens-dataset"


def validate_output(output, sources):
    """Acepta una salida vacía o un dataset de rotuladores de una ejecución anterior."""
    if output.is_symlink() or output.is_junction():
        raise ValueError(f"La salida {output} es un enlace a otra carpeta.")
    output = output.resolve()
    for source in sources:
        source = source.resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"La salida {output} se solapa con la carpeta de origen {source}.")
    if not output.exists():
        return output
    if not output.is_dir():
        raise ValueError(f"La salida {output} debe ser una carpeta.")

    entries = {entry.name for entry in output.iterdir()}
    if not entries:
        return output
    if entries <= {DATASET_MARKER, "data.yaml", "images", "labels"}:
        if (output / DATASET_MARKER).is_file():
            return output
        # Reconoce también los datasets creados antes de introducir la marca.
        manifest = output / "data.yaml"
        if manifest.is_file():
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            if (isinstance(data, dict) and data.get("path") == str(output)
                    and data.get("train") == "images/train" and data.get("val") == "images/val"
                    and data.get("names") == {0: "rotulador"}):
                return output
    raise ValueError(f"La salida {output} contiene archivos ajenos al dataset de rotuladores.")


def label_count(path, classes):
    """Valida cajas YOLO normalizadas. Un archivo vacío es un ejemplo negativo válido."""
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            class_id, x, y, width, height = line.split()
            x, y, width, height = map(float, (x, y, width, height))
            if not (0 <= int(class_id) < len(classes) and 0 < width <= 1 and 0 < height <= 1
                    and 0 <= x <= 1 and 0 <= y <= 1
                    and width / 2 - 1e-6 <= x <= 1 - width / 2 + 1e-6
                    and height / 2 - 1e-6 <= y <= 1 - height / 2 + 1e-6):
                raise ValueError
        except ValueError as error:
            raise ValueError(f"Etiqueta YOLO no válida en {path}, línea {line_number}.") from error
        count += 1
    return count


def publish_dataset(staged, output, sources):
    """Sustituye el dataset completo y restaura el anterior si falla la publicación."""
    validate_output(output, sources)
    previous = staged.parent.with_name(f"{staged.parent.name}-anterior")
    if output.exists():
        output.rename(previous)
    try:
        staged.rename(output)
    except OSError:
        if previous.exists():
            try:
                previous.rename(output)
            except OSError as error:
                raise OSError(f"El dataset anterior se conserva en {previous}.") from error
        raise
    if previous.exists():
        if previous.resolve() != previous or previous.parent != output.parent:
            raise OSError(f"No se pudo comprobar la ruta del dataset anterior: {previous}.")
        shutil.rmtree(previous)


def main():
    cfg = load_config()["pens"]
    parser = argparse.ArgumentParser(description="Monta el dataset YOLO de los rotuladores.")
    parser.add_argument("--val", nargs="+", help="prefijos de vídeo para validación (p. ej. v03). Por defecto, el último")
    args = parser.parse_args()

    frames, labels, output = project_path(cfg["frames"]), project_path(cfg["labels"]), project_path(cfg["dataset"])
    sources = (frames, labels, project_path(cfg["videos"]))
    if not frames.is_dir() or not labels.is_dir():
        raise SystemExit(f"Faltan {frames} o {labels}. Antes: python -m ArtificialVision.pens.frames y "
                         "python -m ArtificialVision.pens.label_sam")
    # Solo las fotos que tienen su .txt de etiquetas (las que se han mirado y etiquetado)
    pairs = [(image, labels / f"{image.stem}.txt") for image in sorted(frames.iterdir())
             if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
             and (labels / f"{image.stem}.txt").is_file()]
    if not pairs:
        raise SystemExit("No hay fotos etiquetadas. Etiqueta antes con python -m ArtificialVision.pens.label_sam")
    if len({image.stem.casefold() for image, _ in pairs}) != len(pairs):
        parser.error("Hay varios fotogramas con el mismo nombre base y una sola etiqueta.")

    frames_map_path = frames / MAPPING_FILE
    labels_map_path = labels / MAPPING_FILE
    try:
        output = validate_output(output, sources)
        frames_map = read_mapping(frames_map_path)
        labels_map = read_mapping(labels_map_path)
        video_mapping(labels_map, frames_map)
    except (ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    videos = {image.stem.split("_")[0].lower() for image, _ in pairs}
    if labels_map_path.exists() or frames_map_path.exists():
        missing_frames = sorted(videos - set(frames_map.values()))
        if missing_frames:
            parser.error(f"Falta el vídeo de {', '.join(missing_frames)} en {frames_map_path}. "
                         "Vuelve a extraer los fotogramas para comprobar sus identificadores.")
    if labels_map_path.exists():
        missing_labels = sorted(videos - set(labels_map.values()))
        if missing_labels:
            parser.error(f"Falta el vídeo de {', '.join(missing_labels)} en {labels_map_path}. "
                         "Comprueba el origen de las etiquetas antes de montar el dataset.")

    validation = {video.lower() for video in (args.val or [max(videos)])}
    unknown = sorted(validation - videos)
    if unknown:
        parser.error(f"Vídeos de validación sin fotos etiquetadas: {', '.join(unknown)}.")
    if validation == videos:
        parser.error("El entrenamiento y la validación necesitan al menos un vídeo distinto cada uno.")

    paired = {label.stem.casefold() for _, label in pairs}
    unused = sum(1 for label in labels.glob("*.txt") if label.is_file()
                 and re.fullmatch(r"v[0-9]+_[0-9]+", label.stem) and label.stem.casefold() not in paired)
    if unused:
        print(f"Aviso: {unused} etiquetas sin fotograma correspondiente quedan fuera del dataset.")

    counts = collections.Counter()
    pens = collections.Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as folder:
        staged = Path(folder) / "dataset"
        for split in ("train", "val"):
            for kind in ("images", "labels"):
                (staged / kind / split).mkdir(parents=True)

        for image, label in pairs:
            split = "val" if image.stem.split("_")[0].lower() in validation else "train"
            shutil.copy2(image, staged / "images" / split / image.name)
            copied_label = staged / "labels" / split / label.name
            shutil.copy2(label, copied_label)
            try:
                pens[split] += label_count(copied_label, cfg["classes"])
            except ValueError as error:
                parser.error(str(error).replace(str(copied_label), str(label)))
            counts[split] += 1

        data = {"path": str(output), "train": "images/train", "val": "images/val",
                "names": dict(enumerate(cfg["classes"]))}
        (staged / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (staged / DATASET_MARKER).touch()
        publish_dataset(staged, output, sources)
    print(f"Vídeos: {', '.join(sorted(videos))} -> validación: {', '.join(sorted(validation))}")
    for split in ("train", "val"):
        print(f"  {split:5s}: {counts[split]} fotos, {pens[split]} rotuladores")
    print(f"Dataset en {output}")


if __name__ == "__main__":
    main()
