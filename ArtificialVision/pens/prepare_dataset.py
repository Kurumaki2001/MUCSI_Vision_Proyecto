"""Monta el dataset de YOLO de los rotuladores con las fotos etiquetadas (como en las prácticas de SAM).

La validación se separa POR VÍDEO: todas las fotos de los vídeos indicados en --val van a
validación y el resto a entrenamiento. Así no hay fotos casi iguales en los dos lados y la
validación mide de verdad cómo funciona con fotos nuevas.

Uso, desde la carpeta del proyecto:
    python -m ArtificialVision.pens.prepare_dataset                 # validación = el último vídeo
    python -m ArtificialVision.pens.prepare_dataset --val v03 v04

Resultado (pens.dataset en config.yaml): images/{train,val}, labels/{train,val} y data.yaml.
"""

import argparse
import collections
import shutil

import yaml

from common.config import load_config, project_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    cfg = load_config()["pens"]
    parser = argparse.ArgumentParser(description="Monta el dataset YOLO de los rotuladores.")
    parser.add_argument("--val", nargs="+", help="prefijos de vídeo para validación (p. ej. v03). Por defecto, el último")
    args = parser.parse_args()

    frames, labels, output = project_path(cfg["frames"]), project_path(cfg["labels"]), project_path(cfg["dataset"])
    if not frames.is_dir() or not labels.is_dir():
        raise SystemExit(f"Faltan {frames} o {labels}. Antes: python -m ArtificialVision.pens.frames y "
                         "python -m ArtificialVision.pens.label_sam")
    # Solo las fotos que tienen su .txt de etiquetas (las que se han mirado y etiquetado)
    pairs = [(image, labels / f"{image.stem}.txt") for image in sorted(frames.iterdir())
             if image.suffix.lower() in IMAGE_EXTENSIONS and (labels / f"{image.stem}.txt").exists()]
    if not pairs:
        raise SystemExit("No hay fotos etiquetadas. Etiqueta antes con python -m ArtificialVision.pens.label_sam")

    videos = sorted({image.name.split("_")[0] for image, _ in pairs})
    validation = args.val or [videos[-1]]
    if len(videos) < 2:
        print("Aviso: todas las fotos son del mismo vídeo; la validación no será independiente.")

    shutil.rmtree(output, ignore_errors=True)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True)

    counts = collections.Counter()
    pens = collections.Counter()
    for image, label in pairs:
        split = "val" if image.name.split("_")[0] in validation else "train"
        shutil.copy2(image, output / "images" / split / image.name)
        shutil.copy2(label, output / "labels" / split / label.name)
        counts[split] += 1
        pens[split] += sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())

    data = {"path": str(output), "train": "images/train", "val": "images/val",
            "names": dict(enumerate(cfg["classes"]))}
    (output / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Vídeos: {', '.join(videos)} -> validación: {', '.join(validation)}")
    for split in ("train", "val"):
        print(f"  {split:5s}: {counts[split]} fotos, {pens[split]} rotuladores")
    print(f"Dataset en {output}")


if __name__ == "__main__":
    main()
