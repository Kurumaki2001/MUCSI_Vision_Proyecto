"""Saca una foto de cada N fotogramas de los vídeos de los rotuladores.

Usa las funciones de tools/extract_frames.py (el mismo extractor que para las fichas reales), con
las carpetas de los rotuladores por defecto. Cada foto se llama vXX_NNNNN.jpg: número de vídeo y
número de fotograma. Los números de las etiquetas se reservan aunque solo se extraiga un vídeo,
y <salida>/videos.txt conserva los nuevos números para las siguientes ejecuciones.
Si los mapas discrepan, comprueba el de las etiquetas y vuelve a extraer en una carpeta vacía.

Cómo grabar los vídeos: ver Documentacion_Rotuladores/video.md (7 vídeos de 30-45 s, cámara fija y
cenital, los tres colores en cada vídeo...).

Uso, desde la carpeta del proyecto:
    python -m ArtificialVision.pens.frames    # reproduce la extracción de las etiquetas: paso 20, nitidez 30
    python -m ArtificialVision.pens.frames data/pens/videos --every 20
    python -m ArtificialVision.pens.frames video1.mp4 video2.mp4 --min-sharpness 50
"""

import argparse
import math
import re
import tempfile
from pathlib import Path

from common.config import load_config, project_path
from tools.extract_frames import (MAPPING_FILE, VIDEO_EXTENSIONS, assign_ids, extract_video, read_mapping,
                                  write_mapping)

FRAME_NAME = re.compile(r"(v[0-9]{2,})_[0-9]{5,}\.jpg", re.IGNORECASE)


def video_mapping(labels, frames):
    """Combina los mapas de etiquetas y fotogramas sin reasignar nombres ni números."""
    merged, names_by_id, names_by_key = {}, {}, {}
    for mapping in (labels, frames):
        for name, video_id in mapping.items():
            if not name or not re.fullmatch(r"v[0-9]{2,}", video_id):
                raise ValueError(f"Identificador de vídeo no válido: {video_id!r} para {name!r}.")
            key = name.casefold()
            known_name = names_by_key.get(key, name)
            known_id = merged.get(known_name, video_id)
            if known_id != video_id or names_by_id.get(video_id, known_name) != known_name:
                raise ValueError(f"Los mapas de vídeos no coinciden: {name} tiene el identificador {video_id}.")
            merged[known_name] = video_id
            names_by_key[key] = known_name
            names_by_id[video_id] = known_name
    return merged


def replace_frames(video, video_id, output, every, min_sharpness, jpeg_quality):
    """Publica una extracción completa del vídeo y retira sus fotogramas obsoletos."""
    with tempfile.TemporaryDirectory(prefix=f".{video_id}-", dir=output) as folder:
        staged = Path(folder)
        read, saved, blurry = extract_video(video, video_id, staged, every, min_sharpness, jpeg_quality)
        if not read:
            return read, saved, blurry
        files = list(staged.glob(f"{video_id}_*.jpg"))
        generated = {path.name.casefold() for path in files}
        for path in files:
            path.replace(output / path.name)
        for path in output.glob(f"{video_id}_*.jpg"):
            if FRAME_NAME.fullmatch(path.name) and path.is_file() and path.name.casefold() not in generated:
                path.unlink()
    return read, saved, blurry


def main():
    config = load_config()
    cfg, real = config["pens"], config["real_data"]
    parser = argparse.ArgumentParser(description="Saca fotogramas de los vídeos de los rotuladores.")
    parser.add_argument("videos", nargs="*", type=Path, help=f"vídeos o carpetas (por defecto, {cfg['videos']})")
    parser.add_argument("--every", type=int, default=cfg.get("frame_step", 20),
                        help="guardar una foto de cada N fotogramas (por defecto, %(default)s)")
    parser.add_argument("--out", type=Path, default=project_path(cfg["frames"]), help="carpeta de salida")
    parser.add_argument("--min-sharpness", type=float, default=cfg.get("min_sharpness", 30),
                        help="nitidez mínima (por defecto, %(default)s; 0 = no descartar)")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every debe ser al menos 1.")
    if not math.isfinite(args.min_sharpness) or args.min_sharpness < 0:
        parser.error("--min-sharpness debe ser un número no negativo.")

    videos = []
    for item in args.videos or [project_path(cfg["videos"])]:
        if item.is_dir():
            videos += sorted(p for p in item.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        elif item.is_file():
            videos.append(item)
    if not videos:
        raise SystemExit(f"No hay vídeos. Copia los vídeos a {cfg['videos']} o indícalos en la línea de órdenes.")

    videos = list({(path.resolve(), path.name.casefold()): path for path in videos}.values())
    if len({video.name.casefold() for video in videos}) != len(videos):
        parser.error("Hay vídeos de distintas carpetas con el mismo nombre.")

    labels_path = project_path(cfg["labels"]) / MAPPING_FILE
    mapping_path = args.out / MAPPING_FILE
    if args.out.resolve() == labels_path.parent.resolve():
        parser.error("La carpeta de fotogramas debe ser distinta de la carpeta de etiquetas.")
    try:
        existing = read_mapping(mapping_path)
        existing_ids = set(existing.values())
        for path in args.out.glob("v*_*.jpg"):
            match = FRAME_NAME.fullmatch(path.name)
            if match and path.is_file() and match[1].lower() not in existing_ids:
                raise ValueError(f"El fotograma {path} no tiene vídeo en {mapping_path}.")
        merged = video_mapping(read_mapping(labels_path), existing)
    except ValueError as error:
        parser.error(f"{error} Comprueba {labels_path} y vuelve a extraer en una carpeta vacía.")
    known_names = {name.casefold(): name for name in merged}
    canonical_names = [known_names.get(video.name.casefold(), video.name) for video in videos]
    mapping = assign_ids(canonical_names, merged)
    args.out.mkdir(parents=True, exist_ok=True)
    write_mapping(mapping_path, mapping)
    total = 0
    for video, name in zip(videos, canonical_names):
        video_id = mapping[name]
        read, saved, blurry = replace_frames(video, video_id, args.out, args.every, args.min_sharpness,
                                             real["jpeg_quality"])
        note = f", {blurry} descartadas por movidas" if blurry else ""
        print(f"{video_id} = {video.name}: {read} fotogramas -> {saved} fotos{note}")
        total += saved
    print(f"Total: {total} fotos en {args.out} (qué vídeo es cada vXX: {mapping_path})")


if __name__ == "__main__":
    main()
