"""Saca una foto de cada N fotogramas de los vídeos de los rotuladores.

Usa las funciones de tools/extract_frames.py (el mismo extractor que para las fichas reales), con
las carpetas de los rotuladores por defecto. Cada foto se llama vXX_NNNNN.jpg: número de vídeo y
número de fotograma. El número de cada vídeo queda en <salida>/videos.txt y se respeta en las
siguientes ejecuciones. Así, al montar el dataset, la validación se separa por vídeo y no hay
fotos casi iguales en entrenamiento y en validación.

Cómo grabar los vídeos: ver Documentacion_Rotuladores/video.md (7 vídeos de 30-45 s, cámara fija y
cenital, los tres colores en cada vídeo...).

Uso, desde la carpeta del proyecto:
    python -m ArtificialVision.pens.frames                          # todos los vídeos de data/pens/videos
    python -m ArtificialVision.pens.frames data/pens/videos --every 30
    python -m ArtificialVision.pens.frames video1.mp4 video2.mp4 --min-sharpness 50
"""

import argparse
from pathlib import Path

from common.config import load_config, project_path
from tools.extract_frames import (MAPPING_FILE, VIDEO_EXTENSIONS, assign_ids, extract_video, read_mapping,
                                  write_mapping)


def main():
    config = load_config()
    cfg, real = config["pens"], config["real_data"]
    parser = argparse.ArgumentParser(description="Saca fotogramas de los vídeos de los rotuladores.")
    parser.add_argument("videos", nargs="*", type=Path, help=f"vídeos o carpetas (por defecto, {cfg['videos']})")
    parser.add_argument("--every", type=int, default=15, help="guardar una foto de cada N fotogramas (15)")
    parser.add_argument("--out", type=Path, default=project_path(cfg["frames"]), help="carpeta de salida")
    parser.add_argument("--min-sharpness", type=float, default=0,
                        help="descartar fotogramas con menos nitidez que esta (0 = no descartar)")
    args = parser.parse_args()

    videos = []
    for item in args.videos or [project_path(cfg["videos"])]:
        if item.is_dir():
            videos += sorted(p for p in item.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        elif item.exists():
            videos.append(item)
    if not videos:
        raise SystemExit(f"No hay vídeos. Copia los vídeos a {cfg['videos']} o indícalos en la línea de órdenes.")

    args.out.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out / MAPPING_FILE
    mapping = assign_ids([video.name for video in videos], read_mapping(mapping_path))
    write_mapping(mapping_path, mapping)
    total = 0
    for video in videos:
        read, saved, blurry = extract_video(video, mapping[video.name], args.out, args.every, args.min_sharpness,
                                            real["jpeg_quality"])
        note = f", {blurry} descartadas por movidas" if blurry else ""
        print(f"{mapping[video.name]} = {video.name}: {read} fotogramas -> {saved} fotos{note}")
        total += saved
    print(f"Total: {total} fotos en {args.out} (qué vídeo es cada vXX: {mapping_path})")


if __name__ == "__main__":
    main()
