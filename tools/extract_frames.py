"""Saca una foto de cada N fotogramas de los vídeos de las fichas reales.

Adaptado de sacar_fotogramas.py (práctica 2.1 de clase). Cada foto se llama vXX_NNNNN.jpg: número
de vídeo y número de fotograma. Así, al repartir las fotos en entrenamiento, validación y test, se
puede separar por vídeo: los fotogramas de un mismo vídeo son casi iguales y, si cayeran a los dos
lados, la validación saldría optimista.

Diferencias con el script de clase:
  - El número de cada vídeo se guarda en <salida>/videos.txt y se respeta en las siguientes
    ejecuciones. Así, un vídeo añadido otro día no cambia los números de los que ya estaban.
  - Opcionalmente descarta los fotogramas movidos (varianza del Laplaciano baja).
  - Lee y escribe con common/images.py, que admite tildes en las rutas.

Uso, desde la carpeta del proyecto:
    python -m tools.extract_frames                                # todos los vídeos de real_data.videos
    python -m tools.extract_frames data/real/videos/ficha_03.mp4  # solo algunos vídeos
    python -m tools.extract_frames --step 10 --min-sharpness 50
"""

import argparse
from pathlib import Path

import cv2

from common.config import load_config, project_path
from common.images import write_image

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAPPING_FILE = "videos.txt"


def read_mapping(path):
    """Lee videos.txt: {nombre del vídeo: "vXX"}. Si no existe, devuelve un diccionario vacío."""
    mapping = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                video_id, name = line.split("\t", 1)
                mapping[name] = video_id
    return mapping


def assign_ids(names, mapping):
    """Da un "vXX" a cada vídeo: el que ya tenía en mapping o, si es nuevo, el siguiente libre.

    Los vídeos se identifican por el nombre del fichero (sin carpeta), así que da igual desde
    dónde se lancen. Devuelve un diccionario nuevo con los antiguos y los nuevos.
    """
    mapping = dict(mapping)
    used = [int(video_id[1:]) for video_id in mapping.values()]
    next_number = max(used, default=0) + 1
    for name in names:
        if name not in mapping:
            mapping[name] = f"v{next_number:02d}"
            next_number += 1
    return mapping


def write_mapping(path, mapping):
    """Escribe videos.txt ordenado por número de vídeo: "vXX<tabulador>nombre"."""
    lines = [f"{video_id}\t{name}" for name, video_id in sorted(mapping.items(), key=lambda item: item[1])]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_name(video_id, frame_index):
    """Nombre de la foto del fotograma frame_index del vídeo video_id, por ejemplo v03_00045.jpg."""
    return f"{video_id}_{frame_index:05d}.jpg"


def sharpness(image):
    """Nitidez: varianza del Laplaciano de la foto en gris. Baja = foto movida o desenfocada."""
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def extract_video(path, video_id, output, step, min_sharpness, jpeg_quality):
    """Guarda una foto de cada 'step' fotogramas. Devuelve (fotogramas leídos, guardadas, movidas)."""
    if step < 1:
        raise ValueError("El paso de extracción debe ser al menos 1.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        print(f"No se puede abrir {path}")
        return 0, 0, 0
    read = saved = blurry = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if read % step == 0:
                if min_sharpness > 0 and sharpness(frame) < min_sharpness:
                    blurry += 1
                else:
                    write_image(output / frame_name(video_id, read), frame, jpeg_quality)
                    saved += 1
            read += 1
    finally:
        capture.release()
    return read, saved, blurry


def main():
    config = load_config()["real_data"]
    parser = argparse.ArgumentParser(description="Saca fotogramas de los vídeos de las fichas reales.")
    parser.add_argument("videos", nargs="*", type=Path,
                        help=f"vídeos (por defecto, todos los de {config['videos']})")
    parser.add_argument("--out", type=Path, default=project_path(config["images"]), help="carpeta de las fotos")
    parser.add_argument("--step", type=int, default=config["frame_step"], help="una foto de cada N fotogramas")
    parser.add_argument("--min-sharpness", type=float, default=config["min_sharpness"],
                        help="descartar fotogramas con menos nitidez que esta (0 = no descartar)")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step debe ser al menos 1.")

    videos = args.videos
    if not videos:
        folder = project_path(config["videos"])
        videos = sorted(path for path in folder.glob("*") if path.suffix.lower() in VIDEO_EXTENSIONS) \
            if folder.exists() else []
    if not videos:
        raise SystemExit(f"No hay vídeos. Copia los vídeos a {config['videos']} o indícalos en la línea de órdenes.")

    args.out.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out / MAPPING_FILE
    mapping = assign_ids([video.name for video in videos], read_mapping(mapping_path))
    write_mapping(mapping_path, mapping)

    total = 0
    for video in videos:
        video_id = mapping[video.name]
        read, saved, blurry = extract_video(video, video_id, args.out, args.step, args.min_sharpness,
                                            config["jpeg_quality"])
        note = f", {blurry} descartadas por movidas" if blurry else ""
        print(f"{video_id} = {video.name}: {read} fotogramas -> {saved} fotos{note}")
        total += saved
    print(f"Total: {total} fotos en {args.out} (qué vídeo es cada vXX: {mapping_path})")


if __name__ == "__main__":
    main()
