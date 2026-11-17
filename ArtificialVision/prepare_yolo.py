"""Convierte el dataset común (fotos + JSON) en un dataset de YOLO troceado en teselas.

Por qué teselas: Ultralytics reduce cada imagen de entrenamiento hasta que su lado mayor mide
imgsz. Las fotos de la hoja miden hasta unos 2900 px de alto, así que a 1024 los puntos se
quedarían a un tercio de su tamaño. Troceando cada foto en teselas de imgsz x imgsz, con solape,
el modelo ve los puntos a su tamaño real.

El JSON común es la única fuente de verdad: las etiquetas YOLO salen de él. Para probar otra
variante (por ejemplo, solo la clase punto) basta con rehacer este paso, sin volver a generar
las fotos.

Salida (yolo.output en config.yaml):
  images/{train,val,test}/photo_XXXXX_rF_cC.jpg   tesela de la fila F y la columna C
  labels/{train,val,test}/photo_XXXXX_rF_cC.txt   una línea por objeto: "clase xc yc ancho alto",
                                                  normalizadas entre 0 y 1
  data.yaml                                       sin "path:": Ultralytics toma como raíz la
                                                  carpeta del propio yaml
  samples/                                        algunas teselas con sus etiquetas dibujadas,
                                                  leídas de los .txt para comprobarlas

Uso, desde la carpeta del proyecto (después de python -m data.generate):
    python -m ArtificialVision.prepare_yolo           # dataset completo
    python -m ArtificialVision.prepare_yolo --quick   # dataset mini de la prueba rápida
"""

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import yaml

from common import json_format
from common.config import clear_generated_folder, load_config, project_path
from common.images import read_image, write_image

SPLITS = ("train", "val", "test")
DOT_CLASS, NUMBER_CLASS = 0, 1
CLASS_COLORS = {DOT_CLASS: (0, 170, 0), NUMBER_CLASS: (255, 90, 0)}  # BGR: verde y azul


def tile_positions(length, size, overlap):
    """Coordenadas de inicio de las teselas que cubren un lado de 'length' píxeles.

    Usa el mínimo número de teselas de lado 'size' que se solapan al menos 'overlap' píxeles,
    repartidas a distancias iguales: la primera empieza en 0 y la última acaba en el borde.
    Con un solape mayor que el objeto más grande, cada objeto cabe entero en alguna tesela.
    """
    if length <= size:
        return [0]
    n = math.ceil((length - overlap) / (size - overlap))
    return [round(i * (length - size) / (n - 1)) for i in range(n)]


def annotation_objects(annotation):
    """Lista de (clase, caja) de una anotación: los puntos con su caja y los números."""
    objects = []
    for dot in annotation["dots"]:
        if not dot.get("box"):
            raise ValueError(f"{annotation['image']}: hay puntos sin caja y YOLO la necesita para entrenar.")
        objects.append((DOT_CLASS, dot["box"]))
    for number in annotation["numbers"]:
        objects.append((NUMBER_CLASS, number["box"]))
    return objects


def tile_labels(objects, x0, y0, width, height, min_visible_fraction):
    """Líneas YOLO de los objetos que se ven en la tesela [x0, x0 + width] x [y0, y0 + height].

    La caja de un objeto cortado por el borde se recorta a la tesela; si queda menos de
    'min_visible_fraction' de su área, el objeto no se etiqueta (entero estará en la tesela vecina).
    """
    lines = []
    for object_class, (a0, b0, a1, b1) in objects:
        c0, d0 = max(a0, x0), max(b0, y0)
        c1, d1 = min(a1, x0 + width), min(b1, y0 + height)
        if c1 <= c0 or d1 <= d0:
            continue  # fuera de la tesela
        if (c1 - c0) * (d1 - d0) < min_visible_fraction * (a1 - a0) * (b1 - b0):
            continue  # se ve demasiado poco
        xc = ((c0 + c1) / 2 - x0) / width
        yc = ((d0 + d1) / 2 - y0) / height
        lines.append(f"{object_class} {xc:.6f} {yc:.6f} {(c1 - c0) / width:.6f} {(d1 - d0) / height:.6f}")
    return lines


def tile_photo(task):
    """Trabajo de cada proceso: trocea una foto y escribe sus teselas y etiquetas.

    Devuelve [teselas, puntos etiquetados, números etiquetados, teselas sin objetos].
    """
    photo_path, annotation_path, output, split, size, overlap, min_visible, jpeg_quality = task
    objects = annotation_objects(json_format.load(annotation_path))
    photo = read_image(photo_path)
    photo_height, photo_width = photo.shape[:2]
    width, height = min(size, photo_width), min(size, photo_height)
    counts = [0, 0, 0, 0]
    for row, y0 in enumerate(tile_positions(photo_height, size, overlap)):
        for col, x0 in enumerate(tile_positions(photo_width, size, overlap)):
            name = f"{photo_path.stem}_r{row}_c{col}"
            write_image(output / "images" / split / f"{name}.jpg", photo[y0:y0 + height, x0:x0 + width], jpeg_quality)
            lines = tile_labels(objects, x0, y0, width, height, min_visible)
            (output / "labels" / split / f"{name}.txt").write_text("".join(line + "\n" for line in lines),
                                                                  encoding="utf-8")
            counts[0] += 1
            counts[1] += sum(line.startswith(f"{DOT_CLASS} ") for line in lines)
            counts[2] += sum(line.startswith(f"{NUMBER_CLASS} ") for line in lines)
            counts[3] += not lines
    return counts


def write_data_yaml(output, classes):
    """data.yaml para Ultralytics, con rutas relativas a su propia carpeta."""
    header = ("# Generado por ArtificialVision/prepare_yolo.py.\n"
              "# Sin 'path:', Ultralytics toma como raíz la carpeta de este fichero, así que el\n"
              "# dataset funciona lo lances desde donde lo lances y se puede copiar tal cual (a Colab, por ejemplo).\n")
    data = {"train": "images/train", "val": "images/val", "test": "images/test",
            "names": dict(enumerate(classes))}
    (output / "data.yaml").write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")


def draw_samples(output, count):
    """Dibuja sobre algunas teselas de train las cajas leídas de sus .txt, para comprobar las etiquetas."""
    (output / "samples").mkdir()
    with_objects = [path for path in sorted((output / "labels" / "train").glob("*.txt")) if path.stat().st_size > 0]
    step = max(1, len(with_objects) // max(1, count))
    for label_path in with_objects[::step][:count]:
        tile = read_image(output / "images" / "train" / f"{label_path.stem}.jpg")
        tile_height, tile_width = tile.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            object_class, xc, yc, w, h = line.split()
            xc, w = float(xc) * tile_width, float(w) * tile_width
            yc, h = float(yc) * tile_height, float(h) * tile_height
            cv2.rectangle(tile, (round(xc - w / 2), round(yc - h / 2)), (round(xc + w / 2), round(yc + h / 2)),
                          CLASS_COLORS[int(object_class)], 2)
        write_image(output / "samples" / f"{label_path.stem}.jpg", tile, 90)


def main():
    parser = argparse.ArgumentParser(description="Trocea el dataset común en teselas con etiquetas YOLO.")
    parser.add_argument("--quick", action="store_true", help="usar el dataset mini de la prueba rápida")
    args = parser.parse_args()

    config = load_config()
    c_yolo = config["yolo"]
    source = project_path(config["generator"]["output"])
    output = project_path(c_yolo["output"])
    if args.quick:
        source = project_path(config["quick"]["data_output"])
        output = project_path(config["quick"]["yolo_output"])
    if not (source / "config_used.yaml").exists():
        raise SystemExit(f"No hay dataset en {source}. Ejecuta antes: python -m data.generate"
                         + (" --quick" if args.quick else ""))

    size = config["training"]["imgsz"]  # las teselas miden lo mismo que la entrada de YOLO
    clear_generated_folder(output, "data.yaml")
    # data.yaml se escribe lo primero: si el proceso se corta, la carpeta ya lleva la marca y la
    # siguiente ejecución puede borrarla.
    output.mkdir(parents=True)
    write_data_yaml(output, c_yolo["classes"])
    tasks = []
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        for annotation_path in sorted((source / split / "annotations").glob("*.json")):
            photo_path = source / split / "images" / f"{annotation_path.stem}.jpg"
            tasks.append((photo_path, annotation_path, output, split, size, c_yolo["overlap"],
                          c_yolo["min_visible_fraction"], c_yolo["jpeg_quality"]))

    print(f"Troceando {len(tasks)} fotos en teselas de {size} px (solape {c_yolo['overlap']} px)...")
    start = time.time()
    totals = {split: [0, 0, 0, 0] for split in SPLITS}
    with ProcessPoolExecutor(max_workers=config["generator"]["processes"]) as executor:
        for task, counts in zip(tasks, executor.map(tile_photo, tasks, chunksize=4)):
            totals[task[3]] = [a + b for a, b in zip(totals[task[3]], counts)]

    draw_samples(output, c_yolo["samples_with_boxes"])
    for split, (tiles, dots, numbers, empty) in totals.items():
        print(f"  {split:5s}: {tiles} teselas ({empty} sin objetos), {dots} puntos y {numbers} números etiquetados")
    print(f"Hecho en {time.time() - start:.0f} s. Dataset en {output}")


if __name__ == "__main__":
    main()
