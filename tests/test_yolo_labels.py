"""Tests de las etiquetas YOLO: coordenadas normalizadas, dentro de la imagen y coherentes con el JSON."""

import pytest

from ArtificialVision.prepare_yolo import (DOT_CLASS, NUMBER_CLASS, annotation_objects, tile_labels,
                                           tile_positions)

SIZE, OVERLAP = 1024, 160


def parse(line):
    object_class, xc, yc, w, h = line.split()
    return int(object_class), float(xc), float(yc), float(w), float(h)


@pytest.mark.parametrize("length", [500, 1024, 1025, 1800, 2263, 2900])
def test_tiles_cover_everything_with_overlap(length):
    positions = tile_positions(length, SIZE, OVERLAP)
    side = min(SIZE, length)
    assert positions[0] == 0
    assert positions[-1] + side == length       # la última acaba justo en el borde
    for a, b in zip(positions, positions[1:]):
        assert a + SIZE - b >= OVERLAP           # vecinas solapadas al menos OVERLAP píxeles


def test_whole_object_is_normalized():
    (line,) = tile_labels([(DOT_CLASS, (100, 200, 120, 220))], 0, 0, SIZE, SIZE, 0.5)
    object_class, xc, yc, w, h = parse(line)
    assert object_class == DOT_CLASS
    assert (xc, yc) == pytest.approx((110 / SIZE, 210 / SIZE), abs=1e-6)
    assert (w, h) == pytest.approx((20 / SIZE, 20 / SIZE), abs=1e-6)


def test_coordinates_are_relative_to_the_tile():
    # El mismo tipo de objeto visto desde una tesela que empieza en (900, 1000).
    (line,) = tile_labels([(NUMBER_CLASS, (1000, 1100, 1030, 1120))], 900, 1000, SIZE, SIZE, 0.5)
    _, xc, yc, _, _ = parse(line)
    assert (xc, yc) == pytest.approx((115 / SIZE, 110 / SIZE), abs=1e-6)


def test_object_cut_by_the_border():
    # Caja de 20 px de ancho con 12 px dentro de la tesela: se ve el 60 % y se etiqueta recortada.
    (line,) = tile_labels([(NUMBER_CLASS, (1012, 0, 1032, 10))], 0, 0, SIZE, SIZE, 0.5)
    _, xc, _, w, _ = parse(line)
    assert w == pytest.approx(12 / SIZE, abs=1e-6)
    assert xc + w / 2 <= 1.0 + 1e-6  # tolerancia por escribir las etiquetas con 6 decimales
    # Con solo 6 px dentro (30 %) y un mínimo del 50 %, no se etiqueta: entero estará en la tesela vecina.
    assert tile_labels([(NUMBER_CLASS, (1018, 0, 1038, 10))], 0, 0, SIZE, SIZE, 0.5) == []
    # Fuera del todo, tampoco.
    assert tile_labels([(DOT_CLASS, (2000, 0, 2010, 10))], 0, 0, SIZE, SIZE, 0.5) == []


def test_labels_of_generated_photos(generated_photos):
    """En fotos del generador: todas las etiquetas son válidas y cada objeto del JSON aparece
    entero en alguna tesela."""
    for photo, annotation, _ in generated_photos:
        objects = annotation_objects(annotation)
        photo_height, photo_width = photo.shape[:2]
        width, height = min(SIZE, photo_width), min(SIZE, photo_height)
        tiles = [(x0, y0) for y0 in tile_positions(photo_height, SIZE, OVERLAP)
                 for x0 in tile_positions(photo_width, SIZE, OVERLAP)]
        for x0, y0 in tiles:
            for line in tile_labels(objects, x0, y0, width, height, 0.3):
                object_class, xc, yc, w, h = parse(line)
                assert object_class in (DOT_CLASS, NUMBER_CLASS)
                assert w > 0 and h > 0
                assert -1e-6 <= xc - w / 2 and xc + w / 2 <= 1 + 1e-6
                assert -1e-6 <= yc - h / 2 and yc + h / 2 <= 1 + 1e-6
        for _, (a0, b0, a1, b1) in objects:
            assert any(x0 <= a0 and a1 <= x0 + width and y0 <= b0 and b1 <= y0 + height for x0, y0 in tiles)
