"""Tests del generador sintético y del formato común de las anotaciones."""

import numpy as np
import pytest

from common import json_format
from common.matching import point_box_distance
from data.generate import generate_photo, shape_positions, text_mask, transform_points


def test_annotation_is_valid(generated_photos):
    for photo, annotation, _ in generated_photos:
        json_format.validate(annotation)  # lanza ValueError si algo está mal
        assert (annotation["height"], annotation["width"]) == photo.shape[:2]
        assert len(annotation["dots"]) > 0
        assert len(annotation["dots"]) == len(annotation["numbers"]) == len(annotation["pairs"])
        assert annotation["unpaired_dots"] == [] and annotation["unpaired_numbers"] == []


def test_numbers_form_sequence_1_to_n(generated_photos):
    for _, annotation, _ in generated_photos:
        values = sorted(number["value"] for number in annotation["numbers"])
        assert values == list(range(1, len(values) + 1))


def test_boxes_inside_image_and_dots_inside_their_box(generated_photos):
    for _, annotation, _ in generated_photos:
        width, height = annotation["width"], annotation["height"]
        for item in annotation["dots"] + annotation["numbers"]:
            x0, y0, x1, y1 = item["box"]
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        for dot in annotation["dots"]:
            x0, y0, x1, y1 = dot["box"]
            assert x0 <= dot["x"] <= x1 and y0 <= dot["y"] <= y1


def test_ground_truth_is_unambiguous(generated_photos):
    """Cada número está más cerca de su punto que de cualquier otro, y cada punto de su número."""
    for _, annotation, _ in generated_photos:
        xs = np.array([dot["x"] for dot in annotation["dots"]])
        ys = np.array([dot["y"] for dot in annotation["dots"]])
        boxes = np.array([number["box"] for number in annotation["numbers"]])
        # distances[i, j] = distancia del punto i a la caja del número j
        distances = point_box_distance(xs[:, None], ys[:, None], boxes[None, :, :])
        for pair in annotation["pairs"]:
            i, j = pair["dot"], pair["number"]
            assert distances[i, j] == pytest.approx(pair["distance"], abs=0.01)
            assert np.argmin(distances[i, :]) == j  # el número más cercano al punto es el suyo
            assert np.argmin(distances[:, j]) == i  # el punto más cercano al número es el suyo


def test_generation_is_reproducible(generator_config, fonts):
    photo_a, annotation_a, _ = generate_photo(generator_config, "val", 7, fonts)
    photo_b, annotation_b, _ = generate_photo(generator_config, "val", 7, fonts)
    assert annotation_a == annotation_b
    assert np.array_equal(photo_a, photo_b)


def test_text_mask_is_cropped_to_the_ink():
    mask = text_mask("47", "sans", 40)  # "sans" es la fuente que trae OpenCV: siempre está
    assert mask.max() == 255
    # Recortada a la tinta: hay tinta en la primera y la última fila y columna.
    assert mask[0].max() > 0 and mask[-1].max() > 0
    assert mask[:, 0].max() > 0 and mask[:, -1].max() > 0
    assert 20 <= mask.shape[0] <= 40  # los dígitos miden algo menos que el tamaño de letra


def test_transform_points_with_identity():
    points = [(10.0, 20.0), (300.5, 7.25)]
    assert np.allclose(transform_points(np.eye(3), points), points)


def test_shape_inside_area():
    rng = np.random.default_rng(0)
    area = (100, 200, 900, 1300)
    positions = np.array(shape_positions(rng, 50, area))
    assert len(positions) == 50
    assert np.all(positions[:, 0] >= area[0]) and np.all(positions[:, 0] <= area[2])
    assert np.all(positions[:, 1] >= area[1]) and np.all(positions[:, 1] <= area[3])
