"""Tests de la inferencia de visión artificial (ArtificialVision/pipeline.py)."""

import numpy as np
import pytest

from ArtificialVision.pipeline import (collect_detections, digits_value, erase_dots, polygon_box, tile_cores,
                                       tile_grid, working_image)
from common import json_format
from common.config import load_config, project_path
from common.evaluation import evaluate_image, summarize

SIZE, OVERLAP = 1024, 160


def test_working_image_only_shrinks_big_photos():
    big, scale = working_image(np.zeros((4000, 3000, 3), np.uint8), 2900)
    assert max(big.shape[:2]) == 2900 and scale == pytest.approx(2900 / 4000)
    small, scale = working_image(np.zeros((1500, 1000, 3), np.uint8), 2900)
    assert small.shape[:2] == (1500, 1000) and scale == 1.0


@pytest.mark.parametrize("length", [500, 1024, 1800, 2263, 2900])
def test_tile_cores_split_the_side_without_gaps(length):
    cores = tile_cores(length, SIZE, OVERLAP)
    assert cores[0][1] == 0 and cores[-1][2] == length       # empiezan en 0 y acaban en el borde
    for (_, _, end), (_, start, _) in zip(cores, cores[1:]):
        assert end == start                                  # sin huecos ni solapes entre zonas propias
    for tile_start, core_start, core_end in cores:
        assert tile_start <= core_start < core_end <= tile_start + SIZE
        # Las fronteras interiores quedan a la mitad del solape: un objeto de menos de OVERLAP / 2
        # de radio cuyo centro esté en la zona propia se ve entero en su tesela.
        if core_start > 0:
            assert core_start - tile_start >= OVERLAP / 2
        if core_end < length:
            assert tile_start + SIZE - core_end >= OVERLAP / 2


def test_object_in_the_overlap_is_kept_once():
    tiles = tile_grid(1800, 900, SIZE, OVERLAP)  # dos teselas en horizontal: x0 = 0 y x0 = 776
    assert [tile[0] for tile in tiles] == [0, 776]
    dots, numbers = [], []
    # El mismo punto, con centro en x = 850 de la imagen, lo detectan las dos teselas.
    collect_detections(tiles[0], [[845, 100, 855, 110]], [0], [0.9], dots, numbers)
    collect_detections(tiles[1], [[845 - 776, 100, 855 - 776, 110]], [0], [0.9], dots, numbers)
    assert len(dots) == 1 and dots[0]["box"] == [845, 100, 855, 110]
    # Un número de la segunda tesela, en coordenadas de la imagen.
    collect_detections(tiles[1], [[500, 50, 520, 60]], [1], [0.8], dots, numbers)
    assert numbers == [{"box": [1276.0, 50.0, 1296.0, 60.0], "confidence": 0.8}]


def test_erase_dots_paints_them_with_the_paper_color():
    image = np.full((100, 100, 3), 160, np.uint8)  # papel gris, como con poca luz
    image[45:55, 45:55] = 0                        # un "punto" negro de 10 px
    erased = erase_dots(image, [{"x": 50.0, "y": 50.0, "box": [45, 45, 55, 55]}], margin=0.3)
    assert (erased[45:55, 45:55] == 160).all()     # el punto ha desaparecido y no queda una mancha blanca
    assert image[50, 50].tolist() == [0, 0, 0]     # la foto original no se modifica


def test_digits_value_and_polygon_box():
    assert digits_value("12") == 12 and digits_value(" 7 ") == 7
    assert digits_value("") is None and digits_value("1a") is None
    assert polygon_box([[10, 20], [30, 20], [30, 28], [10, 28]]) == [10, 20, 30, 28]


# -----------------------------------------------------------------------------
# Integración: la tubería completa sobre fotos generadas (necesita el modelo entrenado)
# -----------------------------------------------------------------------------
CONFIG = load_config()
MODEL_EXISTS = project_path(CONFIG["inference"]["model"]).exists()


@pytest.fixture(scope="module")
def pipeline():
    from ArtificialVision.pipeline import Pipeline
    return Pipeline(CONFIG)


@pytest.mark.skipif(not MODEL_EXISTS, reason="no hay modelo entrenado (python -m ArtificialVision.train)")
@pytest.mark.parametrize("mode, min_pairs", [("crops", 90), ("page", 60)])
def test_pipeline_on_generated_photos(pipeline, generated_photos, mode, min_pairs):
    counts = []
    for photo, annotation, _ in generated_photos[:2]:
        result = pipeline.process(photo, annotation["image"], mode)
        json_format.validate(result)
        stages = {"detection", "erase_dots", "ocr", "matching", "total"}
        if CONFIG.get("aruco", {}).get("enabled"):
            stages.add("aruco")
            assert result["metadata"]["aruco"]["found"] is False  # las fotos sintéticas no llevan la hoja
        assert set(result["times_ms"]) == stages
        assert all("confidence" in item for item in result["dots"] + result["numbers"])
        counts.append(evaluate_image(annotation, result, CONFIG["evaluation"]))
    summary = summarize(counts)
    assert summary["dot_recall"] >= 95 and summary["dot_precision"] >= 95
    assert summary["pairs_correct"] >= min_pairs
