"""Tests del enfoque de visión clásica (ClassicVision/)."""

import cv2
import numpy as np
import pytest

from ClassicVision.digits import Char, build_templates, group_numbers, recognize, render_digit
from ClassicVision.dots import detect_dots, looks_like_dot, shape_features
from ClassicVision.preprocess import darkness, ink_mask, paper_mask
from common import json_format
from common.config import load_config
from common.evaluation import evaluate_image, summarize

CONFIG = load_config()
CFG = CONFIG["classic"]
HERSHEY = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX)  # existen en OpenCV 4 y 5


def blob(draw, size=60):
    """Máscara (uint8 0/255) con lo que dibuje 'draw' sobre un lienzo negro."""
    mask = np.zeros((size, size), np.uint8)
    draw(mask)
    return mask


# -----------------------------------------------------------------------------
# Hoja y tinta
# -----------------------------------------------------------------------------
def test_ink_survives_uneven_lighting():
    # Foto de tamaño real (el núcleo del fondo es el 3 % del lado menor: tiene que ser mayor que
    # el punto). Papel con un degradado fuerte, de 250 a 140, y en la zona oscura un punto gris.
    gradient = np.linspace(250, 140, 1200, dtype=np.float32)[None, :].repeat(1000, axis=0)
    image = cv2.merge([gradient] * 3).astype(np.uint8)
    cv2.circle(image, (1100, 500), 8, (70, 70, 70), -1)
    paper = np.full(image.shape[:2], 255, np.uint8)
    ink = ink_mask(darkness(image, CFG), paper, CFG)
    assert ink[500, 1100] == 255                    # el punto es tinta
    assert cv2.countNonZero(ink) < 2 * np.pi * 10 ** 2   # y el papel en sombra no


def test_paper_is_separated_from_a_wooden_table():
    image = np.full((800, 600, 3), (60, 110, 160), np.uint8)   # mesa marrón (BGR)
    image[100:700, 100:500] = 240                              # hoja blanca
    mask, polygon = paper_mask(image, CFG)
    assert mask[400, 300] == 255 and mask[50, 50] == 0
    assert polygon is not None and len(polygon) == 4             # la hoja es un cuadrilátero


# -----------------------------------------------------------------------------
# Puntos
# -----------------------------------------------------------------------------
def test_dot_shape_rules():
    filled = shape_features(blob(lambda m: cv2.circle(m, (30, 30), 10, 255, -1)))
    hollow = shape_features(blob(lambda m: cv2.circle(m, (30, 30), 15, 255, 3)))
    one = shape_features(blob(lambda m: cv2.putText(m, "1", (15, 45), HERSHEY[0], 1.4, 255, 4)))
    eight = shape_features(blob(lambda m: cv2.putText(m, "8", (10, 50), HERSHEY[0], 1.6, 255, 4)))
    assert looks_like_dot(filled, CFG)
    assert hollow["holes"] == 1 and not looks_like_dot(hollow, CFG)   # circunferencia: tiene agujero
    assert not looks_like_dot(one, CFG)                                # alargado
    assert not looks_like_dot(eight, CFG)


def test_dot_touching_its_number_is_separated():
    ink = np.zeros((200, 400), np.uint8)
    for x in (60, 200):                                   # dos puntos sueltos, para el tamaño típico
        cv2.circle(ink, (x, 60), 9, 255, -1)
    cv2.circle(ink, (300, 120), 9, 255, -1)               # y otro pegado a su número
    cv2.putText(ink, "47", (308, 132), HERSHEY[0], 0.9, 255, 2)
    dots, text_ink, radius = detect_dots(ink, CFG)
    centers = sorted((round(d["x"]), round(d["y"])) for d in dots)
    assert len(dots) == 3 and radius == pytest.approx(9.5, abs=1)
    assert any(abs(x - 300.5) <= 1 and abs(y - 120.5) <= 1 for x, y in centers)
    assert cv2.countNonZero(text_ink[105:140, 315:370]) > 100   # el número sigue en la tinta


# -----------------------------------------------------------------------------
# Números
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("font", HERSHEY)
def test_templates_read_small_digits(font):
    templates = build_templates(HERSHEY)
    for digit in "0123456789":
        big = render_digit(digit, font, size=64)
        x, y, w, h = cv2.boundingRect(big)
        small = cv2.resize(big[y:y + h, x:x + w], None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        assert recognize(small.astype(np.float32), templates)[0] == digit


def test_characters_are_grouped_by_digit_pitch():
    def char(x0, y0=100, w=12, h=20):
        return Char(x0, y0, x0 + w, y0 + h, np.full((h, w), 255, np.uint8))
    # "12" (paso de 15 px), "3" suelto a 40 px y "45" en otra línea
    chars = [char(100), char(115), char(155), char(300, 200), char(315, 200)]
    groups = group_numbers(chars, 20.0, CFG)
    assert sorted(len(g) for g in groups) == [1, 2, 2]
    assert sorted([c.x0 for c in g] for g in groups) == [[100, 115], [155], [300, 315]]


# -----------------------------------------------------------------------------
# Integración: la tubería completa sobre fotos generadas (necesita OpenCV 5 para el generador)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not hasattr(cv2, "FontFace"), reason="el generador necesita cv2.FontFace (OpenCV 5)")
def test_pipeline_on_generated_photos(generated_photos):
    from ClassicVision.pipeline import Pipeline
    pipeline = Pipeline(CONFIG)
    counts = []
    for photo, annotation, _ in generated_photos:
        result = pipeline.process(photo, annotation["image"])
        json_format.validate(result)
        assert {"paper", "ink", "dots", "ocr", "matching", "total"} <= set(result["times_ms"])
        counts.append(evaluate_image(annotation, result, CONFIG["evaluation"]))
    summary = summarize(counts)
    assert summary["dot_recall"] >= 95 and summary["dot_precision"] >= 95
    assert summary["pairs_correct"] >= 85
