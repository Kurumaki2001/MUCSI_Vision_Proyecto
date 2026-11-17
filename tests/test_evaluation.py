"""Tests de la evaluación: una predicción perfecta da el 100 % y cada tipo de error se nota donde debe."""

import copy

import pytest

from common import json_format
from common.evaluation import box_overlap, evaluate_folder, evaluate_image, summarize

CFG = {"dot_tolerance_px": 8, "number_overlap": 0.5}


def perfect_result(annotation):
    """Un resultado idéntico a la anotación, con las confianzas y tiempos que añade un sistema real."""
    result = copy.deepcopy(annotation)
    for item in result["dots"] + result["numbers"]:
        item["confidence"] = 0.99
    result["times_ms"] = {"total": 100.0}
    return result


def evaluate(annotation, result):
    return summarize([evaluate_image(annotation, result, CFG)])


def test_box_overlap():
    assert box_overlap([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == pytest.approx(1)
    assert box_overlap([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(0.5)
    assert box_overlap([[0, 0, 10, 10]], [[2, 2, 4, 4]])[0, 0] == pytest.approx(1)  # una dentro de la otra
    assert box_overlap([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0


def test_loose_box_with_right_value_counts_as_read():
    """Una caja algo más holgada que la real (como las de EasyOCR) no penaliza una lectura correcta."""
    annotation = json_format.create("a.jpg", 100, 100, [], [{"id": 0, "value": 12, "box": [40, 40, 50, 46]}], [])
    result = json_format.create("a.jpg", 100, 100, [], [{"id": 0, "value": 12, "box": [37, 37, 54, 50]}], [])
    assert evaluate(annotation, result)["numbers_read"] == 100


def test_perfect_prediction_gives_100(generated_photos):
    for _, annotation, _ in generated_photos:
        summary = evaluate(annotation, perfect_result(annotation))
        for metric in ("dot_precision", "dot_recall", "dot_f1", "numbers_read", "pairs_correct", "perfect_images"):
            assert summary[metric] == 100, metric
        assert summary["mean_time_ms"] == 100


def test_missing_and_extra_dots(generated_photos):
    annotation = generated_photos[0][1]
    result = perfect_result(annotation)
    lost = result["dots"].pop(0)                                           # un punto sin detectar
    result["pairs"] = [pair for pair in result["pairs"] if pair["dot"] != lost["id"]]
    result["dots"].append({"id": 9999, "x": 1.0, "y": 1.0, "confidence": 0.5})  # uno sobrante
    summary = evaluate(annotation, result)
    n = len(annotation["dots"])
    assert summary["dot_recall"] == pytest.approx(100 * (n - 1) / n, abs=0.01)
    assert summary["dot_precision"] == pytest.approx(100 * (n - 1) / n, abs=0.01)
    assert summary["pairs_correct"] == pytest.approx(100 * (n - 1) / n, abs=0.01)  # el punto perdido cuenta como fallo
    assert summary["perfect_images"] == 0


def test_misread_number(generated_photos):
    annotation = generated_photos[0][1]
    result = perfect_result(annotation)
    result["numbers"][0]["value"] = None  # el OCR no ha podido leerlo
    result["numbers"][1]["value"] += 1    # el OCR lo ha leído mal
    summary = evaluate(annotation, result)
    n = len(annotation["numbers"])
    assert summary["numbers_read"] == pytest.approx(100 * (n - 2) / n, abs=0.01)
    assert summary["pairs_correct"] == pytest.approx(100 * (n - 2) / n, abs=0.01)
    assert summary["dot_recall"] == 100 and summary["perfect_images"] == 0


def test_dot_tolerance():
    annotation = json_format.create("a.jpg", 100, 100, [{"id": 0, "x": 50, "y": 50}], [], [])
    near = json_format.create("a.jpg", 100, 100, [{"id": 0, "x": 55, "y": 50}], [], [])
    far = json_format.create("a.jpg", 100, 100, [{"id": 0, "x": 59, "y": 50}], [], [])
    assert evaluate(annotation, near)["dot_recall"] == 100  # a 5 px: dentro de la tolerancia de 8
    assert evaluate(annotation, far)["dot_recall"] == 0     # a 9 px: fuera


def test_minimal_real_annotation_without_number_boxes():
    """Una anotación real mínima (sin cajas de números) permite evaluar puntos y parejas, pero no la lectura."""
    annotation = json_format.create("a.jpg", 100, 100, [{"id": 0, "x": 10, "y": 10}, {"id": 1, "x": 60, "y": 60}],
                                    [{"id": 0, "value": 1}, {"id": 1, "value": 2}],
                                    [{"dot": 0, "number": 0, "distance": None}, {"dot": 1, "number": 1, "distance": None}])
    result = json_format.create("a.jpg", 100, 100, [{"id": 5, "x": 11, "y": 10}, {"id": 6, "x": 60, "y": 61}],
                                [{"id": 0, "value": 1, "box": [14, 6, 20, 14]}, {"id": 1, "value": 3, "box": [64, 56, 70, 64]}],
                                [{"dot": 5, "number": 0, "distance": 3}, {"dot": 6, "number": 1, "distance": 4}])
    summary = evaluate(annotation, result)
    assert summary["numbers_read"] is None  # sin cajas en la anotación no se puede medir
    assert summary["dot_recall"] == 100
    assert summary["pairs_correct"] == 50   # el segundo punto tiene asignado un 3 en vez de un 2


def test_evaluate_folder_counts_missing_results_as_empty(tmp_path, generated_photos):
    annotations, results = tmp_path / "annotations", tmp_path / "results"
    annotations.mkdir()
    results.mkdir()
    for k, (_, annotation, _) in enumerate(generated_photos[:2]):
        json_format.save(annotation, annotations / f"photo_{k}.json")
    json_format.save(perfect_result(generated_photos[0][1]), results / "photo_0.json")  # falta photo_1.json
    summary, per_image = evaluate_folder(annotations, results, CFG)
    assert summary["missing_results"] == 1
    assert [image["perfect"] for image in per_image] == [True, False]
    assert summary["perfect_images"] == 50
