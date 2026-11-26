"""Tests de la herramienta de etiquetado con SAM (sin abrir ventanas ni cargar SAM)."""

import pytest

from common import json_format
from tools.annotate_sam import LabelState, View, crop_window, plausible_box, polygon_box


# ---------- geometría ----------
def test_view_round_trip_and_fit():
    view = View(1920, 1080, window=(960, 540))
    assert view.scale() == pytest.approx(0.5)               # zoom 1: la foto entera cabe en la ventana
    assert view.to_window(*view.to_image(123.5, 77.5)) == pytest.approx((123.5, 77.5))
    assert view.to_window(0, 0) == pytest.approx((0, 0))
    assert view.to_window(1920, 1080) == pytest.approx((960, 540))


def test_zoom_keeps_the_point_under_the_mouse():
    view = View(1920, 1080, window=(960, 540))
    before = view.to_image(300.5, 200.5)
    view.zoom_at(300.5, 200.5, 4.0)
    assert view.zoom == 4.0
    assert view.to_image(300.5, 200.5) == pytest.approx(before)
    view.zoom_at(300.5, 200.5, 0.01)                         # no se puede alejar más que la foto entera
    assert view.zoom == 1.0


def test_warp_matrix_matches_to_window():
    """La matriz de cv2.warpAffine lleva el centro de cada píxel (x - 0.5) a su sitio en la ventana."""
    view = View(800, 600, window=(400, 300))
    view.zoom_at(100.5, 80.5, 3.0)
    matrix = view.matrix()
    x, y = 250.5, 170.5                                      # centro de un píxel de la foto
    wx = matrix[0, 0] * (x - 0.5) + matrix[0, 2] + 0.5
    wy = matrix[1, 1] * (y - 0.5) + matrix[1, 2] + 0.5
    assert (wx, wy) == pytest.approx(view.to_window(x, y))


def test_crop_window_is_centred_and_stays_inside():
    assert crop_window([(500, 400)], [1], None, 256, 1920, 1080) == (372, 272, 628, 528)
    assert crop_window([(10, 1070)], [1], None, 256, 1920, 1080) == (0, 824, 256, 1080)   # junto a una esquina
    assert crop_window([(50, 50)], [1], None, 256, 200, 150) == (0, 0, 200, 150)          # foto más pequeña
    # Caja dibujada mayor que el recorte: el recorte crece hasta 1,5 veces la caja
    assert crop_window([], [], [800, 500, 1000, 560], 256, 1920, 1080) == (750, 380, 1050, 680)


def test_polygon_box_adds_offset_and_last_pixel():
    assert polygon_box([[2, 3], [6, 3], [6, 9], [2, 9]], 100, 200) == [102, 203, 107, 210]


def test_implausible_boxes_are_rejected():
    assert plausible_box([10, 10, 20, 20], 60, [(15, 15)], [1])
    assert not plausible_box(None, 60)
    assert not plausible_box([0, 0, 300, 20], 60)                   # demasiado grande
    assert not plausible_box([10, 10, 20, 20], 60, [(40, 40)], [1])  # deja fuera el clic "sí"
    assert plausible_box([10, 10, 20, 20], 60, [(40, 40)], [0])      # un clic "no" puede quedar fuera


# ---------- estado de una foto ----------
def label_pair(state, dot, number, text):
    """Simula lo que hace el usuario: propuesta del punto, ESPACIO, propuesta del número, cifras, Intro."""
    state.proposal = dot
    state.confirm()
    state.proposal = number
    for digit in text:
        state.type_digit(digit)
    return state.accept()


def test_pair_workflow_gives_valid_annotation():
    state = LabelState()
    assert label_pair(state, [10, 10, 16, 16], [20, 8, 30, 18], "12")
    state.proposal = [50, 50, 56, 56]                        # punto sin número
    state.accept()
    annotation = state.to_annotation("v01_00000.jpg", 100, 80)
    json_format.validate(annotation)
    assert annotation["dots"] == [{"id": 0, "x": 13.0, "y": 13.0, "box": [10, 10, 16, 16]},
                                  {"id": 1, "x": 53.0, "y": 53.0, "box": [50, 50, 56, 56]}]
    assert annotation["numbers"] == [{"id": 0, "value": 12, "box": [20, 8, 30, 18]}]
    assert annotation["pairs"] == [{"dot": 0, "number": 0, "distance": 7.0}]
    assert annotation["unpaired_dots"] == [1]
    assert annotation["metadata"]["prelabel"] is False


def test_confirm_moves_from_dot_to_number_and_backspace():
    state = LabelState()
    state.add_point(5, 5, True)
    state.proposal = [3, 3, 7, 7]
    state.confirm()
    assert state.active == "number" and state.points == [] and state.current["dot"] == [3, 3, 7, 7]
    state.type_digit("4")
    state.backspace()
    assert state.current["text"] == ""
    state.add_point(9, 9, True)
    state.backspace()                                        # sin cifras: borra los clics
    assert state.points == []


def test_number_without_value_is_kept_as_null():
    state = LabelState()
    label_pair(state, [10, 10, 16, 16], [20, 8, 30, 18], "")
    assert state.missing_values() == 1
    assert state.to_annotation("f.jpg", 100, 80)["numbers"][0]["value"] is None


def test_edit_and_cancel_restores_the_pair():
    state = LabelState()
    label_pair(state, [10, 10, 16, 16], [20, 8, 30, 18], "1")
    label_pair(state, [50, 50, 56, 56], [60, 48, 70, 58], "2")
    assert state.edit_at(65, 52)                             # clic sobre el número 2
    assert state.current["text"] == "2" and len(state.entries) == 1
    state.type_digit("0")
    state.cancel()                                           # se recupera tal como estaba
    assert [entry["text"] for entry in state.entries] == ["1", "2"]


def test_edit_and_accept_replaces_the_pair_in_place():
    state = LabelState()
    label_pair(state, [10, 10, 16, 16], [20, 8, 30, 18], "1")
    label_pair(state, [50, 50, 56, 56], [60, 48, 70, 58], "7")
    state.edit_at(13, 13)
    state.backspace()                                        # "1" -> ""
    state.type_digit("3")
    state.accept()
    assert [entry["text"] for entry in state.entries] == ["3", "7"]


def test_delete_and_undo():
    state = LabelState()
    label_pair(state, [10, 10, 16, 16], [20, 8, 30, 18], "1")
    label_pair(state, [50, 50, 56, 56], [60, 48, 70, 58], "2")
    assert not state.delete_at(90, 90)                       # ahí no hay nada
    assert state.delete_at(12, 12)
    state.undo()
    assert state.entries == []


def test_pipeline_result_is_loaded_as_prelabel_and_round_trips():
    """Un resultado del pipeline (con confianzas, tiempos y un valor null) se carga para corregirlo."""
    result = json_format.create(
        "v01_00030.jpg", 100, 80,
        dots=[{"id": 0, "x": 13, "y": 13, "box": [10, 10, 16, 16], "confidence": 0.9},
              {"id": 1, "x": 53, "y": 53, "box": [50, 50, 56, 56], "confidence": 0.8}],
        numbers=[{"id": 0, "value": 5, "box": [20, 8, 30, 18], "confidence": 0.7},
                 {"id": 1, "value": None, "box": [80, 60, 90, 70], "confidence": 0.2}],
        pairs=[{"dot": 0, "number": 0, "distance": 7.0}], unpaired_dots=[1], unpaired_numbers=[1],
        times_ms={"total": 50.0})
    state = LabelState.from_annotation(result, prelabeled=True)
    assert len(state.entries) == 3 and state.prelabeled
    annotation = state.to_annotation("v01_00030.jpg", 100, 80)
    assert annotation["metadata"]["prelabel"] is True
    assert "times_ms" not in annotation and "confidence" not in annotation["dots"][0]
    reopened = LabelState.from_annotation(annotation)
    assert reopened.prelabeled and reopened.entries == state.entries


def test_annotations_without_boxes_are_refused():
    minimal = json_format.create("f.jpg", 10, 10, [{"id": 0, "x": 1, "y": 1}], [], [], unpaired_dots=[0])
    with pytest.raises(ValueError):
        LabelState.from_annotation(minimal)
