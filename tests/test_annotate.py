"""Tests de la lógica de la herramienta de anotación (sin abrir ventanas)."""

from common import json_format
from tools.annotate import AnnotationState


def test_clicks_digits_and_backspace():
    state = AnnotationState()
    state.add_dot(10.5, 20.5)
    for digit in "123":
        state.type_digit(digit)
    state.backspace()                    # borra el 3
    state.add_dot(40.5, 50.5)
    state.type_digit("7")
    state.backspace()                    # borra el 7
    state.backspace()                    # sin cifras: borra el segundo punto
    assert state.dots == [[10.5, 20.5, "12"]]


def test_annotation_in_common_format():
    state = AnnotationState()
    state.add_dot(10.5, 20.5)
    state.type_digit("4")
    state.add_dot(30.5, 40.5)            # punto sin número
    annotation = state.to_annotation("foto.jpg", 640, 480)
    json_format.validate(annotation)
    assert annotation["dots"] == [{"id": 0, "x": 10.5, "y": 20.5}, {"id": 1, "x": 30.5, "y": 40.5}]
    assert annotation["numbers"] == [{"id": 0, "value": 4}]
    assert annotation["pairs"] == [{"dot": 0, "number": 0, "distance": None}]
    assert annotation["unpaired_dots"] == [1]


def test_saved_annotation_can_be_reopened():
    state = AnnotationState()
    state.add_dot(1.5, 2.5)
    state.type_digit("9")
    state.add_dot(3.5, 4.5)
    reopened = AnnotationState.from_annotation(state.to_annotation("foto.jpg", 10, 10))
    assert reopened.dots == state.dots
