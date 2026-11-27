"""Tests del emparejamiento de puntos y números (algoritmo húngaro con distancia punto-rectángulo)."""

import pytest

from common import matching
from common.matching import assign, match_dots_numbers, max_pair_distance, point_box_distance, sequence_check


def dot(i, x, y):
    return {"id": i, "x": x, "y": y}


def number(i, box, value=None):
    return {"id": i, "box": box, "value": value}


def as_set(pairs):
    return {(pair["dot"], pair["number"]) for pair in pairs}


def test_point_box_distance():
    box = [10, 10, 20, 20]
    assert point_box_distance(15, 15, box) == 0          # dentro
    assert point_box_distance(5, 15, box) == 5           # a la izquierda
    assert point_box_distance(23, 24, box) == pytest.approx(5)  # en diagonal desde la esquina (3, 4, 5)


def test_each_dot_with_its_number():
    dots = [dot(0, 0, 0), dot(1, 100, 0), dot(2, 0, 100)]
    numbers = [number(10, [3, -4, 9, 4], 1), number(11, [103, -4, 109, 4], 2), number(12, [3, 96, 9, 104], 3)]
    pairs, unpaired_dots, unpaired_numbers = match_dots_numbers(dots, numbers, max_distance=30)
    assert as_set(pairs) == {(0, 10), (1, 11), (2, 12)}
    assert unpaired_dots == [] and unpaired_numbers == []
    assert all(pair["distance"] == 3 for pair in pairs)


def test_hungarian_solves_case_where_greedy_fails():
    # El número más cercano al punto A es el de B (a 4 px), pero el de A está a 6 px. Uno voraz
    # daría a A el número de B y B se quedaría sin el suyo. El húngaro minimiza la suma total:
    # A-propio (6) + B-propio (2) = 8, frente a A-ajeno (4) + B-ajeno (16) = 20.
    dots = [dot(0, 0, 0), dot(1, 10, 0)]
    numbers = [number(0, [-10, -2, -6, 2], 1), number(1, [4, -2, 8, 2], 2)]
    pairs, _, _ = match_dots_numbers(dots, numbers, max_distance=30)
    assert as_set(pairs) == {(0, 0), (1, 1)}


def test_distance_to_box_and_not_to_center():
    # El "118" pegado al punto es ancho: su centro está a 22 px, pero su borde a 2 px. El "7" de
    # un punto que no se ha detectado tiene el centro más cerca (a 9 px), pero el borde más lejos
    # (a 6 px). Con la distancia punto-rectángulo, el punto se queda con su número.
    dots = [dot(0, 0, 0)]
    numbers = [number(0, [2, -5, 42, 5], 118), number(1, [-12, -3, -6, 3], 7)]
    pairs, _, unpaired_numbers = match_dots_numbers(dots, numbers, max_distance=30)
    assert as_set(pairs) == {(0, 0)}
    assert unpaired_numbers == [1]


def test_unread_number_still_protects_its_dot():
    # El número de A no se ha podido leer (value None). Si se quitara, A se quedaría con el
    # número de B (a 3 px de A y a 5 px de B). Manteniéndolo, cada punto se queda con el suyo.
    dots = [dot(0, 0, 0), dot(1, 12, 0)]
    numbers = [number(0, [-6, -2, -2, 2], None), number(1, [3, -2, 7, 2], 8)]
    pairs, _, _ = match_dots_numbers(dots, numbers, max_distance=30)
    assert as_set(pairs) == {(0, 0), (1, 1)}
    # Sin el número ilegible, el punto A le quitaría el número a B:
    pairs, unpaired_dots, _ = match_dots_numbers(dots, numbers[1:], max_distance=30)
    assert as_set(pairs) == {(0, 1)} and unpaired_dots == [1]


def test_no_pair_beyond_max_distance():
    dots = [dot(0, 0, 0), dot(1, 100, 100)]
    numbers = [number(0, [3, -2, 7, 2], 1), number(1, [200, 200, 205, 205], 2)]
    pairs, unpaired_dots, unpaired_numbers = match_dots_numbers(dots, numbers, max_distance=30)
    assert as_set(pairs) == {(0, 0)}
    assert unpaired_dots == [1] and unpaired_numbers == [1]


def test_more_dots_than_numbers_and_vice_versa():
    dots = [dot(0, 0, 0), dot(1, 50, 0), dot(2, 100, 0)]
    numbers = [number(0, [53, -2, 57, 2], 5)]
    pairs, unpaired_dots, unpaired_numbers = match_dots_numbers(dots, numbers, max_distance=30)
    assert as_set(pairs) == {(1, 0)} and unpaired_dots == [0, 2] and unpaired_numbers == []

    pairs, unpaired_dots, unpaired_numbers = match_dots_numbers(dots[1:2], [numbers[0], number(1, [0, 0, 4, 4])], 30)
    assert as_set(pairs) == {(1, 0)} and unpaired_dots == [] and unpaired_numbers == [1]


def test_empty_inputs_do_not_fail():
    assert match_dots_numbers([], [], 30) == ([], [], [])
    assert match_dots_numbers([dot(0, 0, 0)], [], 30) == ([], [0], [])
    assert match_dots_numbers([], [number(0, [0, 0, 4, 4])], 30) == ([], [], [0])
    assert assign([[]], 1) == []


def test_max_pair_distance_uses_median_height():
    numbers = [number(0, [0, 0, 5, 10]), number(1, [0, 0, 5, 12]), number(2, [0, 0, 5, 30])]
    assert max_pair_distance(numbers, 3.0) == 36  # mediana 12 x 3
    assert max_pair_distance([], 3.0) is None


def test_sequence_check():
    assert sequence_check([1, 2, 2, 4, None]) == {"n": 4, "missing": [3], "repeated": [2], "out_of_range": []}
    assert sequence_check([3, 1, 2]) == {"n": 3, "missing": [], "repeated": [], "out_of_range": []}
    # Un número mal leído no estira la secuencia: queda fuera de rango.
    assert sequence_check([1, 2, 3, 4, 140]) == {"n": 4, "missing": [], "repeated": [], "out_of_range": [140]}
    assert sequence_check([87, 90, 120]) is None  # un trozo de ficha: no es una secuencia 1..N
    assert sequence_check([None]) is None


def test_sequence_check_ignores_a_very_large_ocr_value(monkeypatch):
    def short_range(*args):
        assert max(args) <= 5
        return range(*args)

    monkeypatch.setattr(matching, "range", short_range, raising=False)
    outlier = 10**400
    assert sequence_check([1, 2, 4, outlier]) == {
        "n": 4, "missing": [3], "repeated": [], "out_of_range": [outlier],
    }


def test_matching_recovers_the_generated_truth(generated_photos):
    """En las fotos sintéticas, el emparejamiento debe dar exactamente las parejas verdaderas."""
    for _, annotation, _ in generated_photos:
        max_distance = max_pair_distance(annotation["numbers"], 3.0)
        pairs, unpaired_dots, unpaired_numbers = match_dots_numbers(annotation["dots"], annotation["numbers"],
                                                                    max_distance)
        assert as_set(pairs) == as_set(annotation["pairs"])
        assert unpaired_dots == [] and unpaired_numbers == []
