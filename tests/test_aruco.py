"""Tests de las coordenadas reales con ArUco."""

import cv2
import numpy as np
import pytest

from ArtificialVision.aruco_coords import (ArucoReference, add_mm_coordinates, inside_markers, leave_one_out,
                                           load_sheet, to_mm)
from common.config import load_config, project_path

PX_PER_MM = 6
CFG = {**load_config()["aruco"], "lens_calibration": None}  # la foto sintética no tiene distorsión
SHEET = load_sheet(project_path(CFG["sheet"]))
# Perspectiva de la "cámara": de píxeles de la hoja dibujada a píxeles de la foto
WARP = np.array([[0.80, 0.10, 150.0],
                 [-0.08, 0.85, 120.0],
                 [0.00004, 0.00006, 1.0]])


def mm_to_photo(points_mm):
    """Dónde cae en la foto sintética un punto de la hoja dado en mm."""
    canvas = np.asarray(points_mm, np.float64) * PX_PER_MM - 0.5  # -0.5: el borde del píxel 0 está en -0.5
    return cv2.perspectiveTransform(canvas.reshape(-1, 1, 2), WARP).reshape(-1, 2)


@pytest.fixture(scope="module")
def photo():
    width, height = 210 * PX_PER_MM, 297 * PX_PER_MM
    canvas = np.full((height, width), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, SHEET["dictionary"]))
    side = int(SHEET["marker_size_mm"] * PX_PER_MM)
    for marker_id, corners in SHEET["corners"].items():
        x0, y0 = (corners[0] * PX_PER_MM).astype(int)
        canvas[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    warped = cv2.warpPerspective(canvas, WARP, (1500, 1700), flags=cv2.INTER_LINEAR, borderValue=200)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def test_sheet_is_read_in_millimetres():
    assert SHEET["marker_size_mm"] == pytest.approx(80)
    assert SHEET["corners"][0][0].tolist() == pytest.approx([15, 35])   # esquina TL del marcador 0
    assert SHEET["corners"][3].mean(0).tolist() == pytest.approx([155, 225])


def test_pixels_go_back_to_the_right_millimetres(photo):
    reference = ArucoReference(CFG).locate(photo)
    assert reference["found"] and reference["ids"] == [0, 1, 2, 3]
    assert reference["rms_mm"] < 0.2
    # Puntos de la hoja que NO son esquinas de marcadores: por ejemplo, donde iría la ficha de puntos
    points_mm = np.array([[105, 150], [20, 280], [200, 10], [105, 75]])
    recovered = to_mm(mm_to_photo(points_mm), reference["homography"])
    assert np.abs(recovered - points_mm).max() < 0.3


def test_not_enough_markers_gives_no_homography(photo):
    covered = photo.copy()
    for marker_id in (1, 2, 3):  # tapamos tres marcadores con papel blanco
        x0, y0 = mm_to_photo(SHEET["corners"][marker_id]).min(0).astype(int) - 10
        x1, y1 = mm_to_photo(SHEET["corners"][marker_id]).max(0).astype(int) + 10
        covered[y0:y1, x0:x1] = 255
    reference = ArucoReference(CFG).locate(covered)
    assert reference["ids"] == [0] and not reference["found"] and reference["homography"] is None


def test_leave_one_out_is_accurate_on_a_perfect_photo(photo):
    errors = leave_one_out(ArucoReference(CFG), photo)
    assert sorted(errors) == [0, 1, 2, 3]
    assert max(errors.values()) < 0.3


def test_detections_inside_a_marker_are_discarded():
    square = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float64)
    markers = [(0, square)]
    assert inside_markers(150, 150, markers, 0.15)          # en medio del marcador
    assert inside_markers(204, 150, markers, 0.15)          # justo fuera, pero dentro del margen
    assert not inside_markers(260, 150, markers, 0.15)      # lejos: es un punto de la ficha


def test_mm_coordinates_and_path_in_numeric_order():
    reference = {"found": True, "ids": [0, 1], "lens": None, "shape": (100, 100), "rms_mm": 0.1, "max_mm": 0.2,
                 "homography": np.diag([0.5, 0.5, 1.0])}       # 2 px = 1 mm, sin giro
    result = {"dots": [{"id": 0, "x": 10.0, "y": 20.0}, {"id": 1, "x": 40.0, "y": 60.0}, {"id": 2, "x": 0, "y": 0}],
              "numbers": [{"id": 0, "value": 2}, {"id": 1, "value": 1}, {"id": 2, "value": None}],
              "pairs": [{"dot": 0, "number": 0}, {"dot": 1, "number": 1}, {"dot": 2, "number": 2}]}
    summary = add_mm_coordinates(result, reference)
    assert (result["dots"][0]["x_mm"], result["dots"][0]["y_mm"]) == (5.0, 10.0)
    # El número 1 va primero; el punto con número ilegible (None) no entra en la ruta
    assert [(p["value"], p["x_mm"], p["y_mm"]) for p in summary["path_mm"]] == [(1, 20.0, 30.0), (2, 5.0, 10.0)]


def test_without_sheet_no_mm_coordinates_are_added():
    reference = {"found": False, "ids": [2], "lens": None}
    result = {"dots": [{"id": 0, "x": 1.0, "y": 1.0}], "numbers": [], "pairs": []}
    summary = add_mm_coordinates(result, reference)
    assert summary == {"found": False, "markers": [2], "lens_correction": False}
    assert "x_mm" not in result["dots"][0]
