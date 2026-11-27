"""Tests de las coordenadas reales con ArUco."""

import sys

import cv2
import numpy as np
import pytest

from ArtificialVision import pipeline as artificial_pipeline
from ClassicVision import pipeline as classic_pipeline
from common import aruco_calibrate, aruco_capture, json_format
from common.aruco_coords import ArucoReference, add_mm_coordinates, inside_markers, leave_one_out, load_sheet, to_mm
from common.config import load_config, project_path
from common.images import write_image

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


def test_duplicate_sheet_marker_invalidates_the_reference(photo, monkeypatch):
    finder = ArucoReference(CFG)
    markers, all_markers = finder.detect_markers(photo)
    duplicate = (0, markers[0] + [200, 0])
    monkeypatch.setattr(finder, "detect_markers", lambda _: (markers, [*all_markers, duplicate]))
    reference = finder.locate(photo)
    assert reference["reason"] == "duplicate_markers" and reference["duplicate_ids"] == [0]
    assert not reference["found"] and reference["homography"] is None
    assert leave_one_out(finder, photo) == {}


def test_inconsistent_markers_give_an_error_but_no_coordinates(photo, monkeypatch):
    finder = ArucoReference(CFG)
    markers = {i: mm_to_photo(corners) for i, corners in SHEET["corners"].items()}
    markers[1] += [30, -10]
    monkeypatch.setattr(finder, "detect_markers", lambda _: (markers, list(markers.items())))
    reference = finder.locate(photo)
    assert reference["reason"] == "high_rms" and reference["rms_mm"] > CFG["max_rms_mm"]
    assert not reference["found"] and reference["homography"] is None
    result = {"dots": [{"id": 0, "x": 10, "y": 20}], "numbers": [], "pairs": []}
    summary = add_mm_coordinates(result, reference)
    assert summary["reason"] == "high_rms" and summary["rms_mm"] > CFG["max_rms_mm"]
    assert "path_mm" not in summary and "x_mm" not in result["dots"][0]


def test_degenerate_markers_give_no_homography(photo, monkeypatch):
    finder = ArucoReference(CFG)
    markers = {i: np.column_stack((np.arange(i * 4, i * 4 + 4, dtype=float), np.ones(4))) for i in (0, 1)}
    monkeypatch.setattr(finder, "detect_markers", lambda _: (markers, list(markers.items())))
    reference = finder.locate(photo)
    assert reference["reason"] == "invalid_homography"
    assert not reference["found"] and reference["homography"] is None


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


def test_calibration_rejects_mixed_capture_resolutions(tmp_path, monkeypatch):
    captures = tmp_path / "captures"
    captures.mkdir()
    write_image(captures / "first.png", np.zeros((48, 64, 3), np.uint8))
    write_image(captures / "second.png", np.zeros((72, 128, 3), np.uint8))
    output = tmp_path / "lens.json"
    output.write_text("calibración anterior", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aruco_calibrate", str(captures), "--out", str(output)])

    with pytest.raises(SystemExit, match="second.png tiene 128x72; se esperaba 64x48"):
        aruco_calibrate.main()

    assert output.read_text(encoding="utf-8") == "calibración anterior"


def test_capture_checks_actual_frame_resolution_before_saving(tmp_path, monkeypatch):
    previous = tmp_path / "previous.png"
    write_image(previous, np.zeros((48, 64, 3), np.uint8))
    frame = np.zeros((72, 128, 3), np.uint8)

    class FakeCamera:
        def get(self, prop):
            return 64 if prop == cv2.CAP_PROP_FRAME_WIDTH else 48  # la resolución real es otra

        def read(self):
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(aruco_capture, "open_camera", lambda *_: FakeCamera())
    monkeypatch.setattr(sys, "argv", ["aruco_capture", "--out", str(tmp_path), "--no-gui", "--interval", "1", "--count", "1"])
    with pytest.raises(SystemExit, match="128x72.*64x48.*--out"):
        aruco_capture.main()
    assert list(tmp_path.iterdir()) == [previous]


@pytest.mark.parametrize("module", [classic_pipeline, artificial_pipeline])
def test_inference_publishes_csv_only_for_the_current_valid_result(module, tmp_path, monkeypatch):
    image_path = tmp_path / "ficha.png"
    write_image(image_path, np.zeros((20, 20, 3), np.uint8))
    output = tmp_path / "output"
    csv_path = output / "ficha_mm.csv"
    references = iter([
        {"found": True, "markers": [0, 1], "rms_mm": 0.2, "path_mm": [{"value": 1, "x_mm": 4, "y_mm": 5}]},
        {"found": False, "markers": []},
    ])

    class FakePipeline:
        last_reference = None

        def __init__(self, _config):
            pass

        def process(self, image, name, *_args):
            assert not csv_path.exists()
            return json_format.create(name, image.shape[1], image.shape[0], [], [], [], times_ms={"total": 0},
                                      metadata={"sequence": None, "aruco": next(references)})

    monkeypatch.setattr(module, "Pipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["pipeline", str(image_path), "--output", str(output)])

    module.main()
    assert csv_path.read_text(encoding="utf-8") == "value,x_mm,y_mm\n1,4,5\n"
    module.main()
    assert not csv_path.exists()
    assert not json_format.load(output / "ficha.json")["metadata"]["aruco"]["found"]


@pytest.mark.parametrize("module", [classic_pipeline, artificial_pipeline])
def test_inference_rejects_output_collisions_before_writing(module, tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    for name in ("Ficha.png", "ficha.jpg"):
        (source / name).touch()
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["pipeline", str(source), "--output", str(output)])

    with pytest.raises(SystemExit, match="mismo nombre de salida"):
        module.main()

    assert not output.exists()
