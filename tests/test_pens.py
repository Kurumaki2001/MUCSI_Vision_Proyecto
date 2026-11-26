"""Tests de los rotuladores (ArtificialVision/pens/).

No necesitan el modelo YOLO: se dibuja una escena sintética con rotuladores tipo Edding 1200
(cuerpo blanco con letras negras, capuchón y tapón del color de la tinta) sobre una mesa oscura,
y se usan sus cajas verdaderas en lugar de las de YOLO.
"""

import sys

import cv2
import numpy as np
import pytest

from ArtificialVision.pens.analysis import pen_color, pen_geometry, segment_grabcut
from ArtificialVision.pens.pipeline import analyse
from common.config import load_config

CFG = load_config()["pens"]
INKS = {"rojo": (40, 40, 200), "azul": (190, 90, 20), "negro": (25, 25, 25)}  # BGR


def draw_pen(image, center, angle_deg, ink, length=300, width=34):
    """Dibuja un rotulador y devuelve su máscara verdadera. Capuchón en un extremo, tapón en el otro."""
    cx, cy = center
    unit = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    normal = np.array([-unit[1], unit[0]])

    def band(t0, t1, w):  # cuadrilátero del tramo del eje entre t0 y t1 (fracciones del largo)
        a, b = np.array([cx, cy]) + unit * length * (t0 - 0.5), np.array([cx, cy]) + unit * length * (t1 - 0.5)
        return np.int32([a + normal * w / 2, b + normal * w / 2, b - normal * w / 2, a - normal * w / 2])

    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [band(0, 1, width)], 255)
    cv2.fillPoly(image, [band(0.06, 0.72, width)], (225, 225, 228))            # cuerpo blanco
    cv2.fillPoly(image, [band(0.72, 1.0, width)], ink)                        # capuchón
    cv2.fillPoly(image, [band(0.0, 0.06, width)], ink)                        # tapón
    for t in (0.25, 0.32, 0.39):                                              # letras impresas
        cv2.fillPoly(image, [band(t, t + 0.025, width * 0.4)], (30, 30, 30))
    return mask


@pytest.fixture(scope="module")
def scene():
    image = np.full((900, 1200, 3), (70, 75, 80), np.uint8)  # mesa gris oscura
    image = cv2.add(image, np.random.default_rng(0).integers(0, 8, image.shape, dtype=np.uint8))
    pens = [("rojo", (300, 250), 20), ("azul", (800, 300), 120), ("negro", (600, 650), 170)]
    masks = {name: draw_pen(image, center, angle, INKS[name]) for name, center, angle in pens}
    return image, pens, masks


def box_of(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return [x, y, x + w, y + h]


def test_geometry_gives_center_angle_and_length(scene):
    _, pens, masks = scene
    for name, center, angle in pens:
        g = pen_geometry(masks[name])
        assert np.hypot(g["center"][0] - center[0], g["center"][1] - center[1]) < 3
        assert min(abs(g["angle_deg"] - angle), 180 - abs(g["angle_deg"] - angle)) < 2
        assert g["length"] == pytest.approx(300, abs=6) and g["width"] == pytest.approx(34, abs=4)


def test_color_of_each_pen(scene):
    image, pens, masks = scene
    for name, _, _ in pens:
        assert pen_color(image, masks[name], CFG["color"])[0] == name


def test_white_body_with_printed_letters_is_not_black():
    image = np.full((400, 600, 3), (70, 75, 80), np.uint8)
    mask = draw_pen(image, (300, 200), 0, (225, 225, 228))  # capuchón y tapón también blancos
    color, fractions = pen_color(image, mask, CFG["color"])
    assert color == "desconocido" and 0 < fractions["negro"] < CFG["color"]["min_black_fraction"]


def test_grabcut_recovers_the_pen_from_its_box(scene):
    image, pens, masks = scene
    for name, _, _ in pens:
        found = segment_grabcut(image, box_of(masks[name])) > 0
        truth = masks[name] > 0
        iou = (found & truth).sum() / (found | truth).sum()
        assert iou > 0.85, f"{name}: IoU {iou:.2f}"


def test_analyse_gives_millimetres_and_the_angle_on_the_table(scene):
    image, pens, masks = scene
    names = [name for name, _, _ in pens]
    boxes, conf, mask_list = [box_of(masks[n]) for n in names], [0.9] * 3, [masks[n] for n in names]
    # Hoja ArUco simulada: 4 px = 1 mm, sin giro ni perspectiva
    reference = {"found": True, "ids": [0, 1, 2, 3], "lens": None, "shape": image.shape[:2], "rms_mm": 0.1,
                 "homography": np.diag([0.25, 0.25, 1.0])}
    result = analyse(image, "escena", boxes, conf, mask_list, reference, CFG)
    for pen, (name, center, angle) in zip(result["pens"], pens):
        assert pen["color"] == name
        assert pen["center_mm"] == pytest.approx([center[0] / 4, center[1] / 4], abs=1)
        assert min(abs(pen["angle_deg"] - angle), 180 - abs(pen["angle_deg"] - angle)) < 2
        assert pen["length_mm"] == pytest.approx(75, abs=2)
    assert result["aruco"]["found"]


def test_frames_numbers_videos_and_saves_every_nth_frame(tmp_path, monkeypatch):
    from ArtificialVision.pens import frames
    video = tmp_path / "rotuladores_1.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    out = tmp_path / "frames"
    monkeypatch.setattr(sys, "argv", ["frames", str(tmp_path), "--every", "5", "--out", str(out)])
    frames.main()
    assert sorted(p.name for p in out.glob("*.jpg")) == [f"v01_{i:05d}.jpg" for i in (0, 5, 10, 15)]
    assert "v01\trotuladores_1.avi" in (out / "videos.txt").read_text(encoding="utf-8")
