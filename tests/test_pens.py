"""Tests de los rotuladores (ArtificialVision/pens/).

No necesitan el modelo YOLO: se dibuja una escena sintética con rotuladores tipo Edding 1200
(cuerpo blanco con letras negras, capuchón y tapón del color de la tinta) sobre una mesa oscura,
y se usan sus cajas verdaderas en lugar de las de YOLO.
"""

import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ArtificialVision.pens.analysis import pen_color, pen_geometry, segment_grabcut
from ArtificialVision.pens.label_sam import Etiquetador, sync_video_mapping
from ArtificialVision.pens.pipeline import PenPipeline, analyse, draw_pens
from common.config import load_config
from common.images import write_image
from tools.extract_frames import read_mapping, write_mapping

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


def test_grabcut_without_background_returns_no_object():
    image = np.full((30, 40, 3), 100, np.uint8)
    mask = segment_grabcut(image, [0, 0, 40, 30])
    assert mask.shape == (30, 40) and not mask.any()


@pytest.mark.parametrize("missing_masks", [None, SimpleNamespace(data=[])])
def test_sam_missing_mask_keeps_next_detection_aligned(scene, missing_masks):
    image, _, truth = scene
    boxes = [box_of(truth["rojo"]), box_of(truth["azul"])]
    confidences = [0.6, 0.9]
    pixels = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: truth["azul"] > 0))
    replies = [missing_masks, SimpleNamespace(data=[pixels])]
    calls = []

    def sam(image, *, bboxes, verbose):
        calls.append(bboxes)
        return [SimpleNamespace(masks=replies[len(calls) - 1])]

    pipeline = PenPipeline.__new__(PenPipeline)
    pipeline.sam = sam
    masks = pipeline.segment(image, boxes)
    assert calls == [[boxes[0]], [boxes[1]]]
    assert masks[0] is None and np.array_equal(masks[1], truth["azul"])

    result = analyse(image, "escena", boxes, confidences, masks, None, CFG)
    assert len(result["pens"]) == 1
    pen = result["pens"][0]
    assert (pen["id"], pen["box"], pen["confidence"], pen["color"]) == (1, boxes[1], 0.9, "azul")
    assert draw_pens(image, result, masks).shape == image.shape


def test_analyse_gives_millimetres_and_the_angle_on_the_table(scene):
    image, pens, masks = scene
    names = [name for name, _, _ in pens]
    boxes, conf, mask_list = [box_of(masks[n]) for n in names], [0.9] * 3, [masks[n] for n in names]
    # Hoja ArUco simulada: 4 px = 1 mm, sin giro ni perspectiva
    reference = {"found": True, "ids": [0, 1, 2, 3], "lens": None, "shape": image.shape[:2], "rms_mm": 0.1,
                 "max_mm": 0.2, "homography": np.diag([0.25, 0.25, 1.0])}
    result = analyse(image, "escena", boxes, conf, mask_list, reference, CFG)
    for pen, (name, center, angle) in zip(result["pens"], pens):
        assert pen["color"] == name
        assert pen["center_mm"] == pytest.approx([center[0] / 4, center[1] / 4], abs=1)
        assert min(abs(pen["angle_deg"] - angle), 180 - abs(pen["angle_deg"] - angle)) < 2
        assert pen["length_mm"] == pytest.approx(75, abs=2)
    assert result["aruco"]["found"]


@pytest.mark.parametrize("found", [False, True])
def test_pen_cli_replaces_previous_csv_only_with_valid_reference(tmp_path, monkeypatch, found):
    from ArtificialVision.pens import pipeline

    photo = tmp_path / "mesa.png"
    write_image(photo, np.zeros((50, 60, 3), np.uint8))
    output = tmp_path / "salida"
    output.mkdir()
    csv = output / "mesa_rotuladores_mm.csv"
    csv.write_text("coordenadas anteriores", encoding="utf-8")

    def process(image, image_name):
        assert not csv.exists()
        return {"pens": [], "aruco": {"found": found, "markers": [0, 1] if found else [], "rms_mm": 0.2},
                "times_ms": {"total": 1}}, []

    monkeypatch.setattr(pipeline, "PenPipeline", lambda *args: SimpleNamespace(process=process, last_reference=None))
    monkeypatch.setattr(sys, "argv", ["pipeline", str(photo), "--output", str(output)])
    pipeline.main()

    assert csv.exists() == found
    if found:
        assert csv.read_text(encoding="utf-8") == "color,x_mm,y_mm,angulo_deg,largo_mm\n"


def test_pen_labeler_opens_non_ascii_paths(tmp_path, monkeypatch):
    photo = tmp_path / "Visión.png"
    expected = np.full((30, 40, 3), [40, 80, 120], np.uint8)
    write_image(photo, expected)

    def legacy_reader(*args):
        pytest.fail("cv2.imread puede fallar con caracteres no ASCII en Windows")

    monkeypatch.setattr(cv2, "imread", legacy_reader)
    labeler = Etiquetador.__new__(Etiquetador)
    labeler.fotos = [str(photo)]
    labeler.salida = str(tmp_path)
    labeler.abrir(0)
    assert np.array_equal(labeler.img, expected)
    assert (labeler.h, labeler.w) == (30, 40)


def test_pen_labeler_persists_the_origin_of_new_labels(tmp_path):
    frames, labels = tmp_path / "frames", tmp_path / "labels"
    frames.mkdir()
    labels.mkdir()
    write_mapping(frames / "videos.txt", {"foto.mp4": "v06", "sin_fotogramas.mp4": "v07"})
    write_mapping(labels / "videos.txt", {"anterior.mp4": "v01"})

    sync_video_mapping([frames / "v06_00000.jpg"], frames, labels)

    assert read_mapping(labels / "videos.txt") == {"anterior.mp4": "v01", "foto.mp4": "v06"}


@pytest.mark.parametrize("has_previous_mapping", [False, True])
def test_pen_labeler_refuses_conflicting_or_unmapped_existing_labels(tmp_path, has_previous_mapping):
    frames, labels = tmp_path / "frames", tmp_path / "labels"
    frames.mkdir()
    labels.mkdir()
    write_mapping(frames / "videos.txt", {"nuevo.mp4": "v06"})
    (labels / "v06_00000.txt").touch()
    original = {"anterior.mp4": "v06"} if has_previous_mapping else {}
    if has_previous_mapping:
        write_mapping(labels / "videos.txt", original)

    message = "Los mapas de vídeos no coinciden" if has_previous_mapping else "No se conoce el vídeo de la etiqueta"
    with pytest.raises(ValueError, match=message):
        sync_video_mapping([frames / "v06_00000.jpg"], frames, labels)

    assert read_mapping(labels / "videos.txt") == original


def test_pen_labeler_accepts_legacy_photos_when_neither_map_exists(tmp_path):
    frames, labels = tmp_path / "frames", tmp_path / "labels"
    frames.mkdir()
    labels.mkdir()
    (labels / "v06_00000.txt").touch()

    sync_video_mapping([frames / "v06_00000.jpg"], frames, labels)

    assert not (labels / "videos.txt").exists()


def test_frames_numbers_videos_and_saves_every_nth_frame(tmp_path, monkeypatch):
    from ArtificialVision.pens import frames
    video = tmp_path / "rotuladores_1.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    out = tmp_path / "frames"
    config = load_config()
    config["pens"]["labels"] = str(tmp_path / "labels")
    monkeypatch.setattr(frames, "load_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["frames", str(tmp_path), "--every", "5", "--min-sharpness", "0",
                                      "--out", str(out)])
    frames.main()
    assert sorted(p.name for p in out.glob("*.jpg")) == [f"v01_{i:05d}.jpg" for i in (0, 5, 10, 15)]
    assert "v01\trotuladores_1.avi" in (out / "videos.txt").read_text(encoding="utf-8")
