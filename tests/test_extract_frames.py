"""Tests del extractor de fotogramas de los vídeos reales."""

import cv2
import numpy as np

from tools.extract_frames import assign_ids, extract_video, frame_name, read_mapping, write_mapping


def test_video_ids_are_kept_between_runs(tmp_path):
    mapping = assign_ids(["ficha_a.mp4", "ficha_b.mp4"], {})
    assert mapping == {"ficha_a.mp4": "v01", "ficha_b.mp4": "v02"}
    write_mapping(tmp_path / "videos.txt", mapping)
    # Otro día se añade un vídeo nuevo y se vuelve a pasar uno antiguo: el antiguo conserva su número
    again = assign_ids(["ficha_c.mp4", "ficha_a.mp4"], read_mapping(tmp_path / "videos.txt"))
    assert again == {"ficha_a.mp4": "v01", "ficha_b.mp4": "v02", "ficha_c.mp4": "v03"}


def test_frame_names_sort_by_video_and_frame():
    assert frame_name("v03", 45) == "v03_00045.jpg"
    assert sorted([frame_name("v02", 100), frame_name("v01", 900), frame_name("v02", 15)]) == \
        ["v01_00900.jpg", "v02_00015.jpg", "v02_00100.jpg"]


def test_extract_one_frame_every_step(tmp_path):
    video = tmp_path / "prueba.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(31):                                      # 31 fotogramas: se guardan el 0, el 10, el 20 y el 30
        frame = np.full((48, 64, 3), 255, np.uint8)
        cv2.putText(frame, str(i), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()
    read, saved, blurry = extract_video(video, "v01", tmp_path, step=10, min_sharpness=0, jpeg_quality=90)
    assert (read, saved, blurry) == (31, 4, 0)
    assert sorted(path.name for path in tmp_path.glob("v01_*.jpg")) == \
        ["v01_00000.jpg", "v01_00010.jpg", "v01_00020.jpg", "v01_00030.jpg"]


def test_blurry_frames_are_discarded(tmp_path):
    video = tmp_path / "liso.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 128, np.uint8))   # gris liso: nitidez casi 0
    writer.release()
    read, saved, blurry = extract_video(video, "v01", tmp_path, step=1, min_sharpness=10, jpeg_quality=90)
    assert (read, saved, blurry) == (5, 0, 5)
