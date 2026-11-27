"""Regresiones de extracción y montaje del dataset de rotuladores."""

import sys
from pathlib import Path

import pytest
import yaml

from ArtificialVision.pens import frames, prepare_dataset
from tools.extract_frames import frame_name, read_mapping, write_mapping


@pytest.fixture
def pens_paths(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("videos", "frames", "labels", "dataset")}
    for name in ("videos", "frames", "labels"):
        paths[name].mkdir()
    config = {"pens": {**paths, "classes": ["rotulador"]}, "real_data": {"jpeg_quality": 95}}
    monkeypatch.setattr(frames, "load_config", lambda: config)
    monkeypatch.setattr(prepare_dataset, "load_config", lambda: config)
    return paths, config["pens"]


@pytest.mark.parametrize("previous_extraction", [False, True])
def test_partial_extraction_preserves_the_label_identifiers(pens_paths, monkeypatch, previous_extraction):
    paths, _ = pens_paths
    mapping = {"Rotus_v01.mp4": "v01", "Rotus_v02.mp4": "v02"}
    write_mapping(paths["labels"] / "videos.txt", mapping)
    stale = paths["frames"] / "v02_00015.jpg"
    other_video = paths["frames"] / "v01_00000.jpg"
    if previous_extraction:
        write_mapping(paths["frames"] / "videos.txt", mapping)
        stale.touch()
        other_video.touch()
    video = paths["videos"] / "Rotus_v02.mp4"
    video.touch()

    def extract(path, video_id, destination, step, min_sharpness, jpeg_quality):
        assert (path, video_id, step, min_sharpness, jpeg_quality) == (video, "v02", 20, 30, 95)
        (destination / frame_name(video_id, 20)).write_bytes(b"nuevo fotograma")
        return 21, 1, 0

    monkeypatch.setattr(frames, "extract_video", extract)
    monkeypatch.setattr(sys, "argv", ["frames", str(video)])
    frames.main()

    assert read_mapping(paths["frames"] / "videos.txt") == mapping
    assert not stale.exists()
    assert other_video.exists() == previous_extraction
    assert (paths["frames"] / "v02_00020.jpg").read_bytes() == b"nuevo fotograma"


@pytest.mark.parametrize("main", [frames.main, prepare_dataset.main])
def test_conflicting_video_maps_are_rejected(pens_paths, monkeypatch, capsys, main):
    paths, _ = pens_paths
    (paths["videos"] / "Rotus_v02.mp4").touch()
    write_mapping(paths["labels"] / "videos.txt", {"Rotus_v01.mp4": "v01", "Rotus_v02.mp4": "v02"})
    wrong_map = {"Rotus_v02.mp4": "v01"}
    write_mapping(paths["frames"] / "videos.txt", wrong_map)
    (paths["frames"] / "v01_00000.jpg").touch()
    (paths["labels"] / "v01_00000.txt").touch()
    monkeypatch.setattr(sys, "argv", ["command"])

    with pytest.raises(SystemExit):
        main()

    assert "Los mapas de vídeos no coinciden" in capsys.readouterr().err
    assert read_mapping(paths["frames"] / "videos.txt") == wrong_map
    assert not paths["dataset"].exists()


@pytest.fixture
def annotated_videos(pens_paths):
    paths, _ = pens_paths
    for video_id in ("v01", "v02"):
        (paths["frames"] / f"{video_id}_00000.jpg").write_bytes(video_id.encode())
        (paths["labels"] / f"{video_id}_00000.txt").write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    return paths


def test_paired_labels_must_have_video_provenance_in_a_mapped_dataset(annotated_videos, monkeypatch, capsys):
    paths = annotated_videos
    write_mapping(paths["frames"] / "videos.txt", {"Rotus_v01.mp4": "v01", "Rotus_v02.mp4": "v02"})
    write_mapping(paths["labels"] / "videos.txt", {"Rotus_v01.mp4": "v01"})
    monkeypatch.setattr(sys, "argv", ["prepare_dataset"])

    with pytest.raises(SystemExit):
        prepare_dataset.main()

    assert f"Falta el vídeo de v02 en {paths['labels'] / 'videos.txt'}" in capsys.readouterr().err
    assert not paths["dataset"].exists()


@pytest.mark.parametrize("validation", [["v02", "v01"], ["v03"]])
def test_invalid_validation_split_is_rejected_before_writing(annotated_videos, monkeypatch, validation):
    monkeypatch.setattr(sys, "argv", ["prepare_dataset", "--val", *validation])
    with pytest.raises(SystemExit):
        prepare_dataset.main()
    assert not annotated_videos["dataset"].exists()


def test_dataset_cannot_use_an_input_folder_as_output(pens_paths, annotated_videos, monkeypatch, capsys):
    paths, cfg = pens_paths
    cfg["dataset"] = paths["labels"]
    monkeypatch.setattr(sys, "argv", ["prepare_dataset"])
    with pytest.raises(SystemExit):
        prepare_dataset.main()
    assert "se solapa con la carpeta de origen" in capsys.readouterr().err
    assert len(list(annotated_videos["labels"].glob("*.txt"))) == 2


def test_failed_replacement_restores_the_previous_dataset(annotated_videos, monkeypatch):
    paths = annotated_videos
    monkeypatch.setattr(sys, "argv", ["prepare_dataset"])
    prepare_dataset.main()
    output = paths["dataset"]
    yaml_path = output / "data.yaml"
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["path"] == str(output)
    train_image = output / "images" / "train" / "v01_00000.jpg"
    assert train_image.read_bytes() == b"v01"
    (paths["frames"] / train_image.name).write_bytes(b"nuevo fotograma")
    rename = Path.rename

    def fail_publication(source, target):
        if source.name == "dataset" and target == output:
            raise OSError("fallo simulado al publicar")
        return rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(OSError, match="fallo simulado al publicar"):
        prepare_dataset.main()

    assert train_image.read_bytes() == b"v01"
    assert yaml_path.is_file()
