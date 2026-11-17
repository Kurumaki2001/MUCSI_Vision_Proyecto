"""Datos compartidos por los tests."""

import pytest

from common.config import load_config
from data.generate import available_fonts, generate_photo


@pytest.fixture(scope="session")
def generator_config():
    return load_config()["generator"]


@pytest.fixture(scope="session")
def fonts(generator_config):
    return available_fonts(generator_config["numbers"]["fonts"])


@pytest.fixture(scope="session")
def generated_photos(generator_config, fonts):
    """Unas cuantas fotos sintéticas (photo, annotation, jpeg_quality) con la configuración del proyecto."""
    return [generate_photo(generator_config, "test", i, fonts) for i in range(4)]
