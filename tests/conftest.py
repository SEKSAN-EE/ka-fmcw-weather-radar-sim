import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radar_sim import load_config  # noqa: E402


@pytest.fixture
def lab_cfg():
    return load_config(ROOT / "config/lab.yaml")


@pytest.fixture
def weather_cfg():
    return load_config(ROOT / "config/weather.yaml")
