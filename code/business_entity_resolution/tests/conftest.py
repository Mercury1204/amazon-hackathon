import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ber.config import Config  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    return Config(
        root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "output",
        seed=42,
        cap=200,
        idf_min=4.0,
        max_block=5000,
        neg_ratio=4,
        lgbm_params={"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63},
    )


ALL_ENV = [
    "BER_DATASET_DIR", "DATA_DIR", "BER_DATA_DIR", "BER_ARTIFACT_DIR", "BER_MODELS_DIR",
    "BER_OUTPUT_DIR", "BER_ROOT", "BER_DUCK_MEMORY_LIMIT", "BER_DUCK_THREADS",
    "BER_DUCK_MAX_TEMP", "BER_DUCK_TMP_DIR",
]


@pytest.fixture()
def clean_env(monkeypatch):
    """Every BER_*/DATA_DIR variable cleared, so a test sees real defaults."""
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture()
def config_file(tmp_path):
    """Factory writing a minimal valid config file; kwargs override any key."""

    def _write(**overrides):
        payload = {
            "dataset_dir": "DATA/student_resource/dataset",
            "data_dir": "data",
            "models_dir": "code/business_entity_resolution/models",
            "output_dir": "output",
            "seed": 42,
            "cap": 200,
        }
        payload.update(overrides)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


def test_config_load_roundtrip(tmp_path):
    payload = {
        "dataset_dir": str(tmp_path / "dataset"),
        "data_dir": str(tmp_path / "data"),
        "models_dir": str(tmp_path / "models"),
        "output_dir": str(tmp_path / "output"),
        "seed": 7,
        "cap": 150,
        "idf_min": 3.5,
        "max_block": 4000,
        "neg_ratio": 3,
        "lgbm_params": {"n_estimators": 100},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = Config.load(path)
    assert loaded.seed == 7
    assert loaded.cap == 150
    assert loaded.dataset_dir == tmp_path / "dataset"
