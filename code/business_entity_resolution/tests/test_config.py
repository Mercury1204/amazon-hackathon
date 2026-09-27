import json
from pathlib import Path

import pytest

from ber.config import Config, _parse_size


def test_env_var_beats_config_file(clean_env, config_file, tmp_path):
    clean_env.setenv("BER_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    cfg = Config.load(config_file())
    assert cfg.data_dir == tmp_path / "artifacts"
    assert cfg.cap == 200


@pytest.mark.parametrize("alias", ["DATA_DIR", "BER_DATA_DIR"])
def test_dataset_dir_aliases(clean_env, config_file, tmp_path, alias):
    clean_env.setenv(alias, str(tmp_path))
    assert Config.load(config_file()).dataset_dir == tmp_path


def test_canonical_env_var_wins_over_alias(clean_env, config_file, tmp_path):
    clean_env.setenv("BER_DATASET_DIR", str(tmp_path / "canonical"))
    clean_env.setenv("DATA_DIR", str(tmp_path / "alias"))
    clean_env.setenv("BER_DATA_DIR", str(tmp_path / "alias2"))
    assert Config.load(config_file()).dataset_dir == tmp_path / "canonical"


def test_all_four_dirs_overridable(clean_env, config_file, tmp_path):
    clean_env.setenv("BER_DATASET_DIR", str(tmp_path / "ds"))
    clean_env.setenv("BER_ARTIFACT_DIR", str(tmp_path / "art"))
    clean_env.setenv("BER_MODELS_DIR", str(tmp_path / "models"))
    clean_env.setenv("BER_OUTPUT_DIR", str(tmp_path / "out"))
    cfg = Config.load(config_file())
    assert (cfg.dataset_dir, cfg.data_dir, cfg.models_dir, cfg.output_dir) == (
        tmp_path / "ds", tmp_path / "art", tmp_path / "models", tmp_path / "out"
    )


def test_data_and_DATA_stay_distinct(clean_env, config_file, tmp_path):
    """The lowercase `data` artifact dir must not collapse into the `DATA` input dir."""
    clean_env.setenv("BER_ROOT", str(tmp_path))
    cfg = Config.load(config_file(data_dir="data", dataset_dir="DATA/student_resource/dataset"))
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.dataset_dir == tmp_path / "DATA/student_resource/dataset"
    assert cfg.data_dir != cfg.dataset_dir


def test_unknown_config_key_is_rejected(clean_env, config_file):
    path = config_file(pass_capz={"1": 10})
    with pytest.raises(ValueError, match="pass_capz"):
        Config.load(path)


def test_relative_paths_resolve_against_root(clean_env, config_file, tmp_path):
    clean_env.setenv("BER_ROOT", str(tmp_path))
    assert Config.load(config_file()).data_dir == tmp_path / "data"


def test_expanduser_applied(clean_env, config_file):
    clean_env.setenv("BER_ARTIFACT_DIR", "~/artifacts")
    resolved = str(Config.load(config_file()).data_dir)
    assert "~" not in resolved
    assert resolved.startswith(str(Path.home()))


def test_duck_knobs_from_env(clean_env, config_file):
    clean_env.setenv("BER_DUCK_MEMORY_LIMIT", "6GB")
    clean_env.setenv("BER_DUCK_THREADS", "4")
    clean_env.setenv("BER_DUCK_MAX_TEMP", "30GiB")
    clean_env.setenv("BER_DUCK_TMP_DIR", "/scratch")
    cfg = Config.load(config_file())
    assert cfg.duck_memory_limit == 6e9
    assert cfg.duck_threads == 4
    assert cfg.duck_max_temp == 30 * 1024 ** 3
    assert cfg.scratch_dir == Path("/scratch/tmp")


def test_duck_defaults_when_unset(clean_env, config_file):
    cfg = Config.load(config_file())
    assert cfg.duck_memory_limit == 8e9
    assert cfg.duck_threads == 8
    assert cfg.duck_max_temp == 50 * 1024 ** 3
    assert cfg.scratch_dir == cfg.data_dir / "tmp"


@pytest.mark.parametrize("text,expected", [
    ("8GB", 8_000_000_000),
    ("50GiB", 50 * 1024 ** 3),
    ("512MB", 512_000_000),
    ("1024", 1024),
    ("2gb", 2_000_000_000),
])
def test_parse_size(text, expected):
    assert _parse_size(text, "T") == expected


@pytest.mark.parametrize("text", ["abc", "8 quids", "", "0"])
def test_parse_size_rejects_garbage(text):
    with pytest.raises(ValueError):
        _parse_size(text, "T")


def test_bad_duck_env_rejected_at_load(clean_env, config_file):
    clean_env.setenv("BER_DUCK_THREADS", "notanint")
    with pytest.raises(ValueError):
        Config.load(config_file())


def test_validate_fails_fast_on_bad_dataset_dir(clean_env, config_file, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    clean_env.setenv("BER_DATASET_DIR", str(empty))
    assert Config.load(config_file()).dataset_dir == empty
    with pytest.raises(FileNotFoundError, match="train/"):
        Config.load(config_file(), validate=True)


def test_validate_passes_with_train_dir(clean_env, config_file, tmp_path):
    (tmp_path / "train").mkdir()
    clean_env.setenv("BER_DATASET_DIR", str(tmp_path))
    assert Config.load(config_file(), validate=True).cap == 200


def _fake_mount(base, relative):
    """Create a minimal mounted-dataset tree at base/relative."""
    root = base / relative
    for split in ("train", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)
        (root / split / f"{split}_source1.tsv").write_text("entity_id\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("layout", [
    "amz-er-2026-raw",                          # older sessions
    "datasets/mercury147/amz-er-2026-raw",       # current sessions
    "datasets/mercury147/amz-er-2026-raw/nested/extra",
])
def test_kaggle_autodetect_finds_both_mount_layouts(tmp_path, layout):
    """The mount depth is not stable across Kaggle sessions; depth-1 and depth-2 globs
    missed the current /kaggle/input/datasets/<owner>/<slug> layout entirely."""
    from ber.config import _kaggle_dataset_dir

    base = tmp_path / "input"
    base.mkdir()
    expected = _fake_mount(base, layout)
    assert _kaggle_dataset_dir(base) == expected


def test_kaggle_autodetect_requires_both_splits(tmp_path):
    from ber.config import _kaggle_dataset_dir

    base = tmp_path / "input"
    (base / "code-dataset" / "train").mkdir(parents=True)
    (base / "code-dataset" / "train" / "train_source1.tsv").write_text("x\n", encoding="utf-8")
    assert _kaggle_dataset_dir(base) is None


def test_kaggle_autodetect_prefers_the_complete_dataset(tmp_path):
    from ber.config import _kaggle_dataset_dir

    base = tmp_path / "input"
    (base / "aaa-partial" / "train").mkdir(parents=True)
    (base / "aaa-partial" / "train" / "train_source1.tsv").write_text("x\n", encoding="utf-8")
    expected = _fake_mount(base, "zzz-complete")
    assert _kaggle_dataset_dir(base) == expected


def test_kaggle_autodetect_returns_none_for_empty_mount(tmp_path):
    from ber.config import _kaggle_dataset_dir

    base = tmp_path / "input"
    base.mkdir()
    assert _kaggle_dataset_dir(base) is None
    assert _kaggle_dataset_dir(tmp_path / "absent") is None


def test_kaggle_autodetect_inert_without_kaggle_input(clean_env, config_file):
    from ber.config import _kaggle_dataset_dir
    from pathlib import Path as P
    if P("/kaggle/input").exists():
        pytest.skip("running on Kaggle; auto-detect is expected to fire")
    assert _kaggle_dataset_dir() is None
    assert Config.load(config_file()).dataset_dir.name == "dataset"


def test_describe_is_json_serialisable(clean_env, config_file):
    assert json.loads(json.dumps(Config.load(config_file()).describe()))["scratch_dir"]


def test_direct_construction_keeps_duck_defaults(tmp_path):
    """conftest's `cfg` fixture bypasses load(); it must still work unchanged."""
    cfg = Config(
        root=tmp_path, dataset_dir=tmp_path / "dataset", data_dir=tmp_path / "data",
        models_dir=tmp_path / "models", output_dir=tmp_path / "output",
    )
    assert cfg.duck_threads == 8
    assert cfg.scratch_dir == tmp_path / "data" / "tmp"
