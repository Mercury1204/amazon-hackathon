import pytest

from ber.config import Config
from ber.duck import _sql_str, connect, format_bytes, temp_size_literal

# The ceilings the stages carried as hard-coded SQL before the runtime-knob
# refactor. format_bytes must keep reproducing these strings exactly, otherwise
# the refactor silently moved a working point.
LEGACY_STAGE_LIMITS = {
    "audit": (8e9, 50 * 1024 ** 3, "8GB", "50GiB"),
    "pairs.train": (10e9, 50 * 1024 ** 3, "10GB", "50GiB"),
    "pairs.inference": (8e9, 50 * 1024 ** 3, "8GB", "50GiB"),
    "features": (12e9, 80 * 1024 ** 3, "12GB", "80GiB"),
    "predict": (8e9, 60 * 1024 ** 3, "8GB", "60GiB"),
    "validation": (10e9, 50 * 1024 ** 3, "10GB", "50GiB"),
    "blocking.idf": (None, None, "8GB", "50GiB"),
    "blocking.generate": (None, None, "8GB", "50GiB"),
    "blocking.run_block": (12e9, 50 * 1024 ** 3, "12GB", "50GiB"),
}


class FakeCon:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, *a, **kw):
        self.sink.append(sql)
        return self

    def __getattr__(self, name):
        return lambda *a, **kw: self


@pytest.fixture()
def statements(monkeypatch, tmp_path):
    sink = []
    monkeypatch.setattr("ber.duck.duckdb.connect", lambda *a, **kw: FakeCon(sink))
    cfg = Config(
        root=tmp_path, dataset_dir=tmp_path / "ds", data_dir=tmp_path / "data",
        models_dir=tmp_path / "models", output_dir=tmp_path / "output",
    )
    return sink, cfg


@pytest.mark.parametrize("value,expected", [
    (8e9, "8GB"),
    (10e9, "10GB"),
    (12e9, "12GB"),
    (4e9, "4GB"),
    (50 * 1024 ** 3, "50GiB"),
    (60 * 1024 ** 3, "60GiB"),
    (80 * 1024 ** 3, "80GiB"),
    (10 * 1024 ** 3, "10GiB"),
])
def test_format_bytes_picks_an_exact_unit(value, expected):
    assert format_bytes(value) == expected


def test_format_bytes_never_emits_a_unitless_value():
    """DuckDB rejects a bare byte count ('Unknown unit for memory'), so the
    formatter must always attach a unit it can parse."""
    for value in (1, 999, 1234, 1234567, 8e9, 50 * 1024 ** 3, 3.7e12):
        assert any(u in format_bytes(value) for u in ("KB", "MB", "GB", "TB",
                                                      "KiB", "MiB", "GiB", "TiB")), value


def test_temp_size_literal_is_quoted():
    assert temp_size_literal(50 * 1024 ** 3) == "'50GiB'"


def test_sql_str_doubles_embedded_apostrophe():
    """DuckDB binds double quotes as identifiers, so a path containing an
    apostrophe must still be emitted as a single-quoted string literal."""
    assert _sql_str("/tmp/o'neil") == "'/tmp/o''neil'"


@pytest.mark.parametrize("stage", sorted(LEGACY_STAGE_LIMITS))
def test_stage_sql_matches_pre_refactor_strings(statements, stage):
    memory, temp, want_memory, want_temp = LEGACY_STAGE_LIMITS[stage]
    sink, cfg = statements
    connect(cfg, memory=memory, temp=temp)
    joined = "\n".join(sink)
    assert f"SET memory_limit='{want_memory}'" in joined
    assert f"PRAGMA max_temp_directory_size='{want_temp}'" in joined
    assert "SET threads=8" in joined
    assert "SET preserve_insertion_order=false" in joined


def test_connect_creates_and_uses_scratch_dir(statements):
    sink, cfg = statements
    connect(cfg)
    joined = "\n".join(sink)
    assert f"SET temp_directory='{(cfg.data_dir / 'tmp').as_posix()}'" in joined
    assert (cfg.data_dir / "tmp").is_dir()


def test_config_knobs_drive_the_session(statements, tmp_path):
    sink, _ = statements
    cfg = Config(
        root=tmp_path, dataset_dir=tmp_path / "ds", data_dir=tmp_path / "data",
        models_dir=tmp_path / "models", output_dir=tmp_path / "out",
        duck_memory_limit=4e9, duck_threads=2, duck_max_temp=10 * 1024 ** 3,
    )
    connect(cfg)
    joined = "\n".join(sink)
    assert "SET memory_limit='4GB'" in joined
    assert "SET threads=2" in joined
    assert "PRAGMA max_temp_directory_size='10GiB'" in joined


def test_scratch_dir_override_from_config(statements, tmp_path):
    sink, _ = statements
    cfg = Config(
        root=tmp_path, dataset_dir=tmp_path / "ds", data_dir=tmp_path / "data",
        models_dir=tmp_path / "models", output_dir=tmp_path / "out",
        duck_tmp_dir=tmp_path / "scratch",
    )
    connect(cfg)
    joined = "\n".join(sink)
    assert f"SET temp_directory='{(tmp_path / 'scratch' / 'tmp').as_posix()}'" in joined
    assert (tmp_path / "scratch" / "tmp").is_dir()


def test_apostrophe_in_path_is_escaped(statements, tmp_path):
    sink, _ = statements
    data_dir = tmp_path / "o'neil" / "data"
    cfg = Config(
        root=tmp_path, dataset_dir=tmp_path / "ds", data_dir=data_dir,
        models_dir=tmp_path / "models", output_dir=tmp_path / "out",
    )
    connect(cfg)
    joined = "\n".join(sink)
    escaped = (data_dir / "tmp").as_posix().replace("'", "''")
    assert f"SET temp_directory='{escaped}'" in joined
    assert 'SET temp_directory="' not in joined


def test_duck_typed_cfg_without_data_dir_creates_no_scratch(monkeypatch, tmp_path):
    """The unit tests hand in stand-in configs; those must not spill into a
    stray ./tmp next to the caller."""
    sink = []
    monkeypatch.setattr("ber.duck.duckdb.connect", lambda *a, **kw: FakeCon(sink))
    monkeypatch.chdir(tmp_path)

    class Stub:
        pass

    connect(Stub())
    joined = "\n".join(sink)
    assert "SET memory_limit='8GB'" in joined
    assert "SET threads=8" in joined
    assert "SET temp_directory=" not in joined
    assert not (tmp_path / "tmp").exists()
