import json
from pathlib import Path

from ber.cache import (
    code_stamp,
    describe_staleness,
    digest,
    fingerprint,
    invalidate,
    is_fresh,
    meta_path,
    read_context,
    read_meta,
    stamp,
    stamps,
    write_meta,
)


def test_digest_is_stable_and_order_independent():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
    assert len(digest({"a": 1})) == 12


def test_stamp_detects_missing_and_present(tmp_path):
    assert stamp(tmp_path / "nope.parquet") is None
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    s = stamp(f)
    assert s["size"] == 5 and s["mtime_ns"] > 0
    f.write_text("hello world", encoding="utf-8")
    assert stamp(f)["size"] != s["size"]


def test_stamp_of_directory_covers_contents(tmp_path):
    d = tmp_path / "parts"
    d.mkdir()
    before = stamp(d)
    (d / "part_0000.parquet").write_text("x", encoding="utf-8")
    after = stamp(d)
    assert after["size"] != before["size"]
    assert after["files"] == 1


def test_write_and_verify_roundtrip(tmp_path):
    art = tmp_path / "candidates.parquet"
    art.write_text("data", encoding="utf-8")
    src = tmp_path / "in.parquet"
    src.write_text("x", encoding="utf-8")
    inputs = stamps([src])
    write_meta(art, inputs, {"cap": 200})
    assert is_fresh(art, inputs, {"cap": 200})
    assert not is_fresh(art, inputs, {"cap": 100})
    assert not is_fresh(art, stamps([src, tmp_path / "other"]), {"cap": 200})


def test_artifact_without_provenance_is_stale(tmp_path):
    """Pre-fingerprint artifacts must not be silently trusted."""
    art = tmp_path / "old.parquet"
    art.write_text("data", encoding="utf-8")
    assert not is_fresh(art, {}, {})
    assert "no provenance" in describe_staleness(art, {}, {})


def test_code_change_invalidates(tmp_path):
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    write_meta(art, {}, {}, {"blocking.py": "aaa"})
    assert is_fresh(art, {}, {}, {"blocking.py": "aaa"})
    assert not is_fresh(art, {}, {}, {"blocking.py": "bbb"})


def test_config_change_invalidates(tmp_path):
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    write_meta(art, {}, {"pass_caps": {"1": 5000}})
    assert is_fresh(art, {}, {"pass_caps": {"1": 5000}})
    assert not is_fresh(art, {}, {"pass_caps": {"1": 100}})


def test_describe_staleness_names_the_cause(tmp_path):
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    src = tmp_path / "in.parquet"
    src.write_text("x", encoding="utf-8")
    write_meta(art, stamps([src]), {"idf_min": 4.0}, {"blocking.py": "aaa"})
    src.write_text("changed", encoding="utf-8")
    reason = describe_staleness(art, stamps([src]), {"idf_min": 6.0}, {"blocking.py": "bbb"})
    assert "in.parquet changed" in reason
    assert "idf_min" in reason
    assert "blocking.py changed" in reason


def test_read_meta_tolerates_corruption(tmp_path):
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    meta_path(art).write_text("{not json", encoding="utf-8")
    assert read_meta(art) is None
    assert not is_fresh(art, {}, {})


def test_invalidate_removes_artifact_and_sidecar(tmp_path):
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    write_meta(art, {}, {})
    invalidate(art)
    assert not art.exists()
    assert not meta_path(art).exists()


def test_invalidate_clears_a_directory(tmp_path):
    d = tmp_path / "parts"
    d.mkdir()
    (d / "p0.parquet").write_text("x", encoding="utf-8")
    write_meta(d, {}, {})
    invalidate(d)
    assert not d.exists()
    assert not meta_path(d).exists()


def test_fingerprint_matches_written_meta(tmp_path):
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    written = write_meta(art, {"i": 1}, {"c": 2}, {"m": "z"})
    meta = json.loads(meta_path(art).read_text(encoding="utf-8"))
    assert meta["fingerprint"] == written
    assert fingerprint({"i": 1}, {"c": 2}, {"m": "z"}) == written


def test_code_stamp_is_content_sensitive(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("x = 1\n", encoding="utf-8")
    before = code_stamp(src)
    src.write_text("x = 2\n", encoding="utf-8")
    assert code_stamp(src) != before
    assert code_stamp(tmp_path / "absent.py") == {"absent.py": None}


def test_context_is_recorded_but_outside_the_fingerprint(tmp_path):
    """A consumer must be able to recover how an artifact was built (which metadata
    split a valfull feature set borrowed) without changing what the fingerprint means."""
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    plain = write_meta(art, {"i": 1}, {"c": 2}, {"m": "z"})
    with_ctx = write_meta(art, {"i": 1}, {"c": 2}, {"m": "z"}, {"meta_split": "train"})
    assert plain == with_ctx, "context must not perturb the fingerprint"
    assert read_context(art) == {"meta_split": "train"}
    assert is_fresh(art, {"i": 1}, {"c": 2}, {"m": "z"})


def test_read_context_defaults_to_empty(tmp_path):
    art = tmp_path / "a.parquet"
    art.write_text("d", encoding="utf-8")
    assert read_context(art) == {}
    meta_path(art).write_text("{bad", encoding="utf-8")
    assert read_context(art) == {}
