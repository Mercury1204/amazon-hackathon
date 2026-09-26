"""Feature provenance: candidate-set degrees and staleness refusal.

Uses a generated fixture because the bug only shows when the training pairs are a
sample of a larger candidate set, which is exactly the production situation.
"""

import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from test_end_to_end import SRC, _build_dataset, _run, _write_config


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the train side of the pipeline once for the whole module."""
    root = tmp_path_factory.mktemp("provenance")
    dataset = _build_dataset(root)
    cfg = _write_config(root, dataset)
    _run(cfg, "prepare")
    _run(cfg, "block", "--split", "train")
    _run(cfg, "features", "--split", "train", "--workers", "1", "--combine")
    return root, cfg


def test_degrees_come_from_the_candidate_set_not_the_sampled_pairs(built):
    root, _ = built
    data = root / "artifacts"
    candidates = data / "candidates" / "train_candidates.parquet"
    pairs = data / "pairs" / "train_pairs.parquet"
    features = data / "features" / "train.parquet"

    con = duckdb.connect()
    cand_s1 = dict(
        con.execute(
            f"SELECT s1_id, count(*) FROM read_parquet('{candidates.as_posix()}') GROUP BY s1_id"
        ).fetchall()
    )
    cand_by_id = dict(
        con.execute(
            f"SELECT cand_id, count(*) FROM read_parquet('{candidates.as_posix()}') GROUP BY cand_id"
        ).fetchall()
    )
    pair_s1 = dict(
        con.execute(
            f"SELECT s1_id, count(*) FROM read_parquet('{pairs.as_posix()}') GROUP BY s1_id"
        ).fetchall()
    )
    con.close()

    frame = pd.read_parquet(features)
    assert len(frame) > 0
    assert (frame["s1_degree"] == frame["s1_id"].map(cand_s1)).all()
    assert (frame["cand_degree"] == frame["cand_id"].map(cand_by_id)).all()

    # the fixture must actually exhibit the skew the fix removes, otherwise this
    # test would pass against the old code too
    sampled_differs = [k for k in pair_s1 if pair_s1[k] != cand_s1.get(k)]
    assert sampled_differs, "fixture does not distinguish sampled pairs from candidates"


def test_degree_tables_are_cached_and_fingerprinted(built):
    root, _ = built
    reports = root / "artifacts" / "reports"
    sdeg = reports / "train_s1_degree.parquet"
    cdeg = reports / "train_cand_degree.parquet"
    assert sdeg.exists() and cdeg.exists()
    assert sdeg.with_name(sdeg.name + ".meta.json").exists()
    frame = pd.read_parquet(sdeg)
    assert list(frame.columns) == ["s1_id", "n"]
    assert len(frame) == len(pd.read_parquet(root / "artifacts" / "candidates" / "train_candidates.parquet", columns=["s1_id"]).drop_duplicates())


def test_degree_tables_rebuild_when_candidates_change(built):
    from ber.config import Config
    from ber.features import candidate_degree_tables

    root, _ = built
    cfg = Config.load(root / "config.json")
    cands = root / "artifacts" / "candidates" / "train_candidates.parquet"
    cdeg = root / "artifacts" / "reports" / "train_cand_degree.parquet"

    # unchanged inputs -> reused, file left alone
    before = cdeg.stat().st_mtime_ns
    con = duckdb.connect()
    _, cdeg_path = candidate_degree_tables(cfg, "train", con)
    con.close()
    assert cdeg_path == cdeg
    assert cdeg.stat().st_mtime_ns == before, "fresh degree tables must be reused, not rewritten"

    # change the candidate set -> the fingerprint no longer matches, so a rebuild is due
    cands.write_bytes(cands.read_bytes() + b"")
    con = duckdb.connect()
    candidate_degree_tables(cfg, "train", con)
    con.close()
    assert cdeg.stat().st_mtime_ns != before, "stale degree tables must be rebuilt"


def test_predict_refuses_stale_feature_parts(tmp_path):
    """A silently stale feature set would corrupt the submission, so predict must stop."""
    root = tmp_path
    dataset = _build_dataset(root)
    cfg_path = _write_config(root, dataset)
    _run(cfg_path, "prepare")
    _run(cfg_path, "block", "--split", "test")
    _run(cfg_path, "features", "--split", "test", "--workers", "1")

    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}
    import os

    env = {**os.environ, "PYTHONPATH": str(SRC)}

    # touch the candidate set so the recorded provenance no longer matches
    cands = root / "artifacts" / "candidates" / "test_candidates.parquet"
    cands.write_bytes(cands.read_bytes() + b"")

    proc = subprocess.run(
        [sys.executable, "-m", "ber.cli", "predict", "--split", "test",
         "--threshold", "0.5", "--config", str(cfg_path)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0
    assert "stale" in proc.stderr, proc.stderr
    assert "features --split test" in proc.stderr, proc.stderr


def _load_cfg(root):
    from ber.config import Config

    return Config.load(root / "config.json")
