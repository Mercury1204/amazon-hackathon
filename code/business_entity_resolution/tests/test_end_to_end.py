import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
REPO = Path(__file__).resolve().parents[3]
VALIDATOR = REPO / "DATA/student_resource/utils/validate_submission.py"

HEADER_REC = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
HEADER_GT = "source1_entity_id\tmatched_entity_ids\n"


def _write_records(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(HEADER_REC)
        for row in rows:
            fh.write("\t".join(row) + "\n")


def _write_gt(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(HEADER_GT)
        for s1, mids in rows:
            fh.write(f"{s1}\t{mids}\n")


def _build_split(directory, n_entities, n_singletons, per_entity, noise, split):
    """Exact-name matches so pass-1 blocking fires, plus unmatched noise records.

    Entity ids must carry the S1-/S2-/S3- prefixes the pipeline and the official
    validator both rely on (blocking derives `is_s2` from `cand_id LIKE 'S2-%'`).
    """
    directory.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    matched = n_entities - n_singletons
    for i in range(n_entities):
        eid = f"S1-{i:08d}"
        if i % 2:
            name = f"Kilimanjaro Traders {i:03d} Pvt"
            addr = f"{100 + i} Linking Road, Pune, Maharashtra 4110{i:02d}"
            country = "India"
        else:
            name = f"Contoso Outlet {i:03d} LLC"
            addr = f"{100 + i} Market Street, Austin, TX {73000 + i:05d}"
            country = "US"
        s1.append((eid, name, addr, country))
        if i >= matched:
            gt.append((eid, ""))
            continue
        mids = []
        for k in range(per_entity):
            cid = f"S2-{(split == 'test') * 10**8 + i * 10 + k:09d}"
            s2.append((cid, name, addr, country))
            mids.append(cid)
        for k in range(per_entity):
            cid = f"S3-{(split == 'test') * 10**8 + i * 10 + k:09d}"
            s3.append((cid, name, addr, country))
            mids.append(cid)
        gt.append((eid, ",".join(mids)))
    for k in range(noise):
        s2.append((f"S2-{(split == 'test') * 10**8 + 9_000_000 + k:09d}",
                   f"Unrelated Emporium {k:03d}",
                   f"{900 + k} Oak Road, Dallas, TX 75001", "US"))
    _write_records(directory / f"{split}_source1.tsv", s1)
    _write_records(directory / f"{split}_source2.tsv", s2)
    _write_records(directory / f"{split}_source3.tsv", s3)
    return s1, gt


def _build_dataset(root):
    dataset = root / "dataset"
    _, gt = _build_split(dataset / "train", n_entities=30, n_singletons=3,
                         per_entity=2, noise=12, split="train")
    _build_split(dataset / "test", n_entities=18, n_singletons=0,
                 per_entity=2, noise=8, split="test")
    _write_gt(dataset / "train" / "train_ground_truth.tsv", gt)
    return dataset


def _write_config(root, dataset):
    payload = {
        "dataset_dir": str(dataset),
        "data_dir": str(root / "artifacts"),
        "models_dir": str(root / "models"),
        "output_dir": str(root / "output"),
        "seed": 42,
        "cap": 200,
        "idf_min": 1.0,
        "max_block": 5000,
        "neg_ratio": 2,
        "pass_caps": {"1": 5000, "3": 300, "4": 3000, "5": 1000, "6": 50,
                      "7": 600, "8": 300, "9": 200, "10": 60},
        "lgbm_params": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15,
                        "min_child_samples": 5},
    }
    path = root / "config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run(cfg, *args):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "ber.cli", *args, "--config", str(cfg)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"ber.cli {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


@pytest.mark.slow
def test_tune_then_outputs_passes_the_official_validator(tmp_path):
    if not VALIDATOR.exists():
        pytest.skip(f"official validator not present at {VALIDATOR}")
    dataset = _build_dataset(tmp_path)
    cfg = _write_config(tmp_path, dataset)

    _run(cfg, "prepare")
    _run(cfg, "block", "--split", "both")
    _run(cfg, "features", "--split", "train", "--workers", "1", "--combine")
    _run(cfg, "features", "--split", "test", "--workers", "1")
    _run(cfg, "train")
    _run(cfg, "tune")

    # tune must leave an authoritative, full-candidate decision behind
    decision = json.loads((tmp_path / "models" / "threshold.json").read_text(encoding="utf-8"))
    assert decision["source"] == "full_candidates", decision
    assert 0.0 < decision["global"] < 1.0
    assert "macro_f05" in decision
    # the tree count train recorded must survive the tune rewrite, or inference would
    # use every built tree instead of the one the threshold was tuned at
    assert decision["best_iteration"] == 1, decision

    _run(cfg, "predict", "--split", "test")
    matching = tmp_path / "output" / "matching_results.tsv"
    candidate = tmp_path / "output" / "candidate_pairs.tsv"
    assert matching.exists() and candidate.exists()

    # outputs must regenerate both files from cached predictions alone
    before = matching.read_bytes()
    (tmp_path / "output" / "matching_results.tsv").unlink()
    (tmp_path / "output" / "candidate_pairs.tsv").unlink()
    _run(cfg, "outputs", "--split", "test")
    assert matching.read_bytes() == before

    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), "--matching", str(matching),
         "--candidate", str(candidate), "--test-dir", str(dataset / "test")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout

    # A conservative tuned threshold can legitimately select nothing on a fixture this
    # small, so also drive a low threshold through the one-to-one branch and re-validate,
    # covering the non-empty match path and the official validator's prefix/dup rules.
    _run(cfg, "predict", "--split", "test", "--threshold", "0.2", "--one-to-one")
    non_empty = sum(
        1 for line in matching.read_text(encoding="utf-8").splitlines()[1:] if line.split("\t")[1]
    )
    assert non_empty > 0, "expected the low threshold to produce matches"
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), "--matching", str(matching),
         "--candidate", str(candidate), "--test-dir", str(dataset / "test"),
         "--check-ids"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.slow
def test_predict_refuses_a_sampled_sweep_threshold(tmp_path):
    """The regression this guards: train writes a 4:1-sampled threshold, and honouring
    it silently shipped a materially worse submission (FAILURES_AND_FIXES F12)."""
    dataset = _build_dataset(tmp_path)
    cfg = _write_config(tmp_path, dataset)
    _run(cfg, "prepare")
    _run(cfg, "block", "--split", "both")
    _run(cfg, "features", "--split", "train", "--workers", "1", "--combine")
    _run(cfg, "features", "--split", "test", "--workers", "1")
    _run(cfg, "train")

    decision = json.loads((tmp_path / "models" / "threshold.json").read_text(encoding="utf-8"))
    assert decision["source"] == "sampled_4to1", decision

    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "ber.cli", "predict", "--split", "test", "--config", str(cfg)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0
    assert "sampled_4to1" in proc.stderr, proc.stderr

    # an explicit --threshold is the documented escape hatch
    proc = subprocess.run(
        [sys.executable, "-m", "ber.cli", "predict", "--split", "test",
         "--threshold", str(decision["global"]), "--config", str(cfg)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.slow
def test_outputs_rejects_a_mismatched_split(tmp_path):
    dataset = _build_dataset(tmp_path)
    cfg = _write_config(tmp_path, dataset)
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "ber.cli", "outputs", "--split", "train", "--config", str(cfg)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0
    assert "not supported" in proc.stderr, proc.stderr
