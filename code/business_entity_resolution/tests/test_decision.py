import json
from pathlib import Path

import pytest

from ber.threshold import (
    FULL_CANDIDATES,
    SAMPLED,
    load_decision,
    resolve_decision,
    save_decision,
)


def _write(cfg, **kwargs):
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return save_decision(cfg, **kwargs)


def test_no_decision_file_raises(cfg):
    with pytest.raises(FileNotFoundError, match="threshold.json"):
        resolve_decision(cfg)


def test_no_decision_file_allowed_with_explicit_threshold(cfg):
    threshold, oto, source = resolve_decision(cfg, threshold_override=0.925)
    assert (threshold, oto, source) == (0.925, False, "cli")


def test_sampled_threshold_is_rejected(cfg):
    """A 4:1-sampled sweep lands far too low for the full candidate set; using it
    silently ships a worse submission, so refuse unless the caller opts in."""
    _write(cfg, threshold=0.70, use_one_to_one=False, source=SAMPLED)
    with pytest.raises(ValueError, match="sampled_4to1"):
        resolve_decision(cfg)


def test_sampled_threshold_allowed_with_explicit_override(cfg):
    _write(cfg, threshold=0.70, use_one_to_one=False, source=SAMPLED)
    threshold, _, source = resolve_decision(cfg, threshold_override=0.925)
    assert (threshold, source) == (0.925, "cli")


def test_full_candidate_threshold_is_used(cfg):
    _write(cfg, threshold=0.925, use_one_to_one=True, source=FULL_CANDIDATES)
    threshold, oto, source = resolve_decision(cfg)
    assert (threshold, oto, source) == (0.925, True, FULL_CANDIDATES)


def test_one_to_one_tri_state(cfg):
    _write(cfg, threshold=0.925, use_one_to_one=True, source=FULL_CANDIDATES)
    assert resolve_decision(cfg)[1] is True
    assert resolve_decision(cfg, one_to_one_override=False)[1] is False
    assert resolve_decision(cfg, one_to_one_override=True)[1] is True


def test_threshold_override_beats_decision(cfg):
    _write(cfg, threshold=0.925, use_one_to_one=True, source=FULL_CANDIDATES)
    assert resolve_decision(cfg, threshold_override=0.5)[0] == 0.5


def test_legacy_decision_file_without_source_is_accepted(cfg):
    """A threshold.json written before the source field existed must still load."""
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    (cfg.models_dir / "threshold.json").write_text(
        json.dumps({"global": 0.925, "use_one_to_one": True}), encoding="utf-8"
    )
    assert resolve_decision(cfg)[0] == 0.925


def test_decision_roundtrip(cfg):
    payload = _write(cfg, threshold=0.925, use_one_to_one=True, source=FULL_CANDIDATES,
                     macro_f05=0.8488, method="one_to_one")
    assert load_decision(cfg) == payload
    assert payload["macro_f05"] == 0.8488
    assert json.loads((cfg.models_dir / "threshold.json").read_text())["method"] == "one_to_one"


def test_load_decision_returns_none_when_absent(cfg):
    assert load_decision(cfg) is None
