import json

import pytest

from ber.tune import tune_threshold

REPORT = {
    "chosen_method": "one_to_one",
    "chosen_macro_f05": 0.8488,
    "ci95": [0.8481, 0.8496],
    "candidate_recall_ceiling": 0.8142,
    "val_s1": 438499,
    "per_country": {"us": 0.889, "india": 0.788},
    "threshold_only": {"macro_f05": 0.8475, "threshold": 0.925},
    "one_to_one": {"macro_f05": 0.8488, "threshold": 0.925},
}


def _write_report(cfg, report=None):
    reports = cfg.data_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "eval_full_candidates.json"
    path.write_text(json.dumps(REPORT if report is None else report), encoding="utf-8")
    return path


def test_tune_persists_the_full_candidate_decision(cfg):
    _write_report(cfg)
    result = tune_threshold(cfg)
    decision = json.loads((cfg.models_dir / "threshold.json").read_text(encoding="utf-8"))
    assert result["threshold"] == 0.925
    assert result["use_one_to_one"] is True
    assert result["method"] == "one_to_one"
    assert decision["source"] == "full_candidates"
    assert decision["macro_f05"] == 0.8488
    assert decision["candidate_recall_ceiling"] == 0.8142
    assert decision["per_country"] == {"us": 0.889, "india": 0.788}


def test_tune_selects_threshold_only_when_that_won(cfg):
    report = dict(REPORT, chosen_method="threshold_only")
    _write_report(cfg, report)
    result = tune_threshold(cfg)
    decision = json.loads((cfg.models_dir / "threshold.json").read_text(encoding="utf-8"))
    assert result["method"] == "threshold_only"
    assert decision["use_one_to_one"] is False
    assert decision["global"] == 0.925


def test_tune_reuses_the_report_without_rerunning(cfg, monkeypatch):
    _write_report(cfg)

    def explode(*a, **kw):
        raise AssertionError("must reuse the existing report")

    monkeypatch.setattr("ber.validation.evaluate_full_candidates", explode)
    assert tune_threshold(cfg)["threshold"] == 0.925


def test_tune_reruns_when_forced(cfg, monkeypatch):
    _write_report(cfg)
    calls = []

    def fake(cfg, **kwargs):
        calls.append(kwargs)
        return REPORT

    monkeypatch.setattr("ber.validation.evaluate_full_candidates", fake)
    assert tune_threshold(cfg, force=True)["threshold"] == 0.925
    assert len(calls) == 1


def test_tune_reruns_when_report_missing(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ber.validation.evaluate_full_candidates",
        lambda cfg, **kw: (calls.append(kw), REPORT)[1],
    )
    assert tune_threshold(cfg)["threshold"] == 0.925
    assert len(calls) == 1


def test_tune_rejects_a_report_without_a_chosen_method(cfg):
    _write_report(cfg, {"macro_f05": 0.5})
    with pytest.raises(KeyError, match="chosen_method"):
        tune_threshold(cfg)


def test_tune_rejects_a_chosen_method_with_no_threshold(cfg):
    _write_report(cfg, dict(REPORT, one_to_one={"macro_f05": 0.8488}))
    with pytest.raises(KeyError, match="no threshold recorded"):
        tune_threshold(cfg)
