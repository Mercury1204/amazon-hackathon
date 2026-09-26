import json
from pathlib import Path

import numpy as np
import pandas as pd

DECISION_FILE = "threshold.json"

# A threshold swept on the 4:1 sampled split is optimistic and must not be used for
# inference (see FAILURES_AND_FIXES F12); only a full-candidate sweep is authoritative.
SAMPLED = "sampled_4to1"
FULL_CANDIDATES = "full_candidates"


def decision_path(cfg) -> Path:
    return Path(cfg.models_dir) / DECISION_FILE


def save_decision(cfg, threshold, use_one_to_one, source, **extra) -> dict:
    payload = {
        "global": float(threshold),
        "use_one_to_one": bool(use_one_to_one),
        "source": source,
    }
    payload.update(extra)
    path = decision_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_decision(cfg):
    path = decision_path(cfg)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_iteration(cfg, default=-1):
    """How many trees to use at inference.

    LightGBM does not persist `best_iteration` inside the model file, so a booster loaded
    with `model_file` reports -1 and `predict()` then silently uses every tree that was
    built -- including any extra trees early stopping appended after the optimum the
    threshold was tuned at. `train` records the value in the decision file for this reason.
    """
    decision = load_decision(cfg)
    if not decision:
        return default
    value = decision.get("best_iteration")
    return int(value) if value is not None else default


def resolve_decision(cfg, threshold_override=None, one_to_one_override=None):
    """Combine the persisted decision with any explicit override.

    A threshold swept on the 4:1 sample is rejected unless the caller passes one
    explicitly: it is materially too low for the full candidate distribution
    (FAILURES_AND_FIXES F12), and silently honouring it ships a worse submission.
    """
    decision = load_decision(cfg)
    if decision is None:
        if threshold_override is None:
            raise FileNotFoundError(
                f"no decision file at {decision_path(cfg)}. Run `ber.cli tune`, or pass "
                f"--threshold explicitly."
            )
        threshold, source = float(threshold_override), "cli"
    else:
        threshold = float(decision["global"])
        source = decision.get("source", "unknown")
    if threshold_override is not None:
        threshold, source = float(threshold_override), "cli"
    if source == SAMPLED:
        raise ValueError(
            f"the persisted threshold {threshold} came from a {SAMPLED} sweep and is not "
            f"valid for inference. Run `ber.cli tune` to re-tune on the full candidate "
            f"set, or pass --threshold {threshold} to override deliberately."
        )
    if one_to_one_override is None:
        use_one_to_one = bool(decision.get("use_one_to_one", False)) if decision else False
    else:
        use_one_to_one = bool(one_to_one_override)
    return threshold, use_one_to_one, source


def entity_f05(groups, labels, predicted, thresholds=None):
    frame = pd.DataFrame({"g": groups, "y": labels})
    pred = np.asarray(predicted) if thresholds is None else (np.asarray(predicted) >= thresholds)
    frame["p"] = pred
    frame["tp"] = (frame["y"].astype(bool) & frame["p"].astype(bool)).astype(int)
    agg = frame.groupby("g").agg(tp=("tp", "sum"), npred=("p", "sum"), ntrue=("y", "sum"))
    scores = np.ones(len(agg), dtype=np.float64)
    has_true = agg["ntrue"] > 0
    no_pred = agg["npred"] == 0
    scores[has_true & no_pred] = 0.0
    valid = has_true & ~no_pred
    tp = agg.loc[valid, "tp"].to_numpy(dtype=np.float64)
    npred = agg.loc[valid, "npred"].to_numpy(dtype=np.float64)
    ntrue = agg.loc[valid, "ntrue"].to_numpy(dtype=np.float64)
    precision = tp / npred
    recall = tp / ntrue
    f = np.zeros_like(precision)
    nz = precision > 0
    f[nz] = (1.25 * precision[nz] * recall[nz]) / (0.25 * precision[nz] + recall[nz])
    scores[valid] = f
    empty_true = ~has_true
    scores[empty_true] = 0.0
    scores[empty_true & no_pred] = 1.0
    return float(scores.mean())


def _entity_f05_from_codes(codes, n_groups, labels, predicted):
    codes = np.asarray(codes)
    y = np.asarray(labels).astype(bool)
    p = np.asarray(predicted).astype(bool)
    ntrue = np.bincount(codes, weights=y.astype(np.float64), minlength=n_groups)
    npred = np.bincount(codes, weights=p.astype(np.float64), minlength=n_groups)
    tp = np.bincount(codes, weights=(y & p).astype(np.float64), minlength=n_groups)
    precision = np.divide(tp, npred, out=np.zeros(n_groups), where=npred > 0)
    recall = np.divide(tp, ntrue, out=np.zeros(n_groups), where=ntrue > 0)
    f = np.zeros(n_groups, dtype=np.float64)
    nz = precision > 0
    f[nz] = (1.25 * precision[nz] * recall[nz]) / (0.25 * precision[nz] + recall[nz])
    has_true = ntrue > 0
    no_pred = npred == 0
    scores = np.ones(n_groups, dtype=np.float64)
    scores[has_true & no_pred] = 0.0
    scores[has_true & ~no_pred] = f[has_true & ~no_pred]
    empty_true = ~has_true
    scores[empty_true] = 0.0
    scores[empty_true & no_pred] = 1.0
    return float(scores.mean())


def sweep_threshold(probs, groups, labels, low=0.05, high=0.95, step=0.025):
    codes, uniques = pd.factorize(groups)
    codes = np.asarray(codes)
    n_groups = len(uniques)
    if n_groups == 0:
        return (0.0, low)
    probs = np.asarray(probs, dtype=np.float64)
    best = (0.0, low)
    for threshold in np.arange(low, high + 1e-9, step):
        score = _entity_f05_from_codes(codes, n_groups, labels, probs >= threshold)
        if score > best[0]:
            best = (score, float(threshold))
    return best
