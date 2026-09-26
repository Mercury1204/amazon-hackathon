import json
from pathlib import Path

from ber.threshold import FULL_CANDIDATES, decision_path, load_decision, save_decision

REPORT_NAME = "eval_full_candidates.json"


def _report_path(cfg):
    return Path(cfg.data_dir) / "reports" / REPORT_NAME


def tune_threshold(cfg, val_frac=0.2, workers=8, reuse=False, force=False):
    """Persist the inference decision (threshold + one-to-one) from a full-candidate sweep.

    The threshold `train` writes is swept on the 4:1 sampled split, which holds roughly four
    negatives per positive against ~135 candidates per entity at inference, so it lands far too
    low (0.70 versus 0.925 here). Only a sweep over the real candidate set is authoritative, and
    this is the step that records it; `predict` refuses to use a sampled-sweep threshold.
    """
    path = _report_path(cfg)
    if force or not path.exists():
        from ber.validation import evaluate_full_candidates

        report = evaluate_full_candidates(cfg, val_frac=val_frac, workers=workers, reuse=reuse)
    else:
        report = json.loads(path.read_text(encoding="utf-8"))
        print(f"[tune] reusing {path} (pass --force to re-run the sweep)", flush=True)

    missing = [k for k in ("chosen_method", "chosen_macro_f05") if k not in report]
    if missing:
        raise KeyError(
            f"{path} is missing {missing}; it was not written by "
            f"validation.evaluate_full_candidates. Re-run with --force."
        )
    method = report["chosen_method"]
    if method not in report or "threshold" not in report[method]:
        raise KeyError(f"{path}: no threshold recorded for chosen_method={method!r}")

    threshold = float(report[method]["threshold"])
    existing = load_decision(cfg) or {}
    payload = save_decision(
        cfg,
        threshold,
        use_one_to_one=(method == "one_to_one"),
        source=FULL_CANDIDATES,
        method=method,
        macro_f05=report["chosen_macro_f05"],
        ci95=report.get("ci95"),
        candidate_recall_ceiling=report.get("candidate_recall_ceiling"),
        val_s1=report.get("val_s1"),
        per_country=report.get("per_country"),
        # carried over from train; the threshold is only meaningful at this tree count
        best_iteration=existing.get("best_iteration"),
    )
    return {
        "threshold": payload["global"],
        "use_one_to_one": payload["use_one_to_one"],
        "method": method,
        "macro_f05": payload["macro_f05"],
        "candidate_recall_ceiling": payload["candidate_recall_ceiling"],
        "best_iteration": payload["best_iteration"],
        "decision_path": str(decision_path(cfg)),
    }
