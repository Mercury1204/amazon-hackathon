import argparse
from pathlib import Path

from ber.blocking import run_block
from ber.config import Config
from ber.prepare import run_prepare


def _default_config():
    package = Path(__file__).resolve().parents[2]
    primary = package / "config.json"
    fallback = package / "src" / "config.json"
    return str(primary if primary.exists() else fallback)


def _resolve_split(requested, command, natural, allowed):
    """Resolve --split for a command that works on one split at a time.

    The CLI default is "both", which is meaningless for these, so each command
    declares the split it acts on when unspecified plus the splits it accepts.
    """
    if requested == "both":
        return natural
    if requested not in allowed:
        raise SystemExit(
            f"ber {command}: --split {requested} is not supported (accepts "
            f"{', '.join(sorted(allowed))}). Use `ber.cli block --split {requested}` "
            f"to build that split."
        )
    return requested


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ber")
    parser.add_argument(
        "command",
        choices=["prepare", "block", "audit", "features", "train", "tune", "predict", "outputs", "evaluate", "validation", "loo", "all"],
    )
    parser.add_argument("--config", default=_default_config())
    parser.add_argument("--split", default="both", choices=["train", "test", "both"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--one-to-one", action="store_true", dest="one_to_one", default=None)
    parser.add_argument("--no-one-to-one", action="store_false", dest="one_to_one")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--reuse-predictions", action="store_true", dest="reuse_predictions")
    parser.add_argument("--force", action="store_true", help="tune: re-run the sweep instead of reusing the report")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--keep-merged", action="store_true", dest="keep_merged")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)

    if args.command in ("prepare", "all"):
        print(run_prepare(cfg))
    if args.command in ("block", "all"):
        splits = ("train", "test") if args.split == "both" else (args.split,)
        for split in splits:
            print(run_block(cfg, split))
    if args.command in ("audit", "all"):
        from ber.audit import run_audit

        print(run_audit(cfg, _resolve_split(args.split, "audit", "train", {"train"})))
    if args.command in ("features", "all"):
        from ber.features import run_features

        split = _resolve_split(args.split, "features", "train", {"train", "test"})
        if split == "test":
            from ber.pairs import write_inference_pairs

            print(write_inference_pairs(cfg, "test"))
        else:
            from ber.pairs import write_training_pairs

            print(write_training_pairs(cfg, "train"))
        print(
            run_features(
                cfg,
                split,
                workers=args.workers,
                combine=args.combine or split == "train",
                keep_merged=args.keep_merged,
            )
        )
    if args.command in ("train", "all"):
        from ber.train import train_model

        print(train_model(cfg))
    if args.command in ("tune",):
        from ber.tune import tune_threshold

        print(
            tune_threshold(
                cfg,
                workers=args.workers,
                reuse=args.reuse_predictions,
                force=args.force,
            )
        )
    if args.command in ("predict",):
        from ber.predict import run_predict

        print(
            run_predict(
                cfg,
                _resolve_split(args.split, "predict", "test", {"test"}),
                use_one_to_one=args.one_to_one,
                threshold_override=args.threshold,
                reuse_predictions=args.reuse_predictions,
            )
        )
    if args.command in ("outputs",):
        from ber.predict import run_outputs

        print(
            run_outputs(
                cfg,
                _resolve_split(args.split, "outputs", "test", {"test"}),
                use_one_to_one=args.one_to_one,
                threshold_override=args.threshold,
            )
        )
    if args.command in ("evaluate",):
        from ber.evaluate import evaluate_marks

        print(evaluate_marks(cfg, "train"))
    if args.command in ("validation",):
        from ber.validation import evaluate_full_candidates

        print(evaluate_full_candidates(cfg, workers=args.workers, reuse=args.reuse_predictions))
    if args.command in ("loo",):
        from ber.validation import evaluate_loo

        print(evaluate_loo(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
