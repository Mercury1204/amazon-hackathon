import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.cache import describe_staleness, fingerprint, read_context, read_meta
from ber.duck import connect
from ber.threshold import best_iteration, load_decision, resolve_decision


def _load_booster(cfg):
    models = Path(cfg.models_dir)
    booster = lgb.Booster(model_file=str(models / "lgbm.txt"))
    features = json.loads((models / "feature_list.json").read_text(encoding="utf-8"))
    decision = load_decision(cfg)
    if decision is None:
        raise FileNotFoundError(
            f"no decision file at {models / 'threshold.json'}. Run `ber.cli tune` to "
            f"choose the threshold on the full candidate set."
        )
    return booster, features, float(decision["global"]), decision


def _feature_sources(cfg, split):
    data = Path(cfg.data_dir)
    parts_dir = data / "tmp" / f"{split}_features"
    combined = data / "features" / f"{split}.parquet"
    if parts_dir.exists() and any(parts_dir.glob("*.parquet")):
        source = parts_dir
    elif combined.exists():
        source = combined
    else:
        raise FileNotFoundError(f"no feature parts or combined features for split={split}")

    meta = read_meta(source)
    if meta is None:
        print(
            f"[predict] warning: {source.name} has no provenance recorded; it was built "
            f"before fingerprints existed, so staleness cannot be checked. Re-run "
            f"`ber.cli features --split {split}` if anything upstream has changed since.",
            flush=True,
        )
    else:
        from ber.features import FEATURE_ORDER, feature_provenance

        # A valfull feature set borrows the train metadata, so recover which split it
        # was built from rather than assuming it matches its own name.
        meta_split = read_context(source).get("meta_split", split)
        inputs, config, code = feature_provenance(cfg, split, meta_split)
        if meta.get("fingerprint") != fingerprint(inputs, config, code):
            reason = describe_staleness(source, inputs, config, code)
            raise ValueError(
                f"{source.name} is stale for split={split}: {reason}. Re-run "
                f"`ber.cli features --split {split} --workers 8` before predicting — "
                f"scoring against mismatched features would silently corrupt the "
                f"submission."
            )
        if list(meta.get("config", {}).get("feature_order", FEATURE_ORDER)) != list(FEATURE_ORDER):
            raise ValueError(
                f"{source.name} was built with a different FEATURE_ORDER than the current "
                f"code; re-run `ber.cli features --split {split}`."
            )
    return sorted(parts_dir.glob("*.parquet")) if source == parts_dir else [combined]


def predict_parts(cfg, split, booster, features, threshold, out_dir=None, num_iteration=None):
    data = Path(cfg.data_dir)
    pred_dir = Path(out_dir) if out_dir is not None else data / "tmp" / f"{split}_pred"
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    sources = _feature_sources(cfg, split)
    if num_iteration is None:
        num_iteration = booster.best_iteration
    total = 0
    for i, part in enumerate(sources):
        writer = None
        part_rows = 0
        for batch in pq.ParquetFile(part).iter_batches(
            batch_size=2_000_000, columns=features + ["s1_id", "cand_id"]
        ):
            frame = batch.to_pandas()
            probs = booster.predict(frame[features], num_iteration=num_iteration)
            mask = probs >= threshold
            if mask.any():
                out = pd.DataFrame(
                    {
                        "s1_id": frame["s1_id"].to_numpy()[mask],
                        "cand_id": frame["cand_id"].to_numpy()[mask],
                        "prob": probs[mask].astype(np.float32),
                    }
                )
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(pred_dir / f"part_{i:04d}.parquet", table.schema)
                writer.write_table(table)
                part_rows += len(out)
        if writer is not None:
            writer.close()
        total += part_rows
        print(f"[predict] part {i + 1}/{len(sources)}: {part_rows:,} matches", flush=True)
    return pred_dir, total


def _write_tsv(cfg, split, use_one_to_one, n_buckets=64):
    data = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = (data / "pairs" / f"{split}_pairs.parquet").as_posix()
    pred_dir = data / "tmp" / f"{split}_pred"
    preds = (pred_dir / "*.parquet").as_posix()
    s1_parquet = (data / "processed" / f"{split}_source1.parquet").as_posix()
    matching = out_dir / "matching_results.tsv"
    candidate = out_dir / "candidate_pairs.tsv"

    con = connect(cfg, memory=8e9, temp=60 * 1024 ** 3)
    con.execute(
        f"CREATE TEMP TABLE s1list AS SELECT entity_id AS source1_entity_id FROM read_parquet('{s1_parquet}')"
    )

    # Nothing above threshold is a legitimate outcome (an over-conservative threshold on a
    # small or noisy split), and predict_parts then writes no part files at all, so the
    # prediction glob would match nothing. Probe the directory, not the glob.
    have_preds = pred_dir.is_dir() and any(pred_dir.glob("*.parquet"))

    if not have_preds:
        con.execute("CREATE TEMP TABLE matches (s1_id VARCHAR, cand_id VARCHAR, prob DOUBLE)")
        print("[predict] no candidate pairs above threshold; writing empty match lists", flush=True)
    elif use_one_to_one:
        con.execute(
            f"""
            CREATE TEMP TABLE matches AS
            SELECT s1_id, cand_id FROM (
                SELECT s1_id, cand_id,
                       row_number() OVER (PARTITION BY cand_id ORDER BY prob DESC, s1_id ASC) AS rn
                FROM read_parquet('{preds}')
            ) WHERE rn = 1
            """
        )
    else:
        con.execute(
            f"CREATE TEMP TABLE matches AS SELECT DISTINCT s1_id, cand_id FROM read_parquet('{preds}')"
        )

    con.execute(
        f"""
        COPY (
            WITH agg AS (
                SELECT s1_id, string_agg(cand_id, ',' ORDER BY cand_id) AS ids
                FROM matches GROUP BY s1_id
            )
            SELECT s.source1_entity_id, coalesce(a.ids, '') AS matched_entity_ids
            FROM s1list s LEFT JOIN agg a ON a.s1_id = s.source1_entity_id
            ORDER BY s.source1_entity_id
        ) TO '{matching.as_posix()}' (FORMAT CSV, HEADER, DELIM '\t', QUOTE '')
        """
    )
    print("[predict] wrote matching_results.tsv", flush=True)

    bucket_dir = data / "tmp" / f"{split}_cand_buckets"
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    con.execute(
        f"COPY (SELECT s1_id, cand_id, hash(s1_id) % {int(n_buckets)} AS __b FROM read_parquet('{pairs}')) "
        f"TO '{bucket_dir.as_posix()}' (FORMAT PARQUET, PARTITION_BY (__b))"
    )
    parts_dir = data / "tmp" / f"{split}_cand_tsv"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_files = []
    for b in range(n_buckets):
        part = parts_dir / f"part_{b:03d}.tsv"
        bucket = bucket_dir / f"__b={b}"
        if not bucket.is_dir() or not any(bucket.glob("*.parquet")):
            # Nothing hashed into this bucket. Every Source 1 entity landing here has
            # no candidates, so emit its empty row directly rather than reading a glob
            # that matches no file.
            con.execute(
                f"""
                COPY (
                    SELECT source1_entity_id, '' AS candidate_entity_ids
                    FROM s1list WHERE hash(source1_entity_id) % {int(n_buckets)} = {b}
                ) TO '{part.as_posix()}' (FORMAT CSV, HEADER false, DELIM '\t', QUOTE '')
                """
            )
            part_files.append(part)
            continue
        glob = bucket.as_posix() + "/*.parquet"
        con.execute(
            f"""
            COPY (
                WITH cand AS (SELECT s1_id, cand_id FROM read_parquet('{glob}')),
                     agg AS (
                         SELECT s1_id, string_agg(cand_id, ',' ORDER BY cand_id) AS ids
                         FROM cand GROUP BY s1_id
                     )
                SELECT s.source1_entity_id, coalesce(a.ids, '') AS candidate_entity_ids
                FROM (SELECT source1_entity_id FROM s1list WHERE hash(source1_entity_id) % {int(n_buckets)} = {b}) s
                LEFT JOIN agg a ON a.s1_id = s.source1_entity_id
            ) TO '{part.as_posix()}' (FORMAT CSV, HEADER false, DELIM '\t', QUOTE '')
            """
        )
        part_files.append(part)
    with open(candidate, "wb") as fout:
        fout.write(b"source1_entity_id\tcandidate_entity_ids\n")
        for part in part_files:
            fout.write(part.read_bytes())
    print("[predict] wrote candidate_pairs.tsv", flush=True)

    match_dist = con.execute(
        """
        WITH agg AS (SELECT s1_id, count(*) AS n FROM matches GROUP BY s1_id)
        SELECT coalesce(a.n, 0) AS n_matches, count(*) AS entities
        FROM s1list s LEFT JOIN agg a ON a.s1_id = s.source1_entity_id
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    n_matching = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{matching.as_posix()}', delim='\t', header=true)"
    ).fetchone()[0]
    n_candidate = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{candidate.as_posix()}', delim='\t', header=true)"
    ).fetchone()[0]
    con.close()
    return {
        "matching_rows": int(n_matching),
        "candidate_rows": int(n_candidate),
        "matching_path": matching.as_posix(),
        "candidate_path": candidate.as_posix(),
        "match_count_distribution": {int(k): int(v) for k, v in match_dist},
    }


def check_outputs(cfg, split):
    """Bounded invariant checks on the two written TSVs.

    Covers the scorer-rejection rules that are cheap to verify: one row per test
    Source 1 entity, no duplicate S1 rows, no repeated ID inside a list, and
    S2-/S3- prefixes only. Matches-subset-of-candidates is not re-derived here
    because both files are built from the same pairs parquet, so it holds by
    construction; `utils/validate_submission.py` re-checks it before submission.
    """
    data = Path(cfg.data_dir)
    matching = (Path(cfg.output_dir) / "matching_results.tsv").as_posix()
    candidate = (Path(cfg.output_dir) / "candidate_pairs.tsv").as_posix()
    s1_parquet = (data / "processed" / f"{split}_source1.parquet").as_posix()

    con = connect(cfg)
    expected = con.execute(f"SELECT COUNT(*) FROM read_parquet('{s1_parquet}')").fetchone()[0]
    read_matching = (
        f"read_csv('{matching}', delim='\\t', header=true, all_varchar=true)"
    )
    rows, distinct_s1, singletons = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT source1_entity_id), "
        f"COUNT(*) FILTER (WHERE matched_entity_ids = '') FROM {read_matching}"
    ).fetchone()
    dup_lists, bad_prefix, self_matches, bad_ids = con.execute(
        f"""
        SELECT COUNT(*) FILTER (WHERE n_ids <> n_uniq),
               COALESCE(SUM(n_bad_prefix), 0),
               COALESCE(SUM(n_self), 0),
               COALESCE(SUM(n_empty), 0)
        FROM (
            SELECT source1_entity_id,
                   COUNT(*) AS n_ids,
                   COUNT(DISTINCT mid) AS n_uniq,
                   COUNT(*) FILTER (WHERE mid = '') AS n_empty,
                   COUNT(*) FILTER (WHERE mid NOT LIKE 'S2-%' AND mid NOT LIKE 'S3-%') AS n_bad_prefix,
                   COUNT(*) FILTER (WHERE mid LIKE 'S1-%') AS n_self
            FROM (
                SELECT source1_entity_id,
                       UNNEST(string_split(matched_entity_ids, ',')) AS mid
                FROM {read_matching} WHERE matched_entity_ids <> ''
            )
            GROUP BY source1_entity_id
        )
        """
    ).fetchone()
    cand_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{candidate}', delim='\\t', header=true, all_varchar=true)"
    ).fetchone()[0]
    con.close()

    problems = []
    if rows != expected:
        problems.append(f"matching_results.tsv has {rows} rows, expected {expected} (one per test S1)")
    if distinct_s1 != rows:
        problems.append(f"matching_results.tsv has {rows - distinct_s1} duplicate source1_entity_id rows")
    if dup_lists:
        problems.append(f"{dup_lists} row(s) repeat an ID inside matched_entity_ids")
    if bad_prefix:
        problems.append(f"{bad_prefix} ID(s) lack an S2-/S3- prefix")
    if self_matches:
        problems.append(f"{self_matches} Source-1 self-match(es) in matched_entity_ids")
    if bad_ids:
        problems.append(f"{bad_ids} empty ID(s) inside a non-empty matched_entity_ids")
    if cand_rows != expected:
        problems.append(f"candidate_pairs.tsv has {cand_rows} rows, expected {expected}")
    return {
        "ok": not problems,
        "problems": problems,
        "expected_rows": int(expected),
        "matching_rows": int(rows),
        "candidate_rows": int(cand_rows),
        "singletons": int(singletons),
        "non_singleton_rows": int(rows - singletons),
    }


def run_outputs(cfg, split="test", use_one_to_one=None, threshold_override=None):
    """Write the submission TSVs from cached predictions, without re-scoring.

    Separating this from `run_predict` lets the two files be regenerated after a
    threshold or one-to-one change without re-scoring every candidate pair.
    """
    threshold, one_to_one, source = resolve_decision(
        cfg, threshold_override=threshold_override, one_to_one_override=use_one_to_one
    )
    pred_dir = Path(cfg.data_dir) / "tmp" / f"{split}_pred"
    if not pred_dir.exists() or not any(pred_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"no cached predictions in {pred_dir}. Run `ber.cli predict --split {split}` first."
        )
    print(
        f"[outputs] split={split} threshold={threshold} (source={source}) "
        f"one_to_one={one_to_one} from {pred_dir}",
        flush=True,
    )
    stats = _write_tsv(cfg, split, one_to_one)
    stats["checks"] = check_outputs(cfg, split)
    stats["use_one_to_one"] = one_to_one
    stats["threshold"] = threshold
    stats["threshold_source"] = source
    if not stats["checks"]["ok"]:
        raise ValueError(
            "output invariants violated: " + "; ".join(stats["checks"]["problems"])
        )
    return stats


def run_predict(cfg, split="test", use_one_to_one=None, threshold_override=None, reuse_predictions=False):
    # Check feature provenance first: stale features invalidate the whole run, and this
    # is the cheapest check, so report it before touching the model or the decision.
    _feature_sources(cfg, split)
    threshold, use_one_to_one, source = resolve_decision(
        cfg, threshold_override=threshold_override, one_to_one_override=use_one_to_one
    )
    booster, features, _, _ = _load_booster(cfg)
    print(
        f"[predict] model={booster.num_trees()} trees, threshold={threshold} (source={source})",
        flush=True,
    )
    data = Path(cfg.data_dir)
    pred_dir = data / "tmp" / f"{split}_pred"
    if reuse_predictions and pred_dir.exists() and any(pred_dir.glob("*.parquet")):
        print("[predict] reusing existing prediction parts", flush=True)
        n_pred = None
    else:
        n_iter = best_iteration(cfg)
        pred_dir, n_pred = predict_parts(
            cfg, split, booster, features, threshold, num_iteration=n_iter
        )
        if n_iter != booster.best_iteration:
            print(f"[predict] using best_iteration={n_iter} of {booster.num_trees()} trees", flush=True)
    stats = _write_tsv(cfg, split, use_one_to_one)
    stats["pairs_above_threshold"] = int(n_pred) if n_pred is not None else "reused"
    stats["use_one_to_one"] = bool(use_one_to_one)
    stats["threshold"] = threshold
    stats["threshold_source"] = source
    return stats
