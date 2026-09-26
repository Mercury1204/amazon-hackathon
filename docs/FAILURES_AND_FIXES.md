# Failures and Fixes

Every non-trivial failure encountered while building the pipeline, with root cause and resolution.
Kept for future reference so the same dead ends are not re-entered.

## F1 — Per-token Metaphone blocking exploded the temp directory
- **Symptom:** `OutOfMemoryException ... 75.6 GiB/75.6 GiB used`; the run never completed.
- **Cause:** per-token phonetic codes collide at scale (a single generic code was shared by >1.2M
  candidate records), so the pass-2 join produced billions of rows.
- **Fix:** removed the Metaphone blocking pass entirely (`RULES`/`DECISIONS` note: do not
  reintroduce). Recall was recovered with name-token prefix-5 and rare-token pair/triple passes.

## F2 — 64-bucket join loop took >3 hours
- **Symptom:** a per-bucket loop over 64 hash buckets ran for the full 3 h tool timeout.
- **Cause:** each bucket re-scanned the entire key tables (64 full scans) instead of one join.
- **Fix:** single DuckDB join with pass-specific block caps (`_valid_sql`); join size fell from
  819M to 38.5M rows in the first tuning step and candidates were produced in minutes.

## F3 — Pandas audit could not handle 98.8M candidate rows
- **Symptom:** `ber.cli audit` hung/OOMed after blocking grew to ~100M candidates.
- **Cause:** the audit materialized candidate tuples into Python sets/dicts.
- **Fix:** rewrote `run_audit` to be DuckDB end-to-end (read_parquet + joins/group-bys) with
  streaming; ~50 s for the full train audit. `audit_candidates` kept only as a pandas test reference.

## F4 — Parquet list columns came back as numpy arrays
- **Symptom:** `ValueError: The truth value of an array with more than one element is ambiguous`
  in `block_keys`.
- **Cause:** `ParquetFile.iter_batches(...).to_pandas()` returns list columns as `numpy.ndarray`,
  not Python lists; `if house and street:` then fails.
- **Fix:** coerce list columns with `list(x)` before use in `block_keys`.

## F5 — Feature pipeline needed a different metadata split
- **Symptom:** validation pairs lived in `pairs/valfull_pairs.parquet` but metadata/processed
  parquet are named for `train`.
- **Fix:** added an optional `meta_split` parameter to `run_features`/`_phase1_merged`.

## F6 — Prediction TSV aggregation OOM
- **Symptom:** `_duckdb.OutOfMemoryException (7.4 GiB/7.4 GiB used)` while writing
  `candidate_pairs.tsv`.
- **Cause:** a single `string_agg` over 250.6M pairs grouped into 1.73M rows exceeded the memory
  limit.
- **Fix:** bucket the candidate pairs by `hash(s1_id) % 64` once, aggregate each bucket separately
  (bounded memory), then byte-concatenate the parts with a single header.

## F7 — Empty match lists written as `""`, validator FAIL
- **Symptom:** `matched_entity_ids contains IDs without an S2-/S3- prefix: ""` (validator exit 1).
- **Cause:** DuckDB's CSV writer quotes empty strings as `""`; the validator then parses a literal
  empty ID.
- **Fix:** add `QUOTE ''` to both CSV `COPY` options so empty lists are written as true empty fields.

## F8 — `data` vs `DATA` path collision on Windows
- **Symptom:** intermediates landed under `DATA/` even though `config.json` said `data`.
- **Cause:** NTFS is case-insensitive and the challenge directory `DATA/` already existed.
- **Fix:** documented as expected behaviour in `AGENTS.md`; the packaged `src/config.json` uses
  `DATA` explicitly.

## F9 — Normalization stripped Indic combining marks
- **Symptom:** test `normalize_name("राम मार्केटिंग")` returned only consonants.
- **Cause:** the regex `[^\w\s]` does not treat combining marks (category Mn/Mc) as word characters.
- **Fix:** replaced regex cleanup with a `unicodedata`-based `_clean_chars` that keeps letters,
  digits, and marks.

## F10 — Suffix class overwritten by stacked suffixes
- **Symptom:** `"Private Ltd"` yielded suffix class `pvt` instead of `ltd`.
- **Cause:** the pop loop kept overwriting the class.
- **Fix:** keep the first (outermost) suffix class while still stripping all suffix tokens;
  `strip_legal_suffix` normalizes internally (uses `fold_name`).

## F11 — `grouped_split` was O(n·m) (43 h)
- **Symptom:** validation split took ~43 h on the full pair set.
- **Cause:** `np.isin(groups, val_groups)` over every group id.
- **Fix (by prior session):** `pd.factorize` + boolean group mask; ~12 s.

## F12 — Threshold tuned on the wrong distribution
- **Symptom:** deployed threshold 0.675, chosen on the 4:1 sample, produced many false merges when
  scored against the full candidate distribution.
- **Cause:** sampled negatives under-represent the confusable candidate mass present at inference.
- **Fix:** re-tuned on full candidates for held-out Source 1 groups -> **0.925**; outputs regenerated.

## F13 — Blocking recall below target
- **Symptom:** initial blocking recall 0.52, then 0.61, then 0.70, then 0.80.
- **Cause:** tight pass caps dropped common-token blocks; only exact/rare-token passes existed.
- **Fix:** added name-token prefix-5, street-token prefix-5, and rare-token pair/triple passes and
  widened caps; final train recall **0.8115**. India (0.727) remains the weak spot; raising recall
  further (embedding/LSH fuzzy blocking) is the top future work item.

## F14 — `repr()` emitted a double-quoted SQL string for the DuckDB spill directory
- **Symptom:** found while centralising the DuckDB session settings. A path containing an
  apostrophe produced `SET temp_directory="/tmp/o'neil/data/tmp"`, and DuckDB rejected it.
- **Cause:** Python's `repr()` switches to double quotes when the string contains a single quote.
  DuckDB follows the Postgres convention where **double quotes delimit identifiers, not strings**,
  so the statement failed to bind. The previous code interpolated the path into an f-string
  (`f"SET temp_directory='{tmp.as_posix()}'"`), which never had this problem.
- **Fix:** `ber/duck.py::_sql_str` always emits a single-quoted literal and doubles any embedded
  apostrophe. Pinned by `test_apostrophe_in_path_is_escaped`.

## F15 — DuckDB silently truncated a fractional memory limit
- **Symptom:** after F14's refactor, `memory_limit` came back as 7,945,689,498 bytes when 8,000,000,000
  was requested — a 0.68% shortfall on every stage, and the same for 10GB and 12GB.
- **Cause:** the formatter rendered `7.45GiB`, and DuckDB's settings parser requires a unit
  (`KB`/`MB`/`GB`/`TB` or `KiB`/.../`TiB`; a bare byte count is a parser error) **and keeps only one
  decimal place**, so `7.45GiB` was stored as `7.4GiB`.
- **Fix:** `ber/duck.py::format_bytes` now picks the most compact unit in which the value is exact,
  so the historical `8GB`/`10GB`/`12GB` and `50GiB`/`60GiB`/`80GiB` working points are reproduced
  byte-for-byte. `test_stage_sql_matches_pre_refactor_strings` pins every stage against the
  pre-refactor strings so a future formatter change cannot silently move a working point.
- **Note:** `current_setting('memory_limit')` also *reports* rounded, so verify the emitted SQL
  string, not the read-back value.

## F16 — `tune` and `outputs` were advertised CLI commands that did nothing
- **Symptom:** `ber.cli tune` and `ber.cli outputs` were both listed in `choices` (and in
  `AGENTS.md`) but no branch handled them, so they exited 0 having done nothing. A reader following
  the documented run order would believe the threshold had been tuned when it had not.
- **Cause:** they were planned in `docs/superpowers/plans/2026-09-25-er-pipeline.md` (Task 12/13) and
  wired into `choices`, but never implemented.
- **Fix:** `ber/tune.py` re-tunes on the full candidate distribution and persists the decision;
  `predict.run_outputs` writes the two TSVs from cached scores and verifies the output invariants.
  `--split` is now resolved explicitly per command, so `outputs --split train` fails instead of
  silently writing test files. Covered by `tests/test_tune.py`, `tests/test_decision.py` and
  `tests/test_end_to_end.py`.

## F17 — Output writer emitted an all-singleton submission when nothing cleared the threshold
- **Symptom:** found by the new end-to-end test. `predict` reported `pairs_above_threshold: 656` and
  wrote prediction parts, yet printed "no candidate pairs above threshold" and wrote
  `matching_results.tsv` with **every** row empty.
- **Cause:** an existence probe tested `Path(preds).is_dir()` where `preds` is the glob string
  `.../test_pred/*.parquet`. A glob is never a directory, so the probe was always false and the
  writer always took the empty-matches branch.
- **Why it mattered:** an all-singleton submission is *format-valid* — `utils/validate_submission.py`
  reported **PASS** on it — so the submission gate would not have caught it. Only an assertion on
  the match content exposed it.
- **Fix:** probe the prediction *directory* (`pred_dir.is_dir() and any(pred_dir.glob("*.parquet"))`).
  `tests/test_end_to_end.py` now asserts a non-empty match list survives the validator.

## F18 — `_write_tsv` crashed on any candidate bucket that hashed to no rows
- **Symptom:** `IOException: No files found that match the pattern .../test_cand_buckets/__b=0/*.parquet`.
- **Cause:** `_write_tsv` buckets candidates into 64 partitions and then reads each
  `__b={b}/*.parquet` glob in a Python loop. DuckDB errors on a glob matching no file, so any empty
  bucket aborted the run. Invisible on the real test set (1.73M entities populate all 64 buckets),
  fatal on any smaller split.
- **Fix:** when a bucket directory has no parquet files, emit its Source 1 rows with empty candidate
  lists directly instead of reading the glob.

## F19 — Inference used every built tree, not the tuned `best_iteration`
- **Symptom:** `booster.best_iteration` is `-1` for a booster loaded with `lgb.Booster(model_file=...)`,
  because LightGBM does not persist that attribute inside `lgbm.txt`. `predict` and `evaluate` passed
  it straight to `predict(num_iteration=...)`, so LightGBM used **all** trees — including any extra
  trees early stopping appended after the optimum — while the threshold had been tuned at
  `best_iteration`.
- **Cause:** `train` recorded `best_iteration` in `training_metrics.json`, which nothing read back.
- **Fix:** `train` stores it in the decision file, `tune` carries it across its rewrite, and
  `threshold.best_iteration` reads it (falling back to `-1`, i.e. current behaviour, for models
  trained before this change).
