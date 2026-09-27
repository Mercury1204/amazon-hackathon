# Project Log — Business Entity Resolution

Chronological record of what was run, on what data, with measured numbers and timings.
All commands run from the repository root with `PYTHONPATH=code/business_entity_resolution/src`
and the Python 3.12 venv (`.venv\Scripts\python.exe`).

## 2026-09-25 / 26 — environment and tooling

| Step | Command | Result |
|---|---|---|
| Create venv | `uv venv .venv --python 3.12` | Python 3.12.14 |
| Install deps | `uv pip install pandas pyarrow numpy scikit-learn lightgbm xgboost rapidfuzz duckdb indic-transliteration jellyfish pytest` | pinned in `requirements.txt` |
| Dataset EDA | `.venv\Scripts\python.exe tools\eda_dataset.py` | `tools/eda_stats.json`, `DATA/student_resource/dataset/DATASET.md` |
| GBDT benchmark | `.venv\Scripts\python.exe tools\bench_gbdt.py` | `tools/bench_results.json` |
| Graphify init | graphify skill on project docs | `graphify-out/` (graph.html, GRAPH_REPORT.md, graph.json) |
| Graphify refresh | graphify skill over docs + code (53 files) | `graphify-out/` 265 nodes / 563 edges, 40.5x token reduction |

## Pipeline runs (measured)

| Stage | Split | Command | Output | Notes |
|---|---|---|---|---|
| Prepare | both | `ber.cli prepare` | `DATA/processed/*.parquet` | 24.2M rows, ~40 min; counts match EDA exactly |
| Block | train | `ber.cli block --split train` | 304,759,423 candidates | keys cached in `DATA/keys/` |
| Block | test | `ber.cli block --split test` | 250,607,135 candidates | |
| Audit | train | `ber.cli audit --split train` | recall 0.8115 (US 0.868 / India 0.727), reduction 29.5 | ~50 s, DuckDB |
| Features | train | `ber.cli features --split train --workers 8 --combine` | 30,494,378 × 36 (`DATA/features/train.parquet`) | 4:1 sampled negatives |
| Train | train | `ber.cli train` | `models/lgbm.txt` (610 trees) | val macro F0.5 0.9803 (4:1, optimistic) |
| Features | test | `ber.cli features --split test --workers 8` | 250,607,135 rows in 16 parts | parallel, ~ minutes |
| Predict | test | `ber.cli predict --split test --one-to-one --threshold 0.925` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | 1,732,544 rows each |
| Evaluate (4:1) | train | `ber.cli evaluate` | `DATA/reports/eval_marks.json` | threshold-only 0.9803 / one-to-one 0.9807 |
| Validation | train | `ber.cli validation --workers 8` | `DATA/reports/eval_full_candidates.json` | **0.8488** on full candidates, held-out S1 |
| LOO | train | `ber.cli loo` | `DATA/reports/eval_loo.json` | unseen-country proxy: 0.668 / 0.804 |
| Validate | test | `validate_submission.py ...` | `PASS` | format gate |

## Final submission

- `output/matching_results.tsv` — 1,732,544 rows (200,982 empty / 1,533,562 non-empty).
- `output/candidate_pairs.tsv` — 1,732,544 rows (554 empty).
- 5,071,867 candidate pairs above threshold 0.925.
- Package staged at `dist/AA.._submission/`; zip `dist/AA..__submission.zip`.

## Corrections made during the run

- Threshold was first tuned on the 4:1 sampled split (0.675) and later correctly re-tuned on the
  full candidate distribution to **0.925**; all submitted outputs use 0.925.
- The 4:1 number (0.98) is optimistic and must not be quoted as the leaderboard score; see
  `RESULTS.md`.

## 2026-09-26 — Kaggle portability: env-var paths and DuckDB runtime knobs

No pipeline stage was re-run; this is a refactor plus hardening. Motivation: RULES §6 makes Kaggle
the only place heavy runs happen, but paths were repo-root-relative and 22 DuckDB limits were
hard-coded for the local 23 GB Windows box.

| Step | Command | Result |
|---|---|---|
| Ignore rules | — | `kaggle.json`, `*.kaggle.json`, `credentials.json` (the name Kaggle CLI 2.x actually reads), `.kaggle/`, `.netrc`, `netrc`, `.env`, `.env.*`, `*.pem`, `*.key` added to `.gitignore` **and** `.graphifyignore`; `DATA/kaggle_upload/` staging dir ignored |
| Path overrides | — | `ber/config.py`: `BER_DATASET_DIR` (aliases `DATA_DIR`, `BER_DATA_DIR`, auto-detect under `/kaggle/input`), `BER_ARTIFACT_DIR`, `BER_MODELS_DIR`, `BER_OUTPUT_DIR`, `BER_ROOT`. Env beats config; `~` expanded; relative paths resolve against `root`; unknown keys rejected (D13); `validate=True` fails fast on a bad `dataset_dir` |
| Runtime knobs | — | `ber/duck.py::connect` centralises `memory_limit` / `threads` / `temp_directory` / `max_temp_directory_size`; all 22 hard-coded sites across `audit`, `blocking`, `features`, `pairs`, `predict`, `validation` migrated. Knobs: `BER_DUCK_MEMORY_LIMIT` (8GB), `BER_DUCK_THREADS` (8), `BER_DUCK_MAX_TEMP` (50GiB), `BER_DUCK_TMP_DIR` |
| Tests | `pytest -q` | **81 passed** (was 31). New: `tests/test_config.py` (23), `tests/test_duck.py` (27) |
| Equivalence proof | — | Every stage emits byte-identical SQL to the pre-refactor strings (`8GB`/`10GB`/`12GB`, `50GiB`/`60GiB`/`80GiB`), pinned by `test_stage_sql_matches_pre_refactor_strings` |

Two bugs were found and fixed during this work, both introduced by the refactor itself and caught
by testing against real DuckDB rather than a stub — see F14 (`repr()` double-quoted a path, which
binds as an identifier) and F15 (DuckDB truncates a fractional `memory_limit` to one decimal, so
`7.45GiB` silently became `7.4GiB`).

Not verified here: `run_block`, `run_prepare`, `write_training_pairs`, `write_inference_pairs`,
`run_predict` and `evaluate_full_candidates` need the full 2.3 GiB dataset and more disk than this
machine has, and `train.py`/`predict.py` cannot import on this host because LightGBM needs
`libomp`. The refactored `run_audit`, `generate_candidates` and `run_features` paths *are* covered
end-to-end by real DuckDB in `test_audit.py`, `test_blocking.py` and `test_parallel_features.py`.

Dataset integrity re-verified before planning the Kaggle upload: all 7 row counts and file sizes
match `DATASET.md` exactly, and UTF-8 is intact (Devanagari and French-accented names round-trip).

## 2026-09-27 — `tune` and `outputs` implemented; decision provenance enforced

| Step | Command | Result |
|---|---|---|
| Ignore rules | — | `kaggle.json`, `*.kaggle.json`, `credentials.json` (the name Kaggle CLI 2.x actually reads), `.kaggle/`, `.netrc`, `netrc`, `.env`, `.env.*`, `*.pem`, `*.key` added to `.gitignore` **and** `.graphifyignore`; `DATA/kaggle_upload/` staging dir ignored |
| Path overrides | — | `ber/config.py`: `BER_DATASET_DIR` (aliases `DATA_DIR`, `BER_DATA_DIR`, auto-detect under `/kaggle/input`), `BER_ARTIFACT_DIR`, `BER_MODELS_DIR`, `BER_OUTPUT_DIR`, `BER_ROOT`. Env beats config; `~` expanded; relative paths resolve against `root`; unknown keys rejected (D13); `validate=True` fails fast on a bad `dataset_dir` |
| Runtime knobs | — | `ber/duck.py::connect` centralises `memory_limit` / `threads` / `temp_directory` / `max_temp_directory_size`; all 22 hard-coded sites across `audit`, `blocking`, `features`, `pairs`, `predict`, `validation` migrated. Knobs: `BER_DUCK_MEMORY_LIMIT` (8GB), `BER_DUCK_THREADS` (8), `BER_DUCK_MAX_TEMP` (50GiB), `BER_DUCK_TMP_DIR` |
| `tune` | `ber.cli tune --workers 8` | Re-tunes on the full candidate distribution and rewrites `models/threshold.json` with `source: full_candidates`, the winning method, the macro F0.5, and `best_iteration`. Reuses `eval_full_candidates.json` unless `--force` |
| `outputs` | `ber.cli outputs --split test` | Rewrites both TSVs from cached scores and verifies the output invariants; non-zero exit on violation |
| Guardrail | `ber.cli predict --split test` | Now **refuses** a `sampled_4to1` threshold (F16/D14); `--threshold` is the explicit opt-in |
| Split safety | `ber.cli outputs --split train` | Rejected instead of silently writing test files |
| Tests | `pytest -q` | **101 passed** (was 31). New: `test_config.py` (23), `test_duck.py` (27), `test_decision.py` (10), `test_tune.py` (7), `test_end_to_end.py` (3) |
| Equivalence proof | — | Every stage emits byte-identical SQL to the pre-refactor strings (`8GB`/`10GB`/`12GB`, `50GiB`/`60GiB`/`80GiB`), pinned by `test_stage_sql_matches_pre_refactor_strings` |
| End-to-end gate | `pytest -m slow` | Generates a 30/18-entity fixture, shells out to the real CLI for the whole run order, and gates on `utils/validate_submission.py` — **PASS**, including `--check-ids` |

Four bugs were found and fixed during this work; F14/F15 came from the DuckDB refactor and F16–F19
from wiring the two commands. F17 is the notable one: an existence probe tested a *glob* with
`Path.is_dir()`, so `predict` wrote an all-singleton `matching_results.tsv` that the official
validator still reported **PASS** on. Only asserting on the match content caught it.

Not verified here: `run_block`, `run_prepare`, `write_training_pairs` and
`evaluate_full_candidates` were exercised only on the generated fixture, never against the full
2.3 GiB dataset, because this machine has ~20 GiB free. The refactored `run_audit`,
`generate_candidates` and `run_features` paths are covered end-to-end by real DuckDB in
`test_audit.py`, `test_blocking.py` and `test_parallel_features.py`.

## 2026-09-27 — degree-feature skew and cache provenance

No production numbers were regenerated; both changes invalidate the current model.

| Step | Change | Result |
|---|---|---|
| Degree fix | `features.candidate_degree_tables` | `s1_degree` / `cand_degree` now derive from `candidates/{meta_split}_candidates.parquet` instead of the split's pair file, closing a train/serve skew where the same feature meant ~5-20 at training and up to 200 at inference (F20/D16). Cached, since each is a pure function of the candidate set. Phase 1 also stops aggregating the 250.6M-row test pair table |
| Cache provenance | `ber/cache.py` | Every artifact gets a `<artifact>.meta.json` fingerprint over its input-file identity, the relevant config keys, and the source of the computing modules. Producers recompute on mismatch; `predict` refuses stale feature parts (F21/D17) |
| Ordering fix | `predict.run_predict` | Feature-provenance check now runs before the model load and the decision lookup, so the actionable "re-run `features`" error fires first |
| Tests | `pytest -q` | **120 passed** (was 101). New: `test_cache.py` (15), `test_feature_provenance.py` (4) |

Two of my own bugs surfaced while testing this and were fixed: the degree tables were written without
creating `reports/`, and `_feature_sources` could not know that a `valfull` feature set borrows the
train metadata, so it computed a different fingerprint than the producer. The artifact now records
its `meta_split` in a `context` field excluded from the fingerprint.

`test_parallel_features.py` gained a candidates file in its fixture, since the degree features now
require one.

**Outstanding:** the committed `models/lgbm.txt`, `models/threshold.json` and the reported 0.8488 were
all produced with the skewed degree features. A full recompute — `features` -> `train` -> `tune` ->
`predict` -> `outputs` — is required before any number here is quoted again. That recompute is best
done as the first Kaggle run, which is why these two fixes were made before the data upload
completed rather than after.

## 2026-09-27 — Kaggle N1 (`prepare`) completed on the Kaggle base image

| Step | Result |
|---|---|
| Dataset mount | `/kaggle/input/datasets/mercury147/amz-er-2026-raw` (private, read-only), 3 levels deep. Auto-detect had globbed at fixed depth 1-2 and missed it; fixed in `51d922c` |
| Code | `git clone` + `git checkout 51d922c` (pinned SHA, so the notebook records exactly which commit ran) |
| Environment | **Deviation:** ran on the Kaggle base image, not `requirements.txt` — pandas 2.3.3, numpy 2.0.2, lightgbm 4.6.0, duckdb 1.3.2, pyarrow 25.0.1, rapidfuzz 3.14.6, plus `indic-transliteration==2.3.82` (the only package the base image lacked) |
| Verified | 126/126 tests pass on that exact stack; all 14 DuckDB SQL constructs the pipeline uses verified working on duckdb 1.3.2 (the pin is 1.5.5) |
| `prepare` | 24,206,512 rows written to `artifacts/processed/` = 4.3 GB. **All 7 row counts match `DATASET.md` exactly** — dataset upload confirmed intact |

**On the version deviation.** `RULES.md §6.5` asks for pinned versions, so this is recorded rather than silently accepted. It was
not a shortcut: an earlier attempt to force the pins produced a mangled numpy (`cannot import name '_center'` — 2.5.3 `.py`
files layered over 2.0.2 `.so` binaries) that could not be repaired in place. The base-image stack was measured to be
compatible rather than assumed to be, and the Kaggle run's numbers are therefore **not bit-comparable** to the local
0.8488. Re-running the local stack is the way to restore strict comparability.

Also trimmed `requirements.txt` to the packages `src/ber/` actually imports. `jellyfish` was pinned but imported nowhere
in the repository and is removed. `xgboost` is imported only by `tools/bench_gbdt.py` (lazily) and is demoted to a
comment, because it pulls `nvidia-nccl-cu13` — a ~305 MB GPU library that is dead weight on a CPU notebook and a real
cost against a 20 GB `/kaggle/working`. `RULES.md §3.7` keeps XGBoost as the documented drop-in, so it stays installable.

**Disk budget for the remaining stages.** `/kaggle/working` is 20 GB, of which 4.3 GB is `processed/`. Two structural
facts make the cascade necessary rather than optional: `keys/` is read only by `run_block` itself, so it is disposable
once candidates exist; and `processed/` is written only by `prepare`, so it can be promoted to a versioned dataset and
re-mounted read-only. DuckDB is configured with a 6 GB memory ceiling and 18 GiB spill cap, which is deliberately
inverted from the local box: Kaggle offers ~30 GB RAM against 20 GB of disk, so a *lower* memory ceiling would force
*more* spilling to the scarcer resource.

**Disk topology correction.** `/kaggle/working` and `/kaggle/lib` are the *same* 20 GB
filesystem (`/dev/loop2`), so Kaggle's own libraries consume part of the budget that looked
like 20 GB free; the real figure was ~13 GB. The `57 GB` quoted earlier was the root overlay
(8.0 TB, 1.1 TB available), not the working volume. The correct split is persistent artifacts
on `/kaggle/working` and DuckDB scratch on the overlay via `BER_DUCK_TMP_DIR`, which keeps
spill off the crowded volume. Two `block` attempts were lost to this before it was measured.
