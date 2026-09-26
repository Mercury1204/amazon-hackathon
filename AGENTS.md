# AGENTS.md — working notes for this repository

Amazon ML Challenge 2026: Business Entity Resolution. Read `RULES.md` (binding rules), `PROBLEM_STATEMENT.md` (full spec), and `DATA/student_resource/dataset/DATASET.md` (measured data facts) before changing code.

Keep documentation and logs current as you work:
- `docs/PROJECT_LOG.md` — chronological run log (commands, timings, numbers).
- `docs/DECISIONS.md` — architecture decisions and reasoning.
- `docs/FAILURES_AND_FIXES.md` — failures and resolutions; add an entry for every non-trivial bug.
- `docs/RESULTS.md` — metrics; update when a new believable evaluation is produced.

## Environment

- ML code runs on **Python 3.12** via the project venv: `.venv\Scripts\python.exe` (created with `uv`, deps in `code/business_entity_resolution/requirements.txt`).
- Graphify runs on system Python 3.14 (`python`, not the venv). Keep the environments separate.
- Harnesses: `opencode` and `mcode` are both agent harnesses used on this repo; keep instructions harness-agnostic.
- Tests: `.venv\Scripts\python.exe -m pytest -q` from the repo root (81 tests). `conftest.py` puts `src/` on `sys.path`.
- CLI: set `PYTHONPATH=code/business_entity_resolution/src` then run `python -m ber.cli <command> --config code/business_entity_resolution/config.json`. Commands: `prepare`, `block`, `audit`, `features`, `train`, `tune`, `predict`, `outputs`, `all`.

## Environment variables (override `config.json`)

Resolved in `ber.config.Config.load`; **env beats the config file**, and relative
config paths resolve against `root`. `Config.load` rejects unknown config keys.
`Config.load(..., validate=True)` additionally fails fast when `dataset_dir` has
neither `train/` nor `test/`.

| Variable | Overrides | Notes |
|---|---|---|
| `BER_DATASET_DIR` | `dataset_dir` | Challenge input, containing `train/` and `test/`. Read-only use only, so a Kaggle `/kaggle/input/...` mount is safe. Aliases: `DATA_DIR`, then `BER_DATA_DIR` (that order). Auto-detected under `/kaggle/input` when unset. |
| `BER_ARTIFACT_DIR` | `data_dir` | Intermediates root (`processed/`, `keys/`, `candidates/`, `pairs/`, `features/`, `reports/`, `tmp/`). Must be writable — on Kaggle use `/kaggle/working`, **not** `/kaggle/input`. |
| `BER_MODELS_DIR` | `models_dir` | |
| `BER_OUTPUT_DIR` | `output_dir` | |
| `BER_ROOT` | `root` | Base for relative paths. Defaults to the current directory. |
| `BER_DUCK_MEMORY_LIMIT` | `duck_memory_limit` | Default `8GB`. Accepts `8GB`/`50GiB`/a byte count. |
| `BER_DUCK_THREADS` | `duck_threads` | Default `8`. |
| `BER_DUCK_MAX_TEMP` | `duck_max_temp` | Default `50GiB`. **Must fit the volume holding the scratch dir** — Kaggle gives ~20 GB in `/kaggle/working`. |
| `BER_DUCK_TMP_DIR` | `duck_tmp_dir` | Spill root; scratch is `<dir>/tmp`. Defaults to the artifact dir. |

Do **not** set `BER_ARTIFACT_DIR` to the same path as `BER_DATASET_DIR`: on a
case-insensitive filesystem `data` and `DATA` collide (see F8). The two
directories are deliberately distinct — the primary `config.json` uses lowercase
`data`, the packaged `src/config.json` uses uppercase `DATA`.

## Pipeline data contracts (all on disk, parquet/JSON)

- `DATA/processed/{split}_source{1,2,3}.parquet` — normalized records from `prepare`. Key columns: `entity_id`, `name_norm`, `name_fold`, `name_roman`, `name_tokens`, `name_idf_tokens`, `name_script`, `name_stripped`, `suffix_class`, `addr_norm`, `addr_raw_missing`, `house_no`, `street_tokens`, `postal`, `state_key`, `landmark_flag`.
- `DATA/processed/train_ground_truth.parquet` — official columns `source1_entity_id`, `matched_entity_ids`.
- `DATA/reports/{split}_token_idf.json` — `{"n", "min_idf", "idf": {token: idf}}` for every token (used for pass-3 rarity and top-4 rarest ranking).
- `DATA/keys/{split}_s1_keys.parquet`, `{split}_source{2,3}_keys.parquet` — blocking keys `(entity_id, pass_id, key, block_score)`.
- `DATA/candidates/{split}_candidates.parquet` — `(s1_id, cand_id, pass_id, block_score, is_s2)`, one row per final candidate pair after dedup and the per-S1 cap. This is exactly the set the matcher scores.

## Blocking contracts

- Passes: 1 exact normalized name; 3 rare-token exact; 4 `house_no|street[0][:5]`; 5 `postal|first name token`; 6 fallback `state|name[:3]`; 7 name-token prefix-5; 8 street-token prefix-5; 9 token pairs over the 4 rarest tokens; 10 token triples over the 3 rarest tokens.
- Per-pass candidate-block caps live in `config.json` under `pass_caps` (keys are strings). Per-S1 cap is `cap` (200). `max_block` is not used by the current join; `_valid_sql` uses the pass caps.
- Tuning loop: edit `pass_caps` → `ber.cli block --split <train|test>` (keys are cached; deleting `DATA/keys/` forces regeneration, ~20 min for train) → `ber.cli audit --split train` (~50 s) → read recall. Current operating point: train recall **0.8115** (US 0.868, India 0.727), 304.8M candidates, reduction 29.5.
- Never run `block` and `audit` concurrently with another heavy job (RAM/disk contention; DuckDB uses up to 12 GB and a 50-60 GiB temp directory).

## Audit contract

- Entry point: `ber.audit.run_audit(cfg, split)`, DuckDB end-to-end (never materializes candidate tuples). `audit_candidates()` is a small-scale pandas reference used only by tests; do not delete it and do not overwrite `run_audit`.
- Report path: `DATA/reports/{split}_blocking_audit.json`. Keys: `recall`, `recall_by_country`, `reduction_ratio`, `candidates_per_s1_mean`, `singleton_candidates_mean`, `per_pass_recall`, `truth_pairs`, `found_pairs`.
- `test_run_audit_matches_pandas_reference` asserts the DuckDB path equals the pandas reference field-by-field.

## Gotchas already learned

- `DATA` and `data` collide on Windows; intermediates land in the existing `DATA/` directory even though `config.json` says `data`.
- Parquet list columns come back from `iter_batches` as numpy arrays, not Python lists.
- Per-token Metaphone blocking explodes (75 GiB temp); it was removed. Do not reintroduce it.
- Commas are data: always `sep="\t"`, `keep_default_na=False`.
- Generated datasets/keys/candidates are git-ignored; never commit them.
