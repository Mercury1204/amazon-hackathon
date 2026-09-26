# Decisions and Reasoning

Architecture Decision Records (lightweight) for the Business Entity Resolution pipeline.
Each entry: decision, context, alternatives, and consequence.

## D1 — Blocking plus classifier, not end-to-end
- **Context:** 2.21M train / 1.73M test Source 1 entities must be matched against 10.3M / 10.0M
  candidate records. The naive cross-product is ~10^13 pairs.
- **Decision:** A hard blocking stage produces a bounded candidate set; a pairwise LightGBM
  classifier scores it; a threshold (plus injective assignment) produces final matches.
- **Alternatives:** end-to-end transformer/page-level model (infeasible on local CPU at 10M
  records); embedding ANN blocking (higher ceiling, but CPU-only machine and no GPU locally).
- **Consequence:** recall is bounded by blocking (0.814), which is now the main ceiling.

## D2 — DuckDB for blocking joins, pandas/pyarrow for the rest
- **Context:** joins over 10M+ rows and 250–300M pairs must run in 23 GB RAM.
- **Decision:** DuckDB does out-of-core joins, dedup, ranking, and CSV writing; pandas/pyarrow
  handle normalization, feature blocks, and I/O.
- **Consequence:** stages are restartable via parquet caches; memory stays bounded.

## D3 — Ten blocking passes with per-pass block caps
- **Context:** exact-name blocking alone recalls only ~22% of true pairs (names are noisy);
  per-token Metaphone exploded; large blocks for common tokens must not dominate.
- **Decision:** union passes 1,3,4,5,6,7,8,9,10 (exact name, rare token, address prefix,
  postal+name, fallback, name/street prefix-5, rare-token pairs and triples) and cap candidate
  block sizes per pass (`pass_caps` in `config.json`), then cap per Source 1 entity (`cap`).
- **Reasoning:** the measured coverage of a shared name-token prefix-5 is ~89% and of >=2 shared
  tokens ~78%, so prefix and token-pair keys are the highest-yield signals.
- **Consequence:** train recall 0.8115; India (0.727) trails US (0.868) because Indic names and
  sparse addresses yield fewer shared rare tokens.

## D4 — Do not reintroduce per-token Metaphone
- **Context:** per-token phonetic codes collide massively (generic codes shared by hundreds of
  thousands of records).
- **Decision:** removed entirely; kept only as an offline experiment.
- **Consequence:** avoided the 75 GiB temp explosion; recall recovered via prefix/pair passes.

## D5 — LightGBM as the matcher
- **Context:** benchmarked LightGBM vs XGBoost on 100k sampled Source 1 with hard negatives.
- **Decision:** LightGBM primary. Quality was tied (AUC identical, F0.5 within ±0.0003);
  LightGBM trained 3–4.5x faster, which matters for local iteration. XGBoost remains a drop-in.
- **Consequence:** 610-tree model, `models/lgbm.txt`.

## D6 — 4:1 negative sampling for training
- **Context:** positives are 7.64M GT pairs; scoring every negative pair during training is wasteful.
- **Decision:** train on all found positives plus negatives sampled 4:1, half hard (same pass) and
  half random country-matched.
- **Consequence:** fast training, but thresholds learned on this sample are miscalibrated for the
  full candidate distribution — see D8.

## D7 — Injective one-to-one post-process
- **Context:** the ground truth is injective (each S2/S3 ID appears in at most one Source 1 list).
- **Decision:** when several Source 1 entities claim the same candidate above threshold, keep only
  the highest-probability assignment.
- **Consequence:** verified to improve macro F0.5 on the full-candidate validation (0.8488 vs 0.8475).

## D8 — Threshold must be tuned on full candidates (0.925, not 0.675)
- **Context:** the 4:1 sampled split suggested 0.675–0.70, but it contains ~4 negatives per
  positive versus ~135 candidates per entity at inference.
- **Decision:** re-tune the threshold on the full candidate set for the held-out Source 1 groups.
  The optimum moves to **0.925**; all submission outputs use it.
- **Consequence:** 200,982 singletons instead of 151,576, and much better precision on the metric.

## D9 — Full-candidate held-out validation is the reported number
- **Context:** the 4:1 mark (0.9807) is optimistic because it undersamples confusable negatives.
- **Decision:** report macro F0.5 on the full candidate set of the 20% held-out Source 1 groups,
  predicted exactly as inference.
- **Consequence:** **0.8488** (95% CI 0.8481–0.8496) is the believable in-domain estimate.

## D10 — Leave-one-country-out as the France proxy
- **Context:** France is ~15% of test and has no labels.
- **Decision:** retrain on US only -> validate India, and India only -> validate US.
- **Consequence:** unseen-country drop of 0.09–0.12 F0.5; realistic leaderboard expectation
  ~0.80–0.85 rather than 0.85+.

## D11 — Ship exactly the required submission tree
- **Context:** the problem statement prescribes `output/`, `code/business_entity_resolution/`
  (`src/`, `README.md`, `requirements.txt`), and `Documentation_template.md`.
- **Decision:** ship only that; no `models/` in the package (a reproducer retrains from data).
  A minimal `src/config.json` is included so the packaged pipeline runs as-is.
- **Consequence:** package is spec-compliant and self-contained; model is reproducible but not shipped.

## D12 — Paths and DuckDB limits come from the environment, not from code
- **Context:** RULES §6 makes Kaggle the only place training and heavy inference run, but the stages
  carried 22 hard-coded `SET memory_limit` / `max_temp_directory_size` / `threads` statements
  written for a 23 GB Windows box, and every path was relative to the repository root.
- **Decision:** `ber/config.py` resolves the four directories and the four DuckDB limits from
  `BER_*` environment variables, with the config file as fallback; `ber/duck.py` is the single place
  a DuckDB session is configured, and each stage passes its previous ceiling as a per-call override.
- **Alternatives:** editing the config file per environment (not possible for a Kaggle notebook
  without rewriting the file); keeping the limits in code and patching them per notebook (fragile,
  22 sites).
- **Consequence:** the same code runs locally and on Kaggle with only environment changes, and
  `BER_DUCK_MAX_TEMP` can be lowered to fit Kaggle's ~20 GB `/kaggle/working` instead of failing to
  spill the local 50-80 GB. The per-stage defaults are unchanged, so no existing run is affected —
  `test_stage_sql_matches_pre_refactor_strings` pins that equivalence. `BER_DUCK_MEMORY_LIMIT` must
  be lowered too on a 30 GB Kaggle CPU session.

## D13 — Unknown config keys are a hard error
- **Context:** `Config.load` silently dropped any key it did not recognise, so a typo such as
  `pass_capz` in `config.json` quietly fell back to `DEFAULT_PASS_CAPS` in `blocking.py` and changed
  candidate generation with no warning.
- **Decision:** `Config.load` raises `ValueError` listing the offending keys.
- **Consequence:** a configuration mistake fails immediately instead of silently producing a
  different candidate set. Both shipped config files were checked to contain only valid keys.

## D14 — `predict` refuses a sampled-sweep threshold instead of defaulting to it
- **Context:** F12 established that the 4:1-sampled threshold (0.70) is materially wrong for the
  full candidate distribution (0.925). The only thing protecting the submission was the operator
  remembering to pass `--threshold 0.925`, and `train` rewrote `models/threshold.json` with the bad
  value on every retrain.
- **Decision:** the decision file records its own provenance in a `source` field.
  `sampled_4to1` is rejected by `threshold.resolve_decision` unless the caller passes
  `--threshold` explicitly; `full_candidates` (written only by `tune`) is accepted. The file also
  records the winning method, so `--one-to-one` / `--no-one-to-one` become tri-state and the
  persisted choice applies when neither is passed.
- **Alternatives:** keeping the sampled default and relying on documentation (the status quo, which
  is how the footgun survived); dropping the sampled sweep entirely (it is still the cheapest
  signal available at train time, and it usefully records the size of the sampling gap).
- **Consequence:** the wrong threshold can no longer be used by accident, and the operator error
  becomes an actionable message instead of a silently worse submission. Pinned by
  `test_predict_refuses_a_sampled_sweep_threshold`.

## D15 — `predict` scores, `outputs` writes
- **Context:** `run_predict` did both jobs, so changing only the threshold or the one-to-one flag
  meant re-scoring 250.6M candidate pairs. Task 12 of the implementation plan also listed `outputs`
  as a distinct command that produces the two TSVs.
- **Decision:** split them. `predict` streams features and caches per-part scores; `outputs`
  aggregates the cached scores into `matching_results.tsv` / `candidate_pairs.tsv` and then verifies
  the scorer-rejection invariants, exiting non-zero on any violation.
- **Consequence:** a threshold or assignment change costs seconds instead of a full scoring pass.
  The verified invariants are the cheap ones (one row per test S1, no duplicate S1 rows, no repeated
  ID inside a list, S2-/S3- prefixes only); matches-subset-of-candidates is not re-derived because
  both files are built from the same pairs parquet, and `utils/validate_submission.py` re-checks it
  before submission.

## D16 — Candidate-set degrees, not pair-set degrees
- **Context:** `s1_degree` and `cand_degree` were computed from each split's own `pairs` table. For
  train that is the 4:1 negative sample; for valfull and test it is the full candidate set. The model
  was therefore trained on a feature meaning ~5-20 and served it meaning up to 200 (F20).
- **Decision:** derive both from `candidates/{meta_split}_candidates.parquet`, cached and keyed to
  it. The `valfull` split reads the train candidate set, which is exact for the same reason its
  pairs are a faithful subset.
- **Alternatives:** dropping the two features (loses real signal — a candidate's ambiguity is
  informative); recomputing degrees inside the model at serving time (not possible with a static
  feature file); normalising the degree (hides the mismatch rather than removing it).
- **Consequence:** feature values change, so `features/`, `models/lgbm.txt` and the tuned threshold
  must be recomputed; the previously reported 0.8488 used the skewed features and is not comparable
  to a post-fix run. Phase 1 also gets cheaper, since it no longer aggregates the 250.6M-row test
  pair table.

## D17 — Fingerprint-based cache provenance, recompute on write and refuse on read
- **Context:** every cache path was keyed by split name alone, so editing `idf_min`, `seed`,
  `val_frac` or a blocking pass silently reused the previous artifact (F21). The documented remedy
  was a manual `rm -rf`.
- **Decision:** `ber/cache.py` fingerprints an artifact's input-file identity (size + mtime, not
  content — these run to gigabytes), the config keys that affect it, and the source of the computing
  modules. Producers recompute on mismatch and print the reason; consumers raise. A missing sidecar
  counts as stale.
- **Alternatives:** content hashing (unaffordable for a 250M-row parquet read many times per stage);
  putting a config hash in the cache filename (proliferates files, leaves garbage, and cannot detect
  a code change); warning instead of failing (the whole point is that these failures are silent).
- **Consequence:** editing a blocking pass or a feature now costs a rebuild rather than a wrong
  answer, and `predict` refuses to score against features that no longer match their inputs. The
  cost is one `stat` per input per stage.
