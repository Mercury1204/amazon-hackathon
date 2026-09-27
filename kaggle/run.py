"""Kaggle entry point for the BER pipeline.

This lives in git rather than being `%%writefile`-ed into the notebook. The
notebook used to carry two `%%writefile /kaggle/working/run.py` cells; the second
silently replaced the first, and running the notebook out of order left the stale
one on disk. That version had no `prepare` branch, so `prepare` fell through to a
bare `print("RUN_OK")` and the run appeared to succeed while writing nothing.
See F23 in docs/FAILURES_AND_FIXES.md.

Usage:
    python3 run.py <command> [extra ber.cli args]

    run.py prepare
    run.py block train
    run.py audit
    run.py features train --workers 8
"""

import os
import sys

REPO = os.environ.get("BER_REPO_DIR", "/kaggle/working/repo")

# Persistent artifacts on the 20 GB working volume; DuckDB scratch on the root
# overlay. These are deliberately different filesystems -- /kaggle/working shares
# /dev/loop2 with /kaggle/lib, so only ~13 GB is actually free there, and spilling
# the candidate join into it fills the disk (F21).
ARTIFACT_DIR = os.environ.get("BER_KAGGLE_ARTIFACT_DIR", "/kaggle/working/er/artifacts")
SCRATCH_DIR = os.environ.get("BER_KAGGLE_SCRATCH_DIR", "/tmp/er_scratch")
DATASET_DIR = os.environ.get(
    "BER_KAGGLE_DATASET_DIR", "/kaggle/input/datasets/mercury147/amz-er-2026-raw"
)

os.environ.update(
    {
        "BER_ROOT": REPO,
        "BER_DATASET_DIR": DATASET_DIR,
        "BER_ARTIFACT_DIR": ARTIFACT_DIR,
        "BER_MODELS_DIR": os.path.join(ARTIFACT_DIR, "..", "models"),
        "BER_OUTPUT_DIR": os.path.join(ARTIFACT_DIR, "..", "output"),
        "BER_DUCK_TMP_DIR": SCRATCH_DIR,
        "BER_DUCK_MEMORY_LIMIT": "20GB",
        "BER_DUCK_THREADS": "8",
        # Spill lives on the overlay, so this is headroom for the candidate join,
        # not a share of the working volume.
        "BER_DUCK_MAX_TEMP": "120GiB",
    }
)
# Deliberately NOT set: BER_DATA_DIR. It is a fallback alias for BER_ARTIFACT_DIR,
# and the notebook used to point it at the read-only dataset mount, which would
# send DuckDB scratch into /kaggle/input if BER_ARTIFACT_DIR were ever dropped.

os.makedirs(SCRATCH_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "code/business_entity_resolution/src"))

from ber.cli import main  # noqa: E402  (needs the env + sys.path set up first)

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python3 run.py <prepare|block|audit|features|train|tune|"
        "predict|outputs|evaluate|validation|loo> [options]"
    )

main(
    [
        sys.argv[1],
        "--config",
        os.path.join(REPO, "code/business_entity_resolution/config.json"),
        *sys.argv[2:],
    ]
)
