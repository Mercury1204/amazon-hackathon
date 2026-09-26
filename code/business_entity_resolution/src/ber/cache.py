"""Stage-cache staleness detection.

Every generated artifact is written beside a `<artifact>.meta.json` sidecar holding a
fingerprint of what produced it: the identity (size + mtime) of its input files, the
config keys that affect it, and the source of the modules that compute it. A stage
recomputes when the fingerprint no longer matches, and a consumer refuses a stale
artifact rather than silently scoring against mismatched features.

Without this, editing `pass_caps`, `idf_min`, `seed`, `val_frac` or a feature list
reuses the previous artifact with no warning, because every cache path is keyed by
split name alone.
"""

import hashlib
import json
from pathlib import Path

META_SUFFIX = ".meta.json"


def digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def stamp(path) -> dict | None:
    """Cheap identity of a file or directory: total size and newest mtime.

    Deliberately not a content hash: these artifacts run to gigabytes and are read
    many times per stage, so a stat-based fingerprint is the only affordable option.
    """
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        info = path.stat()
        return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    total, newest = 0, 0
    for child in path.rglob("*"):
        if child.is_file():
            info = child.stat()
            total += info.st_size
            newest = max(newest, info.st_mtime_ns)
    return {"size": total, "mtime_ns": newest, "files": sum(1 for _ in path.rglob("*") if _.is_file())}


def stamps(paths) -> dict:
    return {str(p): stamp(p) for p in paths}


def code_stamp(*sources) -> dict:
    """Fingerprint the source of the modules that compute an artifact.

    Closing the "I changed the code but not the config" hole: a blocking-pass edit
    must invalidate the key cache even though every config key is unchanged.
    """
    out = {}
    for src in sources:
        src = Path(src)
        out[src.name] = hashlib.sha256(src.read_bytes()).hexdigest()[:12] if src.exists() else None
    return out


def meta_path(artifact) -> Path:
    artifact = Path(artifact)
    return artifact.with_name(artifact.name + META_SUFFIX)


def read_meta(artifact) -> dict | None:
    path = meta_path(artifact)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_meta(artifact, inputs, config, code=None, context=None) -> str:
    payload = {
        "fingerprint": digest({"inputs": inputs, "config": config, "code": code or {}}),
        "inputs": inputs,
        "config": config,
        "code": code or {},
        # Recorded but deliberately outside the fingerprint: lets a consumer recover
        # how the artifact was built (e.g. which metadata split a valfull feature set
        # borrowed) without having to be told.
        "context": context or {},
    }
    path = meta_path(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload["fingerprint"]


def read_context(artifact) -> dict:
    meta = read_meta(artifact) or {}
    return meta.get("context") or {}


def fingerprint(inputs, config, code=None) -> str:
    return digest({"inputs": inputs, "config": config, "code": code or {}})


def is_fresh(artifact, inputs, config, code=None) -> bool:
    """True when the artifact exists and was built from exactly these inputs."""
    artifact = Path(artifact)
    if not artifact.exists():
        return False
    meta = read_meta(artifact)
    if meta is None:
        # Pre-fingerprint artifact. Treat as stale so behaviour is explicit rather
        # than silently trusting something whose provenance was never recorded.
        return False
    return meta.get("fingerprint") == fingerprint(inputs, config, code)


def describe_staleness(artifact, inputs, config, code=None) -> str | None:
    """Human-readable reason an artifact is being recomputed, or None if it is fresh."""
    artifact = Path(artifact)
    if not artifact.exists():
        return "missing"
    meta = read_meta(artifact)
    if meta is None:
        return "no provenance recorded (built before fingerprints existed)"
    want = fingerprint(inputs, config, code)
    if meta.get("fingerprint") == want:
        return None
    reasons = []
    old_inputs, old_config, old_code = meta.get("inputs", {}), meta.get("config", {}), meta.get("code", {})
    for key in sorted(set(old_inputs) | set(inputs)):
        if old_inputs.get(key) != inputs.get(key):
            reasons.append(f"input {Path(key).name} changed")
    for key in sorted(set(old_config) | set(config)):
        if old_config.get(key) != config.get(key):
            reasons.append(f"config {key} {old_config.get(key)!r} -> {config.get(key)!r}")
    for key in sorted(set(old_code) | set(code or {})):
        if old_code.get(key) != (code or {}).get(key):
            reasons.append(f"code {key} changed")
    return "; ".join(reasons) or "fingerprint mismatch"


def invalidate(artifact) -> None:
    """Drop an artifact and its sidecar, plus a directory artifact's contents."""
    artifact = Path(artifact)
    if artifact.is_dir():
        import shutil

        shutil.rmtree(artifact, ignore_errors=True)
    elif artifact.exists():
        artifact.unlink()
    meta_path(artifact).unlink(missing_ok=True)
