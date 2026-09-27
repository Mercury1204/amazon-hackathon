import json
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path

# Order matters: the first variable that is set wins, so the canonical name is
# listed first and DATA_DIR stays a back-compat alias for tools/eda_dataset.py
# and tools/bench_gbdt.py, which already read it.
_DIR_ENV = {
    "dataset_dir": ("BER_DATASET_DIR", "DATA_DIR", "BER_DATA_DIR"),
    "data_dir": ("BER_ARTIFACT_DIR",),
    "models_dir": ("BER_MODELS_DIR",),
    "output_dir": ("BER_OUTPUT_DIR",),
}

_DUCK_ENV = {
    "duck_memory_limit": ("BER_DUCK_MEMORY_LIMIT",),
    "duck_threads": ("BER_DUCK_THREADS",),
    "duck_max_temp": ("BER_DUCK_MAX_TEMP",),
    "duck_tmp_dir": ("BER_DUCK_TMP_DIR",),
}

INT_ENV = {"duck_threads"}
SIZE_ENV = {"duck_memory_limit", "duck_max_temp"}
SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kmgt]?i?b?)\s*$", re.IGNORECASE)
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000, "kb": 1000, "ki": 1024, "kib": 1024,
    "m": 1000 ** 2, "mb": 1000 ** 2, "mi": 1024 ** 2, "mib": 1024 ** 2,
    "g": 1000 ** 3, "gb": 1000 ** 3, "gi": 1024 ** 3, "gib": 1024 ** 3,
    "t": 1000 ** 4, "tb": 1000 ** 4, "ti": 1024 ** 4, "tib": 1024 ** 4,
}


def _first_env(names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _kaggle_dataset_dir(base=None):
    """Locate the mounted challenge dataset when running inside Kaggle.

    Kaggle mounts session inputs read-only, so only `dataset_dir` may point there;
    `data_dir` must stay under /kaggle/working. The mount layout is not stable --
    older sessions used /kaggle/input/<slug>, current ones
    /kaggle/input/datasets/<owner>/<slug> -- so glob recursively instead of guessing
    depths. A directory counts only if it holds both splits, so a mounted code
    dataset that happens to contain a `train/` directory cannot win.
    """
    base = Path(base or "/kaggle/input")
    if not base.is_dir():
        return None
    found = []
    for split in ("train", "test"):
        for hit in sorted(base.glob(f"**/{split}/{split}_source1.tsv")):
            candidate = hit.parent.parent if hit.parent.name == split else hit.parent
            if candidate not in found:
                found.append(candidate)
    for candidate in found:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    return None


def _parse_size(text, name):
    match = SIZE_RE.match(str(text))
    if not match:
        raise ValueError(
            f"{name}: cannot parse size {text!r} (expected e.g. 8GB, 512MB, 50GiB)"
        )
    magnitude, unit = match.groups()
    factor = _SIZE_UNITS.get(unit.lower())
    if factor is None:
        raise ValueError(f"{name}: unknown size unit {unit!r} in {text!r}")
    value = int(float(magnitude) * factor)
    if value <= 0:
        raise ValueError(f"{name}: size must be positive, got {text!r}")
    return value


def _cast_env(field, value, name):
    if field in INT_ENV:
        return int(value)
    if field in SIZE_ENV:
        return float(_parse_size(value, name))
    if field == "duck_tmp_dir":
        return Path(value).expanduser()
    return value


def _resolve(value, root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Config:
    root: Path
    dataset_dir: Path
    data_dir: Path
    models_dir: Path
    output_dir: Path
    seed: int = 42
    cap: int = 200
    idf_min: float = 4.0
    max_block: int = 5000
    neg_ratio: int = 4
    lgbm_params: dict = None
    pass_caps: dict = None
    duck_memory_limit: float = 8e9
    duck_threads: int = 8
    duck_max_temp: int = 50 * 1024 ** 3
    duck_tmp_dir: Path = None
    duck_explicit: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "lgbm_params", self.lgbm_params or {})
        object.__setattr__(self, "pass_caps", self.pass_caps or {})

    @property
    def scratch_dir(self) -> Path:
        base = Path(self.duck_tmp_dir) if self.duck_tmp_dir else Path(self.data_dir)
        return base / "tmp"

    def resolve_scratch(self) -> Path:
        path = self.scratch_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def describe(self) -> dict:
        return {
            "root": str(self.root),
            "dataset_dir": str(self.dataset_dir),
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "output_dir": str(self.output_dir),
            "scratch_dir": str(self.scratch_dir),
            "duck_memory_limit": self.duck_memory_limit,
            "duck_threads": self.duck_threads,
            "duck_max_temp": self.duck_max_temp,
        }

    def validate(self) -> "Config":
        """Fail fast on a mistyped input directory instead of after `prepare`."""
        dataset = Path(self.dataset_dir)
        if not (dataset / "train").is_dir() and not (dataset / "test").is_dir():
            raise FileNotFoundError(
                f"dataset_dir has no train/ or test/ subdirectory: {dataset}\n"
                f"Set BER_DATASET_DIR (alias DATA_DIR) to the directory that "
                f"contains train/ and test/, or fix dataset_dir in the config file."
            )
        return self

    @classmethod
    def load(cls, path, validate=False):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"{path}: unknown config keys {unknown}; "
                f"valid keys are {sorted(known - {'root'})}"
            )

        root_env = os.environ.get("BER_ROOT")
        root = Path(payload.get("root") or root_env or Path.cwd()).expanduser()

        overrides = dict(_DIR_ENV)
        overrides.update(_DUCK_ENV)
        # A knob counts as deliberate if it appears in the config file OR in the
        # environment, so a stage's own default never silently overrides it.
        explicit_duck = {k for k in _DUCK_ENV if k in payload}
        for field, env_names in overrides.items():
            value = _first_env(env_names)
            if value is None and field == "dataset_dir":
                value = _kaggle_dataset_dir()
            if value is None:
                continue
            if field in known:
                payload[field] = _cast_env(field, value, env_names[0])
                if field in _DUCK_ENV:
                    explicit_duck.add(field)
            else:
                raise ValueError(f"{path}: no config field named {field!r} for env var {env_names[0]}")

        kwargs = {}
        for key, value in payload.items():
            if key == "root":
                continue
            if key.endswith("_dir"):
                kwargs[key] = _resolve(value, root)
            else:
                kwargs[key] = value

        cfg = cls(root=root, duck_explicit=frozenset(explicit_duck), **kwargs)
        return cfg.validate() if validate else cfg
