from pathlib import Path

import duckdb

# Fallbacks mirror the Config dataclass defaults. They only apply when a caller
# hands in a duck-typed stand-in (the unit tests do) instead of a real Config;
# every production path passes a real Config, so the Config values always win.
DEFAULT_MEMORY_LIMIT = 8e9
DEFAULT_THREADS = 8
DEFAULT_MAX_TEMP = 50 * 1024 ** 3

# Per-call memory ceilings are preserved from the previous hard-coded settings so
# the out-of-core working points do not change; each one only overrides the
# Config default (BER_DUCK_* / config.json). Setting a ceiling above available
# RAM is harmless in DuckDB — it spills to temp_directory — but the temp
# directory must be sized to the volume that actually holds it, which on Kaggle
# is the ~20 GB /kaggle/working rather than the 50-80 GB the local box allowed.
_SETTINGS = (
    "SET preserve_insertion_order=false",
    "SET temp_directory={temp}",
    "PRAGMA max_temp_directory_size={max_temp}",
    "SET memory_limit={memory}",
    "SET threads={threads}",
)


def format_bytes(value) -> str:
    """Render a byte count the way DuckDB's parser will read it back.

    DuckDB accepts only KB/MB/GB/TB (1000^i) or KiB/MiB/GiB/TiB (1024^i) — a bare
    integer is a parser error — and it keeps a single decimal place, so a raw byte
    count cannot be expressed. Emitting "7.45GiB" would silently become 7.4GiB.
    Pick the most compact unit in which the value is exact, which reproduces the
    historical 8GB/10GB/12GB and 50GiB/60GiB/80GiB working points byte-for-byte.
    """
    size = float(value)
    for unit, factor in (
        ("TB", 1000 ** 4), ("GB", 1000 ** 3), ("MB", 1000 ** 2),
        ("TiB", 1024 ** 4), ("GiB", 1024 ** 3), ("MiB", 1024 ** 2),
    ):
        if size >= factor and abs(size / factor - round(size / factor)) < 1e-9:
            return f"{round(size / factor):g}{unit}"
    return f"{size / 1000 ** 3:.1f}GB"


def _sql_str(value) -> str:
    # DuckDB follows the Postgres convention: single quotes delimit strings and an
    # embedded quote is doubled. repr() must not be used here because a path
    # containing an apostrophe would emit double quotes, which bind as identifiers.
    return "'" + str(value).replace("'", "''") + "'"


def temp_size_literal(value) -> str:
    return _sql_str(format_bytes(value))


def _scratch_dir(cfg):
    """Return the spill directory, or None when cfg carries no artifact dir.

    Returning None leaves temp_directory and max_temp_directory_size at DuckDB's
    own defaults instead of creating a stray `tmp/` next to the caller.
    """
    explicit = getattr(cfg, "duck_tmp_dir", None)
    if explicit:
        return Path(explicit) / "tmp"
    data_dir = getattr(cfg, "data_dir", None)
    if data_dir is None:
        return None
    return Path(data_dir) / "tmp"


def connect(cfg, memory=None, temp=None):
    """Open a DuckDB session honouring the Config runtime knobs.

    `memory` and `temp` are the per-stage ceilings the stages used to hard-code;
    pass None to take the Config value (which is itself env-overridable).
    """
    memory_limit = getattr(cfg, "duck_memory_limit", DEFAULT_MEMORY_LIMIT)
    if memory is not None:
        memory_limit = memory
    max_temp = getattr(cfg, "duck_max_temp", DEFAULT_MAX_TEMP)
    if temp is not None:
        max_temp = temp
    threads = int(getattr(cfg, "duck_threads", DEFAULT_THREADS))

    scratch = _scratch_dir(cfg)
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    for statement in _SETTINGS:
        if "{temp}" in statement:
            if scratch is None:
                continue
            statement = statement.format(
                temp=_sql_str(scratch.as_posix()), max_temp=temp_size_literal(max_temp)
            )
        elif "{max_temp}" in statement:
            statement = statement.format(max_temp=temp_size_literal(max_temp))
        elif "{memory}" in statement:
            statement = statement.format(memory=temp_size_literal(memory_limit))
        elif "{threads}" in statement:
            statement = statement.format(threads=threads)
        con.execute(statement)
    return con
