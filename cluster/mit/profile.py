"""Strict, shell-safe MIT Slurm profile loading."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PYTHON_TEMPLATE = "${RELATIONAL_VENV}/bin/python"
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "partition",
        "gres",
        "gpu_name_regex",
        "cpus",
        "memory_gb",
        "wall_minutes",
        "python",
    }
)
OPTIONAL_FIELDS = frozenset({"account", "qos"})
PROFILE_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GRES_RE = re.compile(
    r"^gpu(?::[A-Za-z0-9][A-Za-z0-9_.-]{0,63})?:1$"
)
# Alternation is required by the approved example. The remaining shell
# metacharacters are deliberately excluded even though callers never use a
# shell.
_GPU_REGEX_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.|()+-]{0,127}$")
_FORBIDDEN_TEXT = (
    "\x00",
    "\n",
    "\r",
    "\t",
    ";",
    "&",
    "<",
    ">",
    "`",
    "$(",
    "${DATA_ROOT}",
    "${OUT_ROOT}",
)
_EMBEDDED_ROOTS = ("/Users/", "/scratch/", "s3://", "file://")


@dataclass(frozen=True)
class MITProfile:
    schema_version: int
    partition: str
    account: str | None
    qos: str | None
    gres: str
    gpu_name_regex: str
    cpus: int
    memory_gb: int
    wall_minutes: int
    python: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return the normalized public profile fields."""

        return {
            "schema_version": self.schema_version,
            "partition": self.partition,
            "account": self.account,
            "qos": self.qos,
            "gres": self.gres,
            "gpu_name_regex": self.gpu_name_regex,
            "cpus": self.cpus,
            "memory_gb": self.memory_gb,
            "wall_minutes": self.wall_minutes,
            "python": self.python,
        }


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"profile contains duplicate key: {key}")
        value[key] = item
    return value


def _safe_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(marker in value for marker in _FORBIDDEN_TEXT):
        raise ValueError(f"{field} contains a forbidden metacharacter")
    lowered = value.lower()
    if (
        value.startswith(("/", "~/"))
        or "\\" in value
        or "/" in value
        or ".." in value
        or any(root.lower() in lowered for root in _EMBEDDED_ROOTS)
    ):
        raise ValueError(f"{field} contains a path or traversal")
    return value


def _name(value: object, *, field: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    text = _safe_text(value, field=field)
    if not _NAME_RE.fullmatch(text):
        raise ValueError(f"{field} contains unsupported characters")
    return text


def _positive_int(
    value: object,
    *,
    field: str,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value <= 0 or value > maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return value


def validate_profile(raw: object, *, sha256: str) -> MITProfile:
    """Validate an already-decoded profile without coercing any field."""

    if not isinstance(raw, dict):
        raise ValueError("profile must contain a JSON object")
    fields = set(raw)
    missing = sorted(REQUIRED_FIELDS - fields)
    unknown = sorted(fields - PROFILE_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"profile fields do not match schema; "
            f"missing={missing}, unknown={unknown}"
        )
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        raise ValueError("schema_version must be exactly 1")

    partition = _name(raw["partition"], field="partition")
    account = _name(raw.get("account"), field="account", optional=True)
    qos = _name(raw.get("qos"), field="qos", optional=True)
    gres = _safe_text(raw["gres"], field="gres")
    if not _GRES_RE.fullmatch(gres):
        raise ValueError("gres must request exactly one GPU")
    gpu_name_regex = _safe_text(
        raw["gpu_name_regex"],
        field="gpu_name_regex",
    )
    if not _GPU_REGEX_TEXT_RE.fullmatch(gpu_name_regex):
        raise ValueError("gpu_name_regex contains unsupported characters")
    try:
        re.compile(gpu_name_regex)
    except re.error as error:
        raise ValueError("gpu_name_regex is invalid") from error
    python = raw["python"]
    if python != PYTHON_TEMPLATE:
        raise ValueError(
            "python must be exactly ${RELATIONAL_VENV}/bin/python"
        )

    return MITProfile(
        schema_version=SCHEMA_VERSION,
        partition=partition,
        account=account,
        qos=qos,
        gres=gres,
        gpu_name_regex=gpu_name_regex,
        cpus=_positive_int(raw["cpus"], field="cpus", maximum=1024),
        memory_gb=_positive_int(
            raw["memory_gb"],
            field="memory_gb",
            maximum=1_048_576,
        ),
        wall_minutes=_positive_int(
            raw["wall_minutes"],
            field="wall_minutes",
            maximum=43_200,
        ),
        python=python,
        sha256=sha256,
    )


def load_profile(path: Path | str) -> MITProfile:
    """Load one regular JSON file under the closed v1 schema."""

    profile_path = Path(path)
    if not profile_path.is_file() or profile_path.is_symlink():
        raise ValueError(f"profile is missing or unsafe: {profile_path}")
    data = profile_path.read_bytes()
    if len(data) > 65_536:
        raise ValueError("profile exceeds 64 KiB")
    digest = hashlib.sha256(data).hexdigest()
    try:
        raw = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"profile contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("profile must contain valid UTF-8 JSON") from error
    return validate_profile(raw, sha256=digest)
