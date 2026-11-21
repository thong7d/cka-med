"""SHA-256 hashing and run manifest utilities."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any


BANGKOK_TZ = timezone(timedelta(hours=7))


def bangkok_now_iso() -> str:
    return datetime.now(BANGKOK_TZ).isoformat(timespec="seconds")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_directory(root: Path | str, patterns: tuple[str, ...] = ("*.py", "*.yaml", "*.yml", "*.json")) -> dict[str, str]:
    root_path = Path(root)
    if not root_path.exists():
        return {}
    hashes: dict[str, str] = {}
    files: set[Path] = set()
    for pat in patterns:
        files.update(root_path.rglob(pat))
    for f in sorted(files):
        if f.is_file() and "__pycache__" not in f.parts:
            hashes[f.relative_to(root_path).as_posix()] = sha256_file(f)
    return hashes


def build_run_manifest(
    *,
    run_id: str,
    model_id: str,
    model_revision: str,
    split: str,
    seed: int,
    config: dict[str, Any],
    data_manifest_path: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    src_hashes = hash_directory(repo_root / "src")
    cfg_hashes = hash_directory(repo_root / "configs")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at_bangkok": bangkok_now_iso(),
        "model_id": model_id,
        "model_revision": model_revision,
        "split": split,
        "seed": seed,
        "config": config,
        "source_tree_sha256": sha256_text(json.dumps(src_hashes, sort_keys=True)),
        "config_tree_sha256": sha256_text(json.dumps(cfg_hashes, sort_keys=True)),
        "data_manifest_sha256": sha256_file(data_manifest_path) if data_manifest_path and Path(data_manifest_path).exists() else None,
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_json(path: Path | str, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p
