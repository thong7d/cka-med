"""SLAKE ingestion, image download, and byte/perceptual deduplication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image


SLAKE_REPO_ID = "BoKelvin/SLAKE"
SLAKE_PINNED_REVISION = "a9083ce6c34ac3ffb17671a605962924d8a8f9e9"


def compute_dhash(image_path: Path | str, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash (dHash) from RGB/grayscale pixels."""
    with Image.open(image_path) as im:
        gray = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(bool(val))
    return f"{bits:016x}"


def hamming_distance_hex(h1: str, h2: str) -> int:
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def ensure_slake_images(
    raw_dir: Path | str,
    *,
    repo_id: str = SLAKE_REPO_ID,
    revision: str = SLAKE_PINNED_REVISION,
) -> Path:
    """Download and extract SLAKE imgs.zip if not already extracted. Never uses segmentation masks."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    imgs_dir = raw_path / "imgs"
    if imgs_dir.exists() and any(imgs_dir.iterdir()):
        return imgs_dir

    zip_local = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="imgs.zip",
            repo_type="dataset",
            revision=revision,
            local_dir=str(raw_path),
        )
    )
    with zipfile.ZipFile(zip_local, "r") as zf:
        for member in zf.namelist():
            # Skip any macOS metadata or non-source files? Extract cleanly, but audit will reject masks.
            zf.extract(member, raw_path)
    if not imgs_dir.exists():
        raise FileNotFoundError(f"Expected extracted directory {imgs_dir} after unzipping {zip_local}")
    return imgs_dir


def resolve_source_image(imgs_dir: Path, img_name: str) -> Path:
    """Resolve and strictly verify that image path points to source image, NEVER segmentation mask."""
    norm_name = img_name.replace("\\", "/").lstrip("/")
    if "mask" in Path(norm_name).name.lower():
        raise ValueError(f"Refusing to load mask image as model input: {img_name}")
    candidate = imgs_dir / norm_name
    if not candidate.exists():
        raise FileNotFoundError(f"Source image missing: {candidate}")
    return candidate


def audit_images_and_dedup(
    split_records: dict[str, list[dict[str, Any]]],
    imgs_dir: Path,
    *,
    near_dup_hamming_threshold: int = 2,
) -> dict[str, Any]:
    """Compute SHA-256 and dHash for every unique image across splits and detect cross-split duplicates.

    Fixed deterministic exclusion rule for cross-split duplicates:
      Precedence order: test > validation > train? No! To protect independent test integrity:
      Keep test intact if duplicate appears in train/validation, OR exclude from train/val first;
      if duplicate between train and validation, exclude from validation? Actually:
      1. If an image in train matches test -> exclude from train (do not leak test into train).
      2. If an image in validation matches test -> exclude from validation.
      3. If an image in train matches validation -> exclude from validation.
      All exclusions are logged explicitly.
    """
    image_info: dict[str, dict[str, Any]] = {}
    sha_to_imgs: dict[str, list[tuple[str, str]]] = {}
    dhash_by_split: dict[str, list[tuple[str, str]]] = {"train": [], "validation": [], "test": []}

    for split_name in ("train", "validation", "test"):
        seen_names: set[str] = set()
        for row in split_records.get(split_name, []):
            img_name = str(row["img_name"])
            if img_name in seen_names:
                continue
            seen_names.add(img_name)
            img_path = resolve_source_image(imgs_dir, img_name)
            data = img_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            dh = compute_dhash(img_path)
            with Image.open(img_path) as im:
                w, h = im.size
                mode = im.mode
            image_info[f"{split_name}::{img_name}"] = {
                "split": split_name,
                "img_name": img_name,
                "img_id": str(row["img_id"]),
                "path": img_path.as_posix(),
                "sha256": sha,
                "dhash": dh,
                "width": w,
                "height": h,
                "mode": mode,
            }
            sha_to_imgs.setdefault(sha, []).append((split_name, img_name))
            dhash_by_split[split_name].append((img_name, dh))

    exact_cross_split_dups: list[dict[str, Any]] = []
    excluded_images: dict[tuple[str, str], str] = {}

    for sha, entries in sha_to_imgs.items():
        splits_present = {sp for sp, _ in entries}
        if len(splits_present) > 1:
            exact_cross_split_dups.append({"sha256": sha, "entries": entries})
            # Priority keep: test > train > validation
            if "test" in splits_present:
                for sp, name in entries:
                    if sp != "test":
                        excluded_images[(sp, name)] = f"exact_sha256_duplicate_with_test:{sha[:12]}"
            else:
                for sp, name in entries:
                    if sp == "validation":
                        excluded_images[(sp, name)] = f"exact_sha256_duplicate_with_train:{sha[:12]}"

    near_cross_split_dups: list[dict[str, Any]] = []
    # Compare (train vs test), (validation vs test), (train vs validation)
    pairs_to_check = [("train", "test"), ("validation", "test"), ("train", "validation")]
    for sp_a, sp_b in pairs_to_check:
        for name_a, dh_a in dhash_by_split[sp_a]:
            for name_b, dh_b in dhash_by_split[sp_b]:
                dist = hamming_distance_hex(dh_a, dh_b)
                if dist <= near_dup_hamming_threshold:
                    near_cross_split_dups.append(
                        {
                            "split_a": sp_a,
                            "img_a": name_a,
                            "dhash_a": dh_a,
                            "split_b": sp_b,
                            "img_b": name_b,
                            "dhash_b": dh_b,
                            "hamming_distance": dist,
                        }
                    )
                    if sp_b == "test" and (sp_a, name_a) not in excluded_images:
                        excluded_images[(sp_a, name_a)] = f"near_dhash_duplicate_with_test(dist={dist})"
                    elif sp_a == "train" and sp_b == "validation" and (sp_b, name_b) not in excluded_images:
                        excluded_images[(sp_b, name_b)] = f"near_dhash_duplicate_with_train(dist={dist})"

    return {
        "total_unique_split_images": len(image_info),
        "image_info": image_info,
        "exact_cross_split_dups": exact_cross_split_dups,
        "near_cross_split_dups": near_cross_split_dups,
        "excluded_images": {f"{sp}::{name}": reason for (sp, name), reason in sorted(excluded_images.items())},
    }
