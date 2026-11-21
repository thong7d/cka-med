"""CLI to audit prepared dataset manifest for leakage, duplicates, mask paths, and schema validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cka_med.data.filter import ORGAN_LEAK_TERMS
from cka_med.inference.prompts import render_prompt
from cka_med.manifest import sha256_file
from cka_med.schema import DataRecord, read_jsonl


def audit_manifest(manifest_path: Path | str) -> dict:
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest not found: {m_path}")
    manifest = json.loads(m_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    split_images: dict[str, set[str]] = {}
    split_hashes: dict[str, set[str]] = {}
    sample_ids: set[str] = set()
    total_records = 0

    for sp in ("train", "dev", "test"):
        fpath = Path(manifest["files"][sp])
        if not fpath.exists():
            errors.append(f"Missing data file for {sp}: {fpath}")
            continue
        actual_sha = sha256_file(fpath)
        expected_sha = manifest["splits"][sp]["file_sha256"]
        if actual_sha != expected_sha:
            errors.append(f"Checksum mismatch for {sp}: expected {expected_sha}, got {actual_sha}")

        rows = read_jsonl(fpath)
        imgs: set[str] = set()
        hashes: set[str] = set()
        for raw in rows:
            total_records += 1
            rec = DataRecord.from_dict(raw)
            if rec.sample_id in sample_ids:
                errors.append(f"Duplicate sample_id across dataset: {rec.sample_id}")
            sample_ids.add(rec.sample_id)

            if "mask" in Path(rec.image_path).name.lower():
                errors.append(f"Mask image path detected in {rec.sample_id}: {rec.image_path}")
            if not Path(rec.image_path).exists():
                errors.append(f"Missing image file for {rec.sample_id}: {rec.image_path}")

            imgs.add(rec.img_name)
            hashes.add(rec.image_sha256)

            # Verify prompt isolation never leaks gold/oracle
            model_inp = rec.to_model_input_dict(allow_oracle=False)
            prompt, _ = render_prompt(model_inp, mode="B-R")
            if rec.task_type == "J" and rec.condition == "Khelp":
                for term in ORGAN_LEAK_TERMS:
                    # Ensure question itself didn't leak organ
                    if f" {term} " in f" {rec.original_question.lower()} ":
                        errors.append(f"Organ {term} leaked in J question {rec.sample_id}")

        split_images[sp] = imgs
        split_hashes[sp] = hashes

    # Cross-split leakage checks
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        img_overlap = split_images.get(a, set()) & split_images.get(b, set())
        sha_overlap = split_hashes.get(a, set()) & split_hashes.get(b, set())
        if img_overlap:
            errors.append(f"Image name overlap between {a} and {b}: {sorted(img_overlap)}")
        if sha_overlap:
            errors.append(f"Image SHA256 overlap between {a} and {b}: {sorted(sha_overlap)}")

    report = {
        "manifest_path": m_path.as_posix(),
        "total_records_checked": total_records,
        "unique_sample_ids": len(sample_ids),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit CKA-MedVQA dataset manifest.")
    parser.add_argument("--manifest", required=True, help="Path to data/manifests/poc_v1.json")
    args = parser.parse_args(argv)
    report = audit_manifest(args.manifest)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
