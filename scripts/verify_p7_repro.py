"""Phase 7 CPU reproducibility audit: verify P4 hashes, re-run data audit, recompute metrics from raw JSONLs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from cka_med.data.audit import audit_manifest
from cka_med.evaluation.bootstrap import paired_cluster_bootstrap
from cka_med.evaluation.metrics import evaluate_rows, join_predictions_with_gold
from cka_med.manifest import hash_directory, save_json, sha256_file
from cka_med.schema import read_jsonl


def main() -> int:
    # 1. Verify P4 frozen hashes match current src/configs/data
    p4_freeze = json.loads(Path("reports/phase_4/freeze_manifest.json").read_text(encoding="utf-8"))
    cur_data_hashes = {
        "train.jsonl": sha256_file("data/processed/train.jsonl"),
        "dev.jsonl": sha256_file("data/processed/dev.jsonl"),
        "test.jsonl": sha256_file("data/processed/test.jsonl"),
        "kg_subset.json": sha256_file("data/processed/kg_subset.json"),
    }
    data_hash_match = cur_data_hashes == p4_freeze["data_file_hashes"]
    cur_cfg_hashes = hash_directory("configs")
    cfg_hash_match = cur_cfg_hashes == p4_freeze["config_hashes"]
    cur_src_hashes = hash_directory("src")
    # Note: src/cka_med/training/trainer.py had A-U unconditional J-branch added after P4 freeze; track exact diff
    changed_src = [
        k for k in sorted(set(cur_src_hashes) | set(p4_freeze["src_hashes"]))
        if cur_src_hashes.get(k) != p4_freeze["src_hashes"].get(k)
    ]

    # 2. Re-run data audit
    audit_res = audit_manifest("data/manifests/poc_v1.json")
    if audit_res["status"] != "PASS":
        raise RuntimeError(f"Data audit failed: {audit_res}")

    # 3. Re-run pytest
    pytest_proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, check=False)

    # 4. Recompute key test metrics directly from raw JSONLs and compare to phase_6/confidence_intervals.json
    test_records = read_jsonl("data/processed/test.jsonl")
    p6_ci = json.loads(Path("reports/phase_6/confidence_intervals.json").read_text(encoding="utf-8"))

    recomputed = {}
    joined_check = {}
    for mid in ("B-N", "B-R", "B-P_v1", "B-F", "B-S_seed42", "M-C_lam05_seed42", "A-U_seed42", "B-S_seed43", "M-C_lam05_seed43"):
        preds = read_jsonl(f"artifacts/runs/{mid}_test/predictions.jsonl")
        j = join_predictions_with_gold(test_records, preds)
        ev = evaluate_rows(j)
        joined_check[mid] = j
        recomputed[mid] = {
            "acc_0_v": ev["summary"]["acc_0_v"],
            "acc_near_v": ev["summary"]["acc_near_v"],
            "kug": ev["summary"]["kug"],
        }
        # Exact match check against saved CI point estimates
        for k, v in recomputed[mid].items():
            saved_v = p6_ci["models"][mid]["point_estimates"][k]
            if abs(float(v) - float(saved_v)) > 1e-12:
                raise AssertionError(f"Mismatch for {mid}.{k}: recomputed={v}, saved={saved_v}")

    # 5. Build bundle checksums index
    checksums = {
        "data_manifests": hash_directory("data/manifests"),
        "configs": cur_cfg_hashes,
        "src": cur_src_hashes,
        "reports_phase_4": hash_directory("reports/phase_4", patterns=("*.yaml", "*.md", "*.json", "*.csv")),
        "reports_phase_6": hash_directory("reports/phase_6", patterns=("*.csv", "*.json", "*.md", "*.png")),
        "test_predictions": {
            p.parent.name: sha256_file(p)
            for p in sorted(Path("artifacts/runs").glob("*_test*/predictions.jsonl"))
        },
        "adapters": {
            p.parent.parent.name: sha256_file(p)
            for p in sorted(Path("artifacts/runs").glob("*/adapter/adapter_model.safetensors"))
        },
    }
    p7_dir = Path("reports/phase_7")
    p7_dir.mkdir(parents=True, exist_ok=True)
    save_json(p7_dir / "bundle_checksums.json", checksums)

    repro_summary = {
        "data_hash_match_p4": data_hash_match,
        "config_hash_match_p4": cfg_hash_match,
        "src_files_changed_after_p4": changed_src,
        "data_audit_status": audit_res["status"],
        "pytest_returncode": pytest_proc.returncode,
        "pytest_summary": pytest_proc.stdout.strip().splitlines()[-1] if pytest_proc.stdout else "",
        "metric_recompute_exact_match": True,
        "n_test_prediction_files_verified": len(checksums["test_predictions"]),
        "n_adapters_checksummed": len(checksums["adapters"]),
    }
    save_json(p7_dir / "repro_verification.json", repro_summary)
    print(json.dumps(repro_summary, indent=2))
    return 0 if pytest_proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
