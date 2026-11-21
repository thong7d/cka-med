"""Generate Phase 4 freeze_manifest.json and dataset_manifest.json before P5 training."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from cka_med.manifest import bangkok_now_iso, hash_directory, save_json, sha256_file


def main() -> None:
    p4_dir = Path("reports/phase_4")
    p4_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2("data/manifests/poc_v1.json", p4_dir / "dataset_manifest.json")

    test_preds = list(Path("artifacts/runs").glob("*test*")) if Path("artifacts/runs").exists() else []
    freeze = {
        "frozen_at_bangkok": bangkok_now_iso(),
        "final_test_predictions_exist": len(test_preds) > 0,
        "test_run_dirs_found": [p.name for p in test_preds],
        "protocol_sha256": sha256_file(p4_dir / "protocol_v1.yaml"),
        "dataset_manifest_sha256": sha256_file("data/manifests/poc_v1.json"),
        "data_file_hashes": {
            "train.jsonl": sha256_file("data/processed/train.jsonl"),
            "dev.jsonl": sha256_file("data/processed/dev.jsonl"),
            "test.jsonl": sha256_file("data/processed/test.jsonl"),
            "kg_subset.json": sha256_file("data/processed/kg_subset.json"),
        },
        "src_hashes": hash_directory("src"),
        "config_hashes": hash_directory("configs"),
    }
    if freeze["final_test_predictions_exist"]:
        raise RuntimeError(f"Final test runs already exist before P4 freeze: {test_preds}")
    save_json(p4_dir / "freeze_manifest.json", freeze)
    print(json.dumps({"frozen_at_bangkok": freeze["frozen_at_bangkok"], "final_test_predictions_exist": False}, indent=2))


if __name__ == "__main__":
    main()
