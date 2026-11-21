"""CLI entry point to run baseline or adapter inference on a data split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from cka_med.inference.qwen_runner import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_qwen_model_and_processor,
    predict_record,
)
from cka_med.manifest import build_run_manifest, save_json
from cka_med.schema import DataRecord, PredictionWriter, read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CKA-MedVQA inference.")
    parser.add_argument("--config", required=True, help="Experiment YAML config path.")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of records (0 = all).")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_path = Path(cfg["data_files"][args.split])
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        return 2

    records = [DataRecord.from_dict(r) for r in read_jsonl(data_path)]
    if args.limit > 0:
        records = records[: args.limit]

    model_id = cfg.get("base_model_id", DEFAULT_MODEL_ID)
    model_rev = cfg.get("base_model_revision", DEFAULT_MODEL_REVISION)
    adapter_path = cfg.get("adapter_path")
    if adapter_path and not Path(adapter_path).exists():
        print(f"Adapter checkpoint not found: {adapter_path}", file=sys.stderr)
        return 2

    model, processor, dtype = load_qwen_model_and_processor(
        model_id=model_id,
        model_revision=model_rev,
        adapter_path=adapter_path,
        min_pixels=int(cfg.get("min_pixels", 256 * 28 * 28)),
        max_pixels=int(cfg.get("max_pixels", 512 * 28 * 28)),
        use_4bit=bool(cfg.get("use_4bit", False)),
    )
    model.eval()

    runs = cfg.get("runs", [])
    if not runs:
        print("No inference runs specified in config.", file=sys.stderr)
        return 2

    out_root = Path(cfg.get("output_dir", "artifacts/runs"))
    for run_spec in runs:
        run_id = f"{run_spec['run_id']}_{args.split}"
        mode = run_spec["mode"]
        prompt_variant = run_spec.get("prompt_variant", "default")
        task_filter = run_spec.get("task_filter")  # optional list e.g. ["J"] for D-K
        allow_oracle = bool(run_spec.get("allow_oracle", False))

        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pred_path = run_dir / "predictions.jsonl"

        # Support safe resume by skipping already written sample_ids
        writer = PredictionWriter(pred_path, overwrite=False)
        existing_sids = {sid for _, sid in writer.seen_keys}

        selected = [
            r for r in records if (task_filter is None or r.task_type in task_filter)
        ]
        for rec in selected:
            if rec.sample_id in existing_sids:
                continue
            pred = predict_record(
                model=model,
                processor=processor,
                rec=rec,
                run_id=run_id,
                model_id=run_spec.get("model_label", mode),
                mode=mode,
                prompt_variant=prompt_variant,
                allow_oracle=allow_oracle,
            )
            writer.write(pred)

        manifest = build_run_manifest(
            run_id=run_id,
            model_id=run_spec.get("model_label", mode),
            model_revision=model_rev,
            split=args.split,
            seed=int(cfg.get("seed", 42)),
            config=run_spec,
            data_manifest_path=cfg.get("data_manifest_path"),
            extra={"dtype": str(dtype), "n_selected": len(selected), "predictions_path": pred_path.as_posix()},
        )
        save_json(run_dir / "run_manifest.json", manifest)
        print(json.dumps({"run_id": run_id, "n_records": len(selected), "predictions": pred_path.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
