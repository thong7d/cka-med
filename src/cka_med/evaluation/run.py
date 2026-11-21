"""CLI entry point for evaluating predictions against a data manifest & protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from cka_med.evaluation.bootstrap import paired_cluster_bootstrap
from cka_med.evaluation.metrics import evaluate_rows, join_predictions_with_gold
from cka_med.manifest import save_json
from cka_med.schema import read_jsonl


def evaluate_prediction_files(
    *,
    data_records: list[dict],
    prediction_paths: list[Path],
    comparator_model: str | None = None,
    n_replicates: int = 2000,
    seed: int = 20261121,
    allow_incomplete: bool = False,
) -> dict:
    joined_by_model: dict[str, list[dict]] = {}
    eval_by_model: dict[str, dict] = {}
    for p in prediction_paths:
        preds = read_jsonl(p)
        if not preds:
            raise ValueError(f"Prediction file is empty: {p}")
        model_ids = {str(row["model_id"]) for row in preds}
        if len(model_ids) != 1:
            raise ValueError(f"Prediction file {p} contains multiple or zero model_ids: {model_ids}")
        model_id = next(iter(model_ids))
        joined = join_predictions_with_gold(data_records, preds)
        ev = evaluate_rows(joined)
        if ev["summary"]["incomplete_coverage"] and not allow_incomplete:
            raise RuntimeError(
                f"Incomplete prediction coverage for {model_id}: "
                f"missing={ev['summary']['missing_predictions']} of {ev['summary']['n_eligible_records']}"
            )
        joined_by_model[model_id] = joined
        eval_by_model[model_id] = ev

    bootstrap_res = paired_cluster_bootstrap(
        joined_by_model,
        comparator_model=comparator_model,
        n_replicates=n_replicates,
        seed=seed,
    )
    return {
        "evaluations": eval_by_model,
        "bootstrap": bootstrap_res,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CKA-MedVQA prediction JSONL files.")
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more prediction JSONL files.")
    parser.add_argument("--data", required=False, help="Path to reference data JSONL.")
    parser.add_argument("--protocol", required=False, help="Path to protocol YAML/JSON.")
    parser.add_argument("--comparator", required=False, default=None, help="Comparator model_id for paired deltas.")
    parser.add_argument("--output", required=False, default=None, help="Optional output JSON report path.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Do not fail on missing predictions.")
    args = parser.parse_args(argv)

    n_replicates = 2000
    seed = 20261121
    data_path = args.data
    comparator = args.comparator

    if args.protocol:
        proto_path = Path(args.protocol)
        if not proto_path.exists():
            print(f"Protocol file not found: {proto_path}", file=sys.stderr)
            return 2
        proto = yaml.safe_load(proto_path.read_text(encoding="utf-8"))
        n_replicates = int(proto.get("bootstrap_replicates", n_replicates))
        seed = int(proto.get("bootstrap_seed", seed))
        if data_path is None:
            data_path = proto.get("test_data_path") or proto.get("dev_data_path")
        if comparator is None:
            comparator = proto.get("comparator_model")

    if data_path is None:
        print("Either --data or protocol with data path must be provided.", file=sys.stderr)
        return 2

    data_records = read_jsonl(Path(data_path))
    pred_paths = [Path(p) for p in args.predictions]
    result = evaluate_prediction_files(
        data_records=data_records,
        prediction_paths=pred_paths,
        comparator_model=comparator,
        n_replicates=n_replicates,
        seed=seed,
        allow_incomplete=args.allow_incomplete,
    )

    if args.output:
        save_json(args.output, result)
    print(json.dumps({k: v["summary"] for k, v in result["evaluations"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
