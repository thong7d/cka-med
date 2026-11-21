"""CLI entry point for B-S (same-data SFT)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cka_med.training.trainer import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train same-data SFT baseline (B-S).")
    parser.add_argument("--config", required=True, help="Path to sft.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = run_training(cfg, objective_mode="sft")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
