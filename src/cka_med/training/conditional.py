"""CLI entry point for M-C (conditional consistency) and A-U (unconditional ablation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cka_med.training.trainer import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train conditional consistency model (M-C or A-U).")
    parser.add_argument("--config", required=True, help="Path to conditional.yaml")
    parser.add_argument("--unconditional", action="store_true", help="Run A-U unconditional consistency ablation.")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mode = "unconditional" if args.unconditional or cfg.get("unconditional", False) else "conditional"
    summary = run_training(cfg, objective_mode=mode)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
