"""Paired variant dataset builder for fair B-S, M-C, and A-U training."""

from __future__ import annotations

import collections
import random
from typing import Any

from cka_med.schema import DataRecord


def build_paired_training_items(
    records: list[DataRecord],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Pair each non-K0 variant (group_id, Kk) with its matching (group_id, K0) record.

    Both B-S and M-C iterate over the exact same paired item list so exposure,
    sampler order, and answer supervision are identical.
    """
    by_group: dict[str, dict[str, DataRecord]] = collections.defaultdict(dict)
    for rec in records:
        if rec.exclusion_reason is not None or rec.oracle_context:
            continue
        by_group[rec.group_id][rec.condition] = rec

    pairs: list[dict[str, Any]] = []
    for gid in sorted(by_group.keys()):
        cond_map = by_group[gid]
        if "K0" not in cond_map:
            raise ValueError(f"Group {gid} missing K0 record for paired training.")
        rec_k0 = cond_map["K0"]
        for cond in sorted(cond_map.keys()):
            if cond == "K0":
                continue
            rec_kk = cond_map[cond]
            pairs.append(
                {
                    "group_id": gid,
                    "task_type": rec_k0.task_type,
                    "condition_k": cond,
                    "rec_k0": rec_k0,
                    "rec_kk": rec_kk,
                    "gold_invariant": bool(rec_kk.gold_invariant),
                    "provenance_quality": rec_kk.provenance_quality,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs
