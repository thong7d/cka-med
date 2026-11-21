"""Image-cluster paired bootstrap for confidence intervals."""

from __future__ import annotations

import collections
from typing import Any

import numpy as np

from cka_med.evaluation.metrics import evaluate_rows


def _ci_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_valid_replicates": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_valid_replicates": int(arr.size),
    }


def _extract_scalar_endpoints(eval_out: dict[str, Any]) -> dict[str, float | None]:
    s = eval_out["summary"]
    return {
        "acc_0_v": s["acc_0_v"],
        "acc_near_v": s["acc_near_v"],
        "acc_irrel_v": s["acc_irrel_v"],
        "cfr_near_v": s["cfr_near_v"],
        "cfr_irrel_v": s["cfr_irrel_v"],
        "acc_0_j": s["acc_0_j"],
        "acc_help_j": s["acc_help_j"],
        "kug": s["kug"],
    }


def paired_cluster_bootstrap(
    joined_rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    comparator_model: str | None = None,
    n_replicates: int = 2000,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Run paired image-cluster bootstrap across one or more models.

    Every replicate resamples unique image_id clusters with replacement and carries
    all QA groups & context variants belonging to each sampled image. The exact same
    image resamples are shared across all models for paired delta CIs.
    """
    if not joined_rows_by_model:
        raise ValueError("joined_rows_by_model must be non-empty.")

    model_names = list(joined_rows_by_model.keys())
    image_sets = {
        m: sorted({str(r["image_id"]) for r in rows})
        for m, rows in joined_rows_by_model.items()
    }
    base_images = image_sets[model_names[0]]
    for m in model_names[1:]:
        if image_sets[m] != base_images:
            raise ValueError(f"Model {m} does not share the exact same image clusters as {model_names[0]}.")

    by_model_by_img: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m, rows in joined_rows_by_model.items():
        bucket: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            bucket[str(r["image_id"])].append(r)
        by_model_by_img[m] = bucket

    point_estimates = {
        m: _extract_scalar_endpoints(evaluate_rows(rows))
        for m, rows in joined_rows_by_model.items()
    }

    rng = np.random.default_rng(seed)
    n_clusters = len(base_images)
    metric_keys = ("acc_0_v", "acc_near_v", "acc_irrel_v", "cfr_near_v", "cfr_irrel_v", "acc_0_j", "acc_help_j", "kug")

    samples_by_model: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in metric_keys} for m in model_names
    }
    undef_cfr_near: dict[str, int] = {m: 0 for m in model_names}
    undef_cfr_irrel: dict[str, int] = {m: 0 for m in model_names}

    delta_samples: dict[str, dict[str, list[float]]] = {}
    if comparator_model is not None:
        if comparator_model not in joined_rows_by_model:
            raise ValueError(f"Comparator model {comparator_model} not found.")
        delta_samples = {
            m: {k: [] for k in metric_keys}
            for m in model_names
            if m != comparator_model
        }

    for _ in range(n_replicates):
        chosen_idx = rng.integers(0, n_clusters, size=n_clusters)
        chosen_imgs = [base_images[int(i)] for i in chosen_idx]
        rep_endpoints: dict[str, dict[str, float | None]] = {}
        for m in model_names:
            rep_rows: list[dict[str, Any]] = []
            # Assign synthetic cluster-instance group_id suffix so duplicated images in the same
            # bootstrap draw keep their own within-image pairing without colliding across draws.
            img_map = by_model_by_img[m]
            for draw_pos, img_id in enumerate(chosen_imgs):
                for r in img_map[img_id]:
                    r_copy = dict(r)
                    r_copy["group_id"] = f"{r['group_id']}__b{draw_pos}"
                    rep_rows.append(r_copy)
            ep = _extract_scalar_endpoints(evaluate_rows(rep_rows))
            rep_endpoints[m] = ep
            for k in metric_keys:
                if ep[k] is not None:
                    samples_by_model[m][k].append(float(ep[k]))
            if ep["cfr_near_v"] is None:
                undef_cfr_near[m] += 1
            if ep["cfr_irrel_v"] is None:
                undef_cfr_irrel[m] += 1

        if comparator_model is not None:
            comp_ep = rep_endpoints[comparator_model]
            for m in delta_samples:
                m_ep = rep_endpoints[m]
                for k in metric_keys:
                    if m_ep[k] is not None and comp_ep[k] is not None:
                        delta_samples[m][k].append(float(m_ep[k]) - float(comp_ep[k]))

    models_out: dict[str, Any] = {}
    for m in model_names:
        models_out[m] = {
            "point_estimates": point_estimates[m],
            "ci_95": {k: _ci_summary(samples_by_model[m][k]) for k in metric_keys},
            "undefined_cfr_near_replicates": undef_cfr_near[m],
            "undefined_cfr_irrel_replicates": undef_cfr_irrel[m],
        }

    deltas_out: dict[str, Any] = {}
    if comparator_model is not None:
        comp_pt = point_estimates[comparator_model]
        for m in delta_samples:
            m_pt = point_estimates[m]
            deltas_out[f"{m}__minus__{comparator_model}"] = {
                k: {
                    "point_delta": (
                        float(m_pt[k]) - float(comp_pt[k])
                        if (m_pt[k] is not None and comp_pt[k] is not None)
                        else None
                    ),
                    **_ci_summary(delta_samples[m][k]),
                }
                for k in metric_keys
            }

    return {
        "n_replicates": n_replicates,
        "seed": seed,
        "n_image_clusters": n_clusters,
        "comparator_model": comparator_model,
        "models": models_out,
        "paired_deltas": deltas_out,
    }
