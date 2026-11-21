"""Modal GPU runner for CKA-MedVQA PoC (profiling, baselines, training, evaluation).

Uses persistent Modal Volume 'cka-med-vol' for HF model cache and run artifacts,
and syncs results back to local workspace paths.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import modal


APP_NAME = "cka-med-poc"
VOLUME_NAME = "cka-med-vol"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "peft==0.14.0",
        "qwen-vl-utils==0.0.10",
        "huggingface_hub==0.29.1",
        "pyyaml==6.0.2",
        "pillow==11.1.0",
        "numpy==2.1.3",
        "scipy==1.15.2",
        "pandas==2.2.3",
    )
    .env({"HF_HOME": "/vol/hf_cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_dir("configs", remote_path="/workspace/configs")
    .add_local_dir("data/processed", remote_path="/workspace/data/processed")
    .add_local_dir("data/manifests", remote_path="/workspace/data/manifests")
    .add_local_dir(
        "data/raw/slake/imgs",
        remote_path="/workspace/data/raw/slake/imgs",
        ignore=["*.png", "*.json", ".DS_Store"],
    )
)


def _setup_workspace() -> None:
    import os
    import sys

    os.chdir("/workspace")
    if "/workspace/src" not in sys.path:
        sys.path.insert(0, "/workspace/src")


@app.function(
    image=image,
    gpu="L4",
    timeout=2400,
    volumes={"/vol": vol},
)
def run_p3_profile_and_dev_baselines() -> dict:
    """P3: Measure load time, VRAM, inference throughput, and run all dev baselines."""
    _setup_workspace()
    import shutil
    import torch
    import yaml

    from cka_med.inference.qwen_runner import (
        load_qwen_model_and_processor,
        predict_record,
    )
    from cka_med.manifest import build_run_manifest, save_json
    from cka_med.schema import DataRecord, PredictionWriter, read_jsonl

    t0 = time.perf_counter()
    model, processor, dtype = load_qwen_model_and_processor(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model.eval()
    load_time_s = time.perf_counter() - t0
    vol.commit()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load(Path("configs/experiments/baselines.yaml").read_text(encoding="utf-8"))
    dev_records = [DataRecord.from_dict(r) for r in read_jsonl(Path("data/processed/dev.jsonl"))]
    # Fix local Windows path strings inside records if any
    for r in dev_records:
        r.image_path = f"data/raw/slake/imgs/{r.img_name}"

    out_root = Path("/vol/artifacts/runs")
    out_root.mkdir(parents=True, exist_ok=True)

    # Warmup 3 samples + profile 120 samples on B-R
    warmup_recs = dev_records[:3]
    for r in warmup_recs:
        _ = predict_record(model=model, processor=processor, rec=r, run_id="warmup", model_id="warmup", mode="B-R")
    torch.cuda.reset_peak_memory_stats()
    t_prof0 = time.perf_counter()
    prof_recs = dev_records[:120]
    for r in prof_recs:
        _ = predict_record(model=model, processor=processor, rec=r, run_id="prof", model_id="prof", mode="B-R")
    prof_elapsed = time.perf_counter() - t_prof0
    inf_peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    # Profile 10 paired train steps (20 branches) for training VRAM & speed estimate
    from cka_med.training.trainer import run_training

    t_tr0 = time.perf_counter()
    smoke_cfg = yaml.safe_load(Path("configs/experiments/conditional.yaml").read_text(encoding="utf-8"))
    smoke_cfg["run_id"] = "P3_profile_train_10steps"
    smoke_cfg["output_dir"] = "/vol/artifacts/runs/P3_profile_train_10steps"
    smoke_cfg["train_data_path"] = "data/processed/train.jsonl"
    smoke_cfg["limit_groups"] = 16
    smoke_cfg["grad_accum_steps"] = 2
    smoke_cfg["max_steps"] = 10
    # Ensure train image paths resolve on Linux container
    train_raw = read_jsonl(Path("data/processed/train.jsonl"))
    fixed_train = []
    for row in train_raw:
        row["image_path"] = f"data/raw/slake/imgs/{row['img_name']}"
        fixed_train.append(row)
    from cka_med.schema import write_jsonl
    write_jsonl(Path("/tmp/train_linux.jsonl"), fixed_train)
    smoke_cfg["train_data_path"] = "/tmp/train_linux.jsonl"

    del model
    torch.cuda.empty_cache()
    tr_prof = run_training(smoke_cfg, objective_mode="conditional")
    tr_prof["wall_time_s_total"] = time.perf_counter() - t_tr0

    # Reload clean base model for dev baseline evaluations
    model, processor, dtype = load_qwen_model_and_processor(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model.eval()

    run_summaries = {}
    for run_spec in cfg["runs"]:
        run_id = f"{run_spec['run_id']}_dev"
        mode = run_spec["mode"]
        prompt_variant = run_spec.get("prompt_variant", "default")
        task_filter = run_spec.get("task_filter")
        selected = [r for r in dev_records if (task_filter is None or r.task_type in task_filter)]

        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pred_path = run_dir / "predictions.jsonl"
        writer = PredictionWriter(pred_path, overwrite=True)

        t_r0 = time.perf_counter()
        for rec in selected:
            pred = predict_record(
                model=model,
                processor=processor,
                rec=rec,
                run_id=run_id,
                model_id=run_spec.get("model_label", mode),
                mode=mode,
                prompt_variant=prompt_variant,
                allow_oracle=False,
            )
            writer.write(pred)
        r_elapsed = time.perf_counter() - t_r0
        manifest = build_run_manifest(
            run_id=run_id,
            model_id=run_spec.get("model_label", mode),
            model_revision=MODEL_REVISION,
            split="dev",
            seed=42,
            config=run_spec,
            data_manifest_path="data/manifests/poc_v1.json",
            extra={"dtype": str(dtype), "gpu": gpu_name, "elapsed_s": r_elapsed, "n_selected": len(selected)},
        )
        save_json(run_dir / "run_manifest.json", manifest)
        run_summaries[run_id] = {"n_records": len(selected), "elapsed_s": r_elapsed}

    profile_data = {
        "gpu_name": gpu_name,
        "dtype": str(dtype),
        "model_load_time_s": load_time_s,
        "inference_120_records_s": prof_elapsed,
        "inference_examples_per_s": len(prof_recs) / prof_elapsed,
        "inference_peak_vram_mb": inf_peak_vram_mb,
        "train_10_opt_steps_profile": tr_prof,
    }
    save_json(out_root / "p3_profile.json", profile_data)
    vol.commit()
    return {"profile": profile_data, "runs": run_summaries}


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={"/vol": vol},
)
def run_p5_training_and_dev_eval(
    *,
    run_id: str,
    objective_mode: str,
    lambda_cond: float,
    seed: int = 42,
    epochs: int = 1,
    grad_accum_steps: int = 8,
    learning_rate: float = 1e-4,
) -> dict:
    """P5: Train B-S, M-C, or A-U on train.jsonl and evaluate on dev.jsonl."""
    _setup_workspace()
    import torch
    import yaml

    from cka_med.inference.qwen_runner import load_qwen_model_and_processor, predict_record
    from cka_med.manifest import build_run_manifest, save_json
    from cka_med.schema import DataRecord, PredictionWriter, read_jsonl, write_jsonl
    from cka_med.training.trainer import run_training

    train_raw = read_jsonl(Path("data/processed/train.jsonl"))
    for row in train_raw:
        row["image_path"] = f"data/raw/slake/imgs/{row['img_name']}"
    linux_train = Path(f"/tmp/train_{run_id}.jsonl")
    write_jsonl(linux_train, train_raw)

    cfg = {
        "run_id": run_id,
        "base_model_id": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "seed": seed,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 512 * 28 * 28,
        "use_4bit": False,
        "train_data_path": linux_train.as_posix(),
        "data_manifest_path": "data/manifests/poc_v1.json",
        "output_dir": f"/vol/artifacts/runs/{run_id}",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "grad_accum_steps": grad_accum_steps,
        "epochs": epochs,
        "max_steps": 0,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "save_every_steps": 40,
        "lambda_cond": lambda_cond,
    }
    tr_summary = run_training(cfg, objective_mode=objective_mode)

    # Reload trained adapter cleanly from disk and evaluate on dev
    torch.cuda.empty_cache()
    model, processor, dtype = load_qwen_model_and_processor(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        adapter_path=tr_summary["adapter_dir"],
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model.eval()

    dev_records = [DataRecord.from_dict(r) for r in read_jsonl(Path("data/processed/dev.jsonl"))]
    for r in dev_records:
        r.image_path = f"data/raw/slake/imgs/{r.img_name}"

    eval_dir = Path(f"/vol/artifacts/runs/{run_id}_dev")
    eval_dir.mkdir(parents=True, exist_ok=True)
    writer = PredictionWriter(eval_dir / "predictions.jsonl", overwrite=True)
    t0 = time.perf_counter()
    for rec in dev_records:
        pred = predict_record(
            model=model,
            processor=processor,
            rec=rec,
            run_id=f"{run_id}_dev",
            model_id=run_id,
            mode="B-R",
            allow_oracle=False,
        )
        writer.write(pred)
    eval_elapsed = time.perf_counter() - t0
    save_json(
        eval_dir / "run_manifest.json",
        build_run_manifest(
            run_id=f"{run_id}_dev",
            model_id=run_id,
            model_revision=MODEL_REVISION,
            split="dev",
            seed=seed,
            config={"adapter_dir": tr_summary["adapter_dir"]},
            data_manifest_path="data/manifests/poc_v1.json",
            extra={"dtype": str(dtype), "elapsed_s": eval_elapsed, "n_selected": len(dev_records)},
        ),
    )
    vol.commit()
    return {"train_summary": tr_summary, "dev_eval_elapsed_s": eval_elapsed}


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={"/vol": vol},
)
def run_p6_final_eval(
    *,
    adapter_runs: list[str],
    selected_mc_run: str,
    run_stress: bool = True,
    include_baselines: bool = True,
) -> dict:
    """P6: Evaluate all mandatory baselines, trained models, dependency controls, and stress tests on frozen test.jsonl."""
    _setup_workspace()
    import torch
    import yaml

    from cka_med.data.contexts import build_context_pack
    from cka_med.inference.prompts import PROMPT_VERSION_STRESS
    from cka_med.inference.qwen_runner import load_qwen_model_and_processor, predict_record
    from cka_med.manifest import build_run_manifest, save_json
    from cka_med.schema import DataRecord, PredictionWriter, read_jsonl

    test_records = [DataRecord.from_dict(r) for r in read_jsonl(Path("data/processed/test.jsonl"))]
    for r in test_records:
        r.image_path = f"data/raw/slake/imgs/{r.img_name}"

    kg_subset = json.loads(Path("data/processed/kg_subset.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path("configs/experiments/baselines.yaml").read_text(encoding="utf-8"))

    # Filter baselines: use selected B-P variant (default v1) + B-N, B-R, B-F, D-Q, D-K
    baseline_specs = [r for r in cfg["runs"] if r["run_id"] != "B-P_v2"]
    out_root = Path("/vol/artifacts/runs")

    # 1. Base model baselines on test
    results = {}
    if include_baselines:
        model, processor, dtype = load_qwen_model_and_processor(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        model.eval()

        for spec in baseline_specs:
            run_id = f"{spec['run_id']}_test"
            task_filter = spec.get("task_filter")
            selected = [r for r in test_records if (task_filter is None or r.task_type in task_filter)]
            run_dir = out_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            writer = PredictionWriter(run_dir / "predictions.jsonl", overwrite=True)
            t0 = time.perf_counter()
            for rec in selected:
                writer.write(
                    predict_record(
                        model=model,
                        processor=processor,
                        rec=rec,
                        run_id=run_id,
                        model_id=spec.get("model_label", spec["mode"]),
                        mode=spec["mode"],
                        prompt_variant=spec.get("prompt_variant", "default"),
                    )
                )
            elapsed = time.perf_counter() - t0
            save_json(
                run_dir / "run_manifest.json",
                build_run_manifest(
                    run_id=run_id,
                    model_id=spec.get("model_label", spec["mode"]),
                    model_revision=MODEL_REVISION,
                    split="test",
                    seed=42,
                    config=spec,
                    data_manifest_path="data/manifests/poc_v1.json",
                    extra={"dtype": str(dtype), "elapsed_s": elapsed, "n_selected": len(selected)},
                ),
            )
            results[run_id] = elapsed

        del model
        torch.cuda.empty_cache()

    # 2. Trained adapters on test (B-S, selected M-C, A-U, optional seed43)
    for ad_run in adapter_runs:
        adapter_dir = out_root / ad_run / "adapter"
        model, processor, dtype = load_qwen_model_and_processor(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            adapter_path=adapter_dir,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        model.eval()
        run_id = f"{ad_run}_test"
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = PredictionWriter(run_dir / "predictions.jsonl", overwrite=True)
        t0 = time.perf_counter()
        for rec in test_records:
            writer.write(
                predict_record(
                    model=model,
                    processor=processor,
                    rec=rec,
                    run_id=run_id,
                    model_id=ad_run,
                    mode="B-R",
                )
            )
        elapsed = time.perf_counter() - t0
        save_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                run_id=run_id,
                model_id=ad_run,
                model_revision=MODEL_REVISION,
                split="test",
                seed=42,
                config={"adapter_dir": adapter_dir.as_posix()},
                data_manifest_path="data/manifests/poc_v1.json",
                extra={"dtype": str(dtype), "elapsed_s": elapsed, "n_selected": len(test_records)},
            ),
        )
        results[run_id] = elapsed

        # Stress test (held-out rendering template + reversed pack order) for B-S and selected M-C
        if run_stress and (ad_run.startswith("B-S") or ad_run == selected_mc_run):
            stress_id = f"{ad_run}_test_stress"
            s_dir = out_root / stress_id
            s_dir.mkdir(parents=True, exist_ok=True)
            s_writer = PredictionWriter(s_dir / "predictions.jsonl", overwrite=True)
            t_s0 = time.perf_counter()
            for rec in test_records:
                rec_s = DataRecord.from_dict(rec.to_dict())
                pack_id, ctx_txt, _ = build_context_pack(
                    question=rec_s.original_question,
                    task_type=rec_s.task_type,
                    question_family=rec_s.question_family,
                    condition=rec_s.condition,
                    kg_subset=kg_subset,
                    rendering_version=PROMPT_VERSION_STRESS,
                    reverse_order=True,
                )
                rec_s.context_pack_id = pack_id
                rec_s.context_text = ctx_txt
                rec_s.rendering_version = PROMPT_VERSION_STRESS
                s_writer.write(
                    predict_record(
                        model=model,
                        processor=processor,
                        rec=rec_s,
                        run_id=stress_id,
                        model_id=f"{ad_run}_stress",
                        mode="B-R",
                        rendering_version=PROMPT_VERSION_STRESS,
                    )
                )
            s_elapsed = time.perf_counter() - t_s0
            results[stress_id] = s_elapsed

        del model
        torch.cuda.empty_cache()

    vol.commit()
    return results


@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={"/vol": vol},
)
def run_p7_repro_subset(adapter_run: str = "M-C_lam05_seed42", n_samples: int = 15) -> dict:
    """P7: Reload saved adapter from disk and verify inference reproducibility on 15 test samples."""
    _setup_workspace()
    from cka_med.inference.qwen_runner import load_qwen_model_and_processor, predict_record
    from cka_med.schema import DataRecord, read_jsonl

    test_records = [DataRecord.from_dict(r) for r in read_jsonl(Path("data/processed/test.jsonl"))][:n_samples]
    for r in test_records:
        r.image_path = f"data/raw/slake/imgs/{r.img_name}"

    adapter_dir = Path(f"/vol/artifacts/runs/{adapter_run}/adapter")
    model, processor, _ = load_qwen_model_and_processor(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        adapter_path=adapter_dir,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model.eval()

    saved_preds = {
        r["sample_id"]: r["normalized_answer"]
        for r in read_jsonl(Path(f"/vol/artifacts/runs/{adapter_run}_test/predictions.jsonl"))
    }
    matches = 0
    details = []
    for rec in test_records:
        pred = predict_record(
            model=model,
            processor=processor,
            rec=rec,
            run_id=f"{adapter_run}_p7_repro",
            model_id=adapter_run,
            mode="B-R",
        )
        ok = pred.normalized_answer == saved_preds.get(rec.sample_id)
        if ok:
            matches += 1
        details.append({"sample_id": rec.sample_id, "repro": pred.normalized_answer, "saved": saved_preds.get(rec.sample_id), "match": ok})
    return {"adapter_run": adapter_run, "n_samples": n_samples, "exact_matches": matches, "details": details}


@app.local_entrypoint()
def main(
    action: str = "p3",
    run_id: str = "",
    objective_mode: str = "sft",
    lambda_cond: float = 0.0,
    seed: int = 42,
    epochs: int = 1,
    adapter_runs: str = "",
    selected_mc_run: str = "",
    include_baselines: bool = True,
):
    if action == "p3":
        res = run_p3_profile_and_dev_baselines.remote()
        print(json.dumps(res, indent=2))
    elif action == "p5":
        res = run_p5_training_and_dev_eval.remote(
            run_id=run_id,
            objective_mode=objective_mode,
            lambda_cond=lambda_cond,
            seed=seed,
            epochs=epochs,
        )
        print(json.dumps(res, indent=2))
    elif action == "p6":
        runs = [x.strip() for x in adapter_runs.split(",") if x.strip()]
        res = run_p6_final_eval.remote(
            adapter_runs=runs,
            selected_mc_run=selected_mc_run,
            run_stress=True,
            include_baselines=include_baselines,
        )
        print(json.dumps(res, indent=2))
    elif action == "p7":
        res = run_p7_repro_subset.remote()
        print(json.dumps(res, indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
