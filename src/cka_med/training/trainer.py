"""Shared LoRA trainer for B-S (same-data SFT), M-C (conditional), and A-U (unconditional)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from cka_med.inference.prompts import render_prompt
from cka_med.inference.qwen_runner import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    get_binary_choice_token_ids,
    load_qwen_model_and_processor,
    prepare_single_input,
)
from cka_med.manifest import build_run_manifest, save_json
from cka_med.schema import DataRecord, read_jsonl
from cka_med.training.dataset import build_paired_training_items
from cka_med.training.objectives import (
    compute_answer_only_token_nll,
    compute_binary_choice_nll_and_probs,
    compute_cka_loss,
    compute_conditional_mask,
)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_j_supervised_inputs(processor, rec: DataRecord, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    model_inp = rec.to_model_input_dict(allow_oracle=False)
    prompt_text, _ = render_prompt(model_inp, mode="B-R")
    with Image.open(rec.image_path) as im:
        pil_img = im.convert("RGB")
    user_msgs = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt_text}]}]
    full_msgs = user_msgs + [{"role": "assistant", "content": [{"type": "text", "text": rec.original_answer}]}]

    prompt_chat = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    full_chat = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prompt_inputs = processor(text=[prompt_chat], images=[pil_img], return_tensors="pt")
    full_inputs = processor(text=[full_chat], images=[pil_img], return_tensors="pt")
    if "pixel_values" not in full_inputs or full_inputs["pixel_values"].numel() == 0:
        raise RuntimeError(f"Missing pixel_values in training J sample {rec.sample_id}")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_inputs.items()}
    return full_dev, labels.to(device), prompt_len


def run_training(cfg: dict[str, Any], *, objective_mode: str) -> dict[str, Any]:
    """Run fair training for 'sft' (B-S), 'conditional' (M-C), or 'unconditional' (A-U)."""
    seed = int(cfg.get("seed", 42))
    _set_all_seeds(seed)

    run_id = cfg["run_id"]
    out_dir = Path(cfg.get("output_dir", f"artifacts/runs/{run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    ckpt_dir = out_dir / "checkpoint_state"

    train_records = [DataRecord.from_dict(r) for r in read_jsonl(Path(cfg["train_data_path"]))]
    limit_groups = int(cfg.get("limit_groups", 0))
    if limit_groups > 0:
        kept_gids = sorted({r.group_id for r in train_records})[:limit_groups]
        kept_set = set(kept_gids)
        train_records = [r for r in train_records if r.group_id in kept_set]

    pairs = build_paired_training_items(train_records, seed=seed)

    model_id = cfg.get("base_model_id", DEFAULT_MODEL_ID)
    model_rev = cfg.get("base_model_revision", DEFAULT_MODEL_REVISION)
    model, processor, dtype = load_qwen_model_and_processor(
        model_id=model_id,
        model_revision=model_rev,
        min_pixels=int(cfg.get("min_pixels", 256 * 28 * 28)),
        max_pixels=int(cfg.get("max_pixels", 512 * 28 * 28)),
        use_4bit=bool(cfg.get("use_4bit", False)),
    )

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if bool(cfg.get("use_4bit", False)):
        model = prepare_model_for_kbit_training(model)

    lora_targets = cfg.get(
        "lora_targets",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    # Explicitly restrict LoRA target modules to language-model layers ONLY (never visual encoder)
    explicit_targets = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and "visual" not in name
        and "lm_head" not in name
        and any(name.endswith(f".{t}") for t in lora_targets)
    ]
    if not explicit_targets:
        raise RuntimeError(f"No language-model target modules matched {lora_targets}")
    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_rank", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=explicit_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Double-check frozen vision encoder: NO trainable visual parameters
    trainable_visual = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    if trainable_visual:
        raise RuntimeError(f"Vision encoder parameters unexpectedly trainable: {trainable_visual[:5]}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters found.")

    lr = float(cfg.get("learning_rate", 1e-4))
    grad_accum = int(cfg.get("grad_accum_steps", 8))
    epochs = int(cfg.get("epochs", 1))
    max_steps = int(cfg.get("max_steps", 0))  # 0 = full epochs
    lambda_cond = float(cfg.get("lambda_cond", 0.0 if objective_mode == "sft" else 0.5))
    if objective_mode == "sft":
        lambda_cond = 0.0

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.01)))
    total_pair_steps = len(pairs) * epochs
    total_opt_steps = max(1, math.ceil(total_pair_steps / grad_accum))
    if max_steps > 0:
        total_opt_steps = min(total_opt_steps, max_steps)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=lr * 0.1)

    start_epoch = 0
    start_pair_idx = 0
    opt_step = 0
    resume_state_file = ckpt_dir / "trainer_state.pt"
    if bool(cfg.get("resume", False)) and resume_state_file.exists():
        state = torch.load(resume_state_file, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        start_pair_idx = int(state["pair_idx"])
        opt_step = int(state["opt_step"])
        torch.set_rng_state(state["torch_rng"])

    device = next(model.parameters()).device
    id_a, id_b = get_binary_choice_token_ids(processor)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable") and bool(cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    logs: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    exposure_records = 0
    accum_ans = 0.0
    accum_cond = 0.0
    accum_active = 0
    accum_items = 0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        for pair_idx, item in enumerate(pairs):
            if epoch == start_epoch and pair_idx < start_pair_idx:
                continue
            if max_steps > 0 and opt_step >= max_steps:
                break

            rec_k0: DataRecord = item["rec_k0"]
            rec_kk: DataRecord = item["rec_kk"]
            exposure_records += 2

            if item["task_type"] == "V":
                y_idx = torch.tensor([0 if rec_k0.normalized_answer == "yes" else 1], device=device, dtype=torch.long)
                p0_txt, m0 = render_prompt(rec_k0.to_model_input_dict(allow_oracle=False), mode="B-R")
                pk_txt, mk = render_prompt(rec_kk.to_model_input_dict(allow_oracle=False), mode="B-R")
                inp0 = prepare_single_input(processor, prompt_text=p0_txt, image_path=rec_k0.image_path, uses_image=m0["uses_image"], device=device)
                inpk = prepare_single_input(processor, prompt_text=pk_txt, image_path=rec_kk.image_path, uses_image=mk["uses_image"], device=device)

                out0 = model(**inp0)
                outk = model(**inpk)
                l0, probs0 = compute_binary_choice_nll_and_probs(out0.logits[:, -1, [id_a, id_b]], y_idx)
                lk, probsk = compute_binary_choice_nll_and_probs(outk.logits[:, -1, [id_a, id_b]], y_idx)

                cond_mask = compute_conditional_mask(
                    task_types=["V"],
                    conditions=[rec_kk.condition],
                    gold_invariants=[rec_kk.gold_invariant],
                    provenance_qualities=[rec_kk.provenance_quality],
                    unconditional=(objective_mode == "unconditional"),
                )
                loss_out = compute_cka_loss(
                    ans_losses=[l0, lk],
                    probs_k0=probs0,
                    probs_kk=probsk,
                    cond_mask=cond_mask,
                    lambda_cond=lambda_cond,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                inp0, lbl0, plen0 = _prepare_j_supervised_inputs(processor, rec_k0, device)
                inpk, lblk, plenk = _prepare_j_supervised_inputs(processor, rec_kk, device)
                out0 = model(**inp0)
                outk = model(**inpk)
                l0 = compute_answer_only_token_nll(out0.logits, lbl0)
                lk = compute_answer_only_token_nll(outk.logits, lblk)
                if objective_mode == "unconditional":
                    # Ablation A-U: unconditionally enforce consistency on J (first answer token) as well
                    probs0_j = torch.softmax(out0.logits[:, plen0 - 1, :].float(), dim=-1)
                    probsk_j = torch.softmax(outk.logits[:, plenk - 1, :].float(), dim=-1)
                    j_mask = torch.tensor([True], dtype=torch.bool)
                    loss_out = compute_cka_loss(
                        ans_losses=[l0, lk],
                        probs_k0=probs0_j,
                        probs_kk=probsk_j,
                        cond_mask=j_mask,
                        lambda_cond=lambda_cond,
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    empty_mask = torch.tensor([False], dtype=torch.bool)
                    loss_out = compute_cka_loss(
                        ans_losses=[l0, lk],
                        probs_k0=None,
                        probs_kk=None,
                        cond_mask=empty_mask,
                        lambda_cond=lambda_cond,
                        device=device,
                        dtype=torch.float32,
                    )

            scaled_loss = loss_out.total_loss / float(grad_accum)
            scaled_loss.backward()

            accum_ans += float(loss_out.ans_loss.item())
            accum_cond += float(loss_out.cond_loss.item())
            accum_active += int(loss_out.active_count)
            accum_items += 1

            is_last_in_epoch = (pair_idx + 1) == len(pairs)
            if (accum_items % grad_accum == 0) or is_last_in_epoch:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg.get("max_grad_norm", 1.0))).item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                log_row = {
                    "opt_step": opt_step,
                    "epoch": epoch,
                    "pair_idx": pair_idx,
                    "ans_loss": accum_ans / max(1, accum_items),
                    "cond_loss": accum_cond / max(1, accum_items),
                    "active_mask_count": accum_active,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm": grad_norm,
                    "exposure_records": exposure_records,
                    "elapsed_s": time.perf_counter() - t_start,
                    "max_vram_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0,
                }
                logs.append(log_row)
                accum_ans = 0.0
                accum_cond = 0.0
                accum_active = 0
                accum_items = 0

                if opt_step % int(cfg.get("save_every_steps", 50)) == 0:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(adapter_dir)
                    processor.save_pretrained(adapter_dir)
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            "pair_idx": pair_idx + 1,
                            "opt_step": opt_step,
                            "torch_rng": torch.get_rng_state(),
                        },
                        resume_state_file,
                    )

        if max_steps > 0 and opt_step >= max_steps:
            break

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    save_json(out_dir / "train_logs.json", logs)

    summary = {
        "run_id": run_id,
        "objective_mode": objective_mode,
        "lambda_cond": lambda_cond,
        "opt_steps": opt_step,
        "exposure_records": exposure_records,
        "n_pairs": len(pairs),
        "wall_time_s": time.perf_counter() - t_start,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0,
        "adapter_dir": adapter_dir.as_posix(),
        "final_ans_loss": logs[-1]["ans_loss"] if logs else None,
        "final_cond_loss": logs[-1]["cond_loss"] if logs else None,
    }
    manifest = build_run_manifest(
        run_id=run_id,
        model_id=objective_mode,
        model_revision=model_rev,
        split="train",
        seed=seed,
        config=cfg,
        data_manifest_path=cfg.get("data_manifest_path"),
        extra=summary,
    )
    save_json(out_dir / "run_manifest.json", manifest)
    return summary
