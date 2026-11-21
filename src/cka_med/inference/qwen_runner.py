"""Qwen2.5-VL-3B-Instruct adapter loading, image verification, and constrained/short inference."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from PIL import Image
import torch

from cka_med.inference.prompts import render_prompt
from cka_med.inference.scoring import normalize_answer
from cka_med.schema import DataRecord, PredictionRecord


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def select_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_qwen_model_and_processor(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    adapter_path: str | Path | None = None,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
    use_4bit: bool = False,
):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = select_torch_dtype()
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=model_revision,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    load_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    # Freeze vision encoder strictly
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    return model, processor, dtype


def get_binary_choice_token_ids(processor) -> tuple[int, int]:
    tok_a = processor.tokenizer.encode("A", add_special_tokens=False)
    tok_b = processor.tokenizer.encode("B", add_special_tokens=False)
    if len(tok_a) != 1 or len(tok_b) != 1:
        raise RuntimeError(f"Expected single-token A/B mapping, got A={tok_a}, B={tok_b}")
    return int(tok_a[0]), int(tok_b[0])


def prepare_single_input(
    processor,
    *,
    prompt_text: str,
    image_path: str | Path | None,
    uses_image: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if uses_image:
        if image_path is None:
            raise ValueError("uses_image=True requires an image_path")
        img_p = Path(image_path)
        if "mask" in img_p.name.lower():
            raise ValueError(f"Forbidden mask image path: {img_p}")
        with Image.open(img_p) as im:
            pil_img = im.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=[pil_img], padding=True, return_tensors="pt")
        if "pixel_values" not in inputs or inputs["pixel_values"].numel() == 0:
            raise RuntimeError(f"Image tensor missing or empty for multimodal sample {img_p}")
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], padding=True, return_tensors="pt")

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


@torch.no_grad()
def predict_record(
    *,
    model,
    processor,
    rec: DataRecord,
    run_id: str,
    model_id: str,
    mode: str,
    prompt_variant: str = "default",
    rendering_version: str | None = None,
    allow_oracle: bool = False,
    max_new_tokens_j: int = 24,
) -> PredictionRecord:
    t0 = time.perf_counter()
    id_a, id_b = get_binary_choice_token_ids(processor)
    device = next(model.parameters()).device

    model_inp = rec.to_model_input_dict(allow_oracle=allow_oracle)
    prompt_text, meta = render_prompt(
        model_inp,
        mode=mode,
        prompt_variant=prompt_variant,
        rendering_version=rendering_version or rec.rendering_version,
        allow_oracle=allow_oracle,
    )
    inputs = prepare_single_input(
        processor,
        prompt_text=prompt_text,
        image_path=rec.image_path,
        uses_image=meta["uses_image"],
        device=device,
    )

    if rec.task_type == "V":
        out = model(**inputs)
        last_logits = out.logits[0, -1, [id_a, id_b]].float()
        probs = torch.softmax(last_logits, dim=-1)
        p_yes = float(probs[0].item())
        p_no = float(probs[1].item())
        raw_out = "A" if p_yes >= p_no else "B"
        norm_ans, abstained, valid = normalize_answer(raw_out, task_type="V")
        scores = {"yes": p_yes, "no": p_no}
    else:
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_j,
            do_sample=False,
        )
        new_tokens = gen_ids[0, prompt_len:]
        raw_out = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        norm_ans, abstained, valid = normalize_answer(
            raw_out,
            task_type=rec.task_type,
            allowed_vocab=rec.answer_choices or None,
        )
        scores = None

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return PredictionRecord(
        run_id=run_id,
        model_id=model_id,
        group_id=rec.group_id,
        sample_id=rec.sample_id,
        image_id=rec.image_id,
        condition=rec.condition,
        task_type=rec.task_type,
        context_pack_id=rec.context_pack_id,
        oracle_context=rec.oracle_context,
        raw_output=raw_out,
        normalized_answer=norm_ans,
        label_scores=scores,
        abstained=abstained,
        valid_output=valid,
        error=None,
        latency_ms=elapsed_ms,
    )
