"""Conditional consistency and answer-only loss objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ConditionalLossOutput:
    total_loss: torch.Tensor
    ans_loss: torch.Tensor
    cond_loss: torch.Tensor
    active_count: int


def compute_conditional_mask(
    *,
    task_types: list[str],
    conditions: list[str],
    gold_invariants: list[bool],
    provenance_qualities: list[str],
    unconditional: bool = False,
) -> torch.Tensor:
    """Compute boolean mask(g, k) for consistency regularization.

    Conditional (M-C):
      task_is_visual AND gold_answer_is_unchanged AND provenance_accepted AND k in {Kirrel, Knear}
    Unconditional ablation (A-U):
      gold_answer_is_unchanged AND provenance_accepted AND k != K0 (includes J/Khelp if binary or matched)
    """
    flags: list[bool] = []
    for t, k, inv, pq in zip(task_types, conditions, gold_invariants, provenance_qualities):
        accepted = pq in ("accepted", "synthetic_fixture")
        if unconditional:
            flags.append((t == "V") and bool(inv) and accepted and (k != "K0"))
        else:
            flags.append((t == "V") and bool(inv) and accepted and (k in ("Kirrel", "Knear")))
    return torch.tensor(flags, dtype=torch.bool)


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric JS divergence between two batched probability distributions [N, C].

    Keeps gradients flowing through both p and q.
    """
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    m = 0.5 * (p_safe + q_safe)
    kl_pm = torch.sum(p_safe * (torch.log(p_safe) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q_safe * (torch.log(q_safe) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def compute_answer_only_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard causal-LM next-token NLL restricted strictly to non-(-100) answer tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels != -100
    if not torch.any(valid_mask):
        return logits.sum() * 0.0
    loss_per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss_per_tok


def compute_binary_choice_nll_and_probs(
    logits_at_answer: torch.Tensor,
    target_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2-way answer CE and probabilities over constrained [A, B] logits [N, 2]."""
    probs = F.softmax(logits_at_answer.float(), dim=-1)
    loss = F.cross_entropy(logits_at_answer.float(), target_indices, reduction="mean")
    return loss, probs


def compute_cka_loss(
    *,
    ans_losses: list[torch.Tensor],
    probs_k0: torch.Tensor | None,
    probs_kk: torch.Tensor | None,
    cond_mask: torch.Tensor,
    lambda_cond: float,
    device: torch.device,
    dtype: torch.dtype,
) -> ConditionalLossOutput:
    """Combine L_ans and L_cond safely, returning exact scalar zero on empty mask."""
    if not ans_losses:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return ConditionalLossOutput(total_loss=zero, ans_loss=zero, cond_loss=zero, active_count=0)

    l_ans = torch.stack([x.float() for x in ans_losses]).mean()
    active_count = int(cond_mask.sum().item()) if cond_mask.numel() > 0 else 0

    if active_count == 0 or probs_k0 is None or probs_kk is None:
        l_cond = torch.zeros((), device=device, dtype=torch.float32)
    else:
        mask_dev = cond_mask.to(device=probs_k0.device)
        js_vals = jensen_shannon_divergence(probs_k0[mask_dev], probs_kk[mask_dev])
        l_cond = js_vals.mean()

    if float(lambda_cond) == 0.0:
        l_total = l_ans
    else:
        l_total = l_ans + float(lambda_cond) * l_cond

    return ConditionalLossOutput(
        total_loss=l_total,
        ans_loss=l_ans,
        cond_loss=l_cond,
        active_count=active_count,
    )
