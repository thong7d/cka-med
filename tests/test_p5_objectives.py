"""P5 scientific & numerical unit checks for conditional loss, masks, gradients, and lambda=0 equivalence."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cka_med.training.objectives import (
    compute_answer_only_token_nll,
    compute_binary_choice_nll_and_probs,
    compute_cka_loss,
    compute_conditional_mask,
    jensen_shannon_divergence,
)


def test_answer_only_token_mask_ignores_prompt_tokens():
    # Batch=1, seq=5, vocab=4. Only last position (label index 4 -> shifted index 3) is active.
    logits = torch.zeros((1, 5, 4), dtype=torch.float32)
    logits[0, 3, 2] = 10.0  # predicts token 2 strongly at answer pos
    labels_masked = torch.tensor([[-100, -100, -100, -100, 2]])
    loss = compute_answer_only_token_nll(logits, labels_masked)
    assert loss.item() < 0.05

    # Changing prompt logits (positions 0..2) must NOT affect answer-only loss
    logits2 = logits.clone()
    logits2[0, :3, :] = 50.0
    loss2 = compute_answer_only_token_nll(logits2, labels_masked)
    assert torch.isclose(loss, loss2)


def test_conditional_mask_rules():
    mask_mc = compute_conditional_mask(
        task_types=["V", "V", "V", "J", "V"],
        conditions=["Knear", "Kirrel", "K0", "Khelp", "Knear"],
        gold_invariants=[True, True, True, True, False],
        provenance_qualities=["accepted", "accepted", "accepted", "accepted", "accepted"],
        unconditional=False,
    )
    assert mask_mc.tolist() == [True, True, False, False, False]


def test_empty_mask_returns_scalar_zero_and_no_nan():
    logits_k0 = torch.tensor([[2.0, 0.5], [0.1, 1.5]], requires_grad=True)
    logits_kk = torch.tensor([[1.8, 0.7], [0.2, 1.3]], requires_grad=True)
    targets = torch.tensor([0, 1])

    loss0, probs0 = compute_binary_choice_nll_and_probs(logits_k0, targets)
    lossk, probsk = compute_binary_choice_nll_and_probs(logits_kk, targets)
    empty_mask = torch.tensor([False, False], dtype=torch.bool)

    out = compute_cka_loss(
        ans_losses=[loss0, lossk],
        probs_k0=probs0,
        probs_kk=probsk,
        cond_mask=empty_mask,
        lambda_cond=0.5,
        device=logits_k0.device,
        dtype=logits_k0.dtype,
    )
    assert out.active_count == 0
    assert out.cond_loss.item() == 0.0
    assert not torch.isnan(out.total_loss)
    out.total_loss.backward()
    assert logits_k0.grad is not None and torch.isfinite(logits_k0.grad).all()


def test_js_gradient_flows_to_both_branches():
    lin = nn.Linear(4, 2)
    x0 = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    xk = torch.tensor([[0.2, 0.9, -0.3, 0.1]])
    p0 = torch.softmax(lin(x0), dim=-1)
    pk = torch.softmax(lin(xk), dim=-1)
    js = jensen_shannon_divergence(p0, pk).mean()
    assert js.item() > 0.0
    js.backward()
    assert lin.weight.grad is not None
    assert torch.isfinite(lin.weight.grad).all()
    assert lin.weight.grad.abs().sum().item() > 0.0


def test_lambda_zero_matches_sft_numerically():
    torch.manual_seed(42)
    model_bs = nn.Linear(6, 2)
    model_mc = nn.Linear(6, 2)
    model_mc.load_state_dict(model_bs.state_dict())

    x0 = torch.randn(4, 6)
    xk = torch.randn(4, 6)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.tensor([True, True, False, True], dtype=torch.bool)

    # B-S forward + backward
    l0_bs, _ = compute_binary_choice_nll_and_probs(model_bs(x0), y)
    lk_bs, _ = compute_binary_choice_nll_and_probs(model_bs(xk), y)
    out_bs = 0.5 * (l0_bs + lk_bs)
    out_bs.backward()

    # M-C with lambda=0.0 forward + backward on same batch
    l0_mc, p0_mc = compute_binary_choice_nll_and_probs(model_mc(x0), y)
    lk_mc, pk_mc = compute_binary_choice_nll_and_probs(model_mc(xk), y)
    out_mc = compute_cka_loss(
        ans_losses=[l0_mc, lk_mc],
        probs_k0=p0_mc,
        probs_kk=pk_mc,
        cond_mask=mask,
        lambda_cond=0.0,
        device=x0.device,
        dtype=x0.dtype,
    )
    out_mc.total_loss.backward()

    assert torch.allclose(out_bs, out_mc.total_loss, atol=1e-7)
    assert torch.allclose(model_bs.weight.grad, model_mc.weight.grad, atol=1e-7)
    assert torch.allclose(model_bs.bias.grad, model_mc.bias.grad, atol=1e-7)
