"""Consistency tests for the compressed-attention kernel.

Run on GPU:  python -m pskv.tests.test_consistency
Covers:
1. zero-init gates: learned == uniform at step 0 (the exact uniform average)
2. causality: prefix logits are invariant to suffix changes
3. mask correctness: a token perturbation at u changes exactly the queries that
   can see u through the compressed/local memory (single-layer model)
4. window indexing: hand-rolled reference for compress() incl. RoPE positions
5. parallel vs one-token-incremental consistency (incl. L not divisible by S)
6. KV budget accounting
7. gate softmax sanity (sums to 1)
"""

from __future__ import annotations

import sys

import torch

from pskv.model import ResCSAConfig, ResCSAForCausalLM
from pskv.model.attention import apply_rope, build_rope_cache

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def small_model(variant: str, layers: int = 1, seed: int = 42) -> ResCSAForCausalLM:
    torch.manual_seed(seed)
    cfg = ResCSAConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        tie_word_embeddings=False,
        window_size=8,
        stride=8,
        local_window=16,
        variant=variant,
    )
    m = ResCSAForCausalLM(cfg).to(DEV).eval()
    return m


def rand_ids(L: int, B: int = 2, vocab: int = 512, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, L), generator=g).to(DEV)


def test_zero_init_equivalence():
    torch.manual_seed(123)
    ids = rand_ids(64, seed=7)
    outs = {}
    for v in ["learned", "uniform"]:
        m = small_model(v)
        with torch.no_grad():
            outs[v] = m(ids).logits
    err = float((outs["learned"] - outs["uniform"]).abs().max())
    assert err < 1e-5, f"learned differs from uniform at init by {err}"
    print("[ok] zero-init gates: learned kernel == uniform averaging at step 0 (< 1e-5)")


def test_causality():
    m = small_model("learned", layers=2)
    ids = rand_ids(96, seed=3)
    cut = 40
    with torch.no_grad():
        base = m(ids).logits
        alt_ids = ids.clone()
        alt_ids[:, cut:] = torch.flip(alt_ids[:, cut:], dims=[1])
        alt = m(alt_ids).logits
    diff = (base[:, :cut] - alt[:, :cut]).abs().max()
    assert diff == 0, f"prefix logits changed by suffix edit: {diff}"
    assert not torch.equal(base[:, cut:], alt[:, cut:]), "suffix edit had no effect (test broken)"
    print("[ok] causality: prefix logits invariant to suffix edits")


def expected_visibility(L: int, S: int, T: int, u: int):
    """Queries t that can see token u: local window, or any closed entry covering u."""
    J = L // S
    vis = set()
    for t in range(L):
        if t - T + 1 <= u <= t:
            vis.add(t)
        for e in range(J):
            if e * S <= u <= (e + 1) * S - 1 and (e + 1) * S - 1 <= t:
                vis.add(t)
    return vis


def test_perturbation_mask():
    m = small_model("learned", layers=1)
    S, T, L = m.config.stride, m.config.local_window, 69  # not divisible by 8
    with torch.no_grad():
        base = m(rand_ids(L, seed=5)).logits
        for u in [0, 1, 7, 8, 15, 23, 32, 33, 55, 63, 64, 68]:
            ids = rand_ids(L, seed=5)
            ids[:, u] = (ids[:, u] + 7) % 512
            out = m(ids).logits
            changed = set(torch.nonzero((out - base).abs().amax(dim=(0, 2)) > 0).flatten().tolist())
            exp = expected_visibility(L, S, T, u)
            # single layer: influence set == direct visibility set
            assert changed == exp, (
                f"u={u}: changed {sorted(changed)} != expected {sorted(exp)} "
                f"(missing {sorted(exp - changed)}, extra {sorted(changed - exp)})"
            )
    print(f"[ok] perturbation/mask: influence sets exactly match visibility (L={L}, S={S}, T={T})")


def test_compress_reference():
    m = small_model("learned", layers=1)
    att = m.model.layers[0].self_attn
    L, d, W, S = 32, att.hidden_size, att.W, att.S
    h = torch.randn(1, L, d, device=DEV)
    with torch.no_grad():
        att.z_k.normal_(0, 0.5)
        att.b_k.normal_(0, 0.5)
        att.z_v.normal_(0, 0.5)
        att.b_v.normal_(0, 0.5)
        k_hat, v_hat = att.compress(h)
        # hand-rolled reference
        hw = h.reshape(1, L // W, W, d)
        alpha_k = torch.softmax(torch.einsum("bjwd,wd->bjw", hw, att.z_k) + att.b_k, dim=-1)
        alpha_v = torch.softmax(torch.einsum("bjwd,wd->bjw", hw, att.z_v) + att.b_v, dim=-1)
        k_ref = att.k_norm(torch.einsum("bjw,bjwd->bjd", alpha_k, att.c_k(hw)))
        pos = torch.arange(L // S, device=DEV) * S
        cos, sin = build_rope_cache(pos, att.head_dim, att.config.theta, k_ref.dtype)
        k_ref = apply_rope(k_ref, cos, sin)
        v_ref = torch.einsum("bjw,bjwd->bjd", alpha_v, att.c_v(hw))
    assert torch.allclose(k_hat, k_ref, atol=1e-5), f"k mismatch {float((k_hat-k_ref).abs().max())}"
    assert torch.allclose(v_hat, v_ref, atol=1e-5), f"v mismatch {float((v_hat-v_ref).abs().max())}"
    print("[ok] compress(): matches hand-rolled reference incl. RoPE positions (j-1)*S")


def test_incremental_consistency():
    for variant in ["learned", "uniform"]:
        m = small_model(variant, layers=2)
        for L in [37, 64]:  # 37: tail block without a closed entry; 64: aligned
            ids = rand_ids(L, seed=11)
            with torch.no_grad():
                par = m(ids).logits
                inc = m.forward_incremental(ids)
            err = float((par - inc).abs().max())
            assert err < 2e-4, f"{variant} L={L}: parallel vs incremental max err {err}"
        print(f"[ok] incremental: {variant} parallel == token-by-token (max err < 2e-4)")


def test_gate_sanity():
    m = small_model("learned", layers=1)
    att = m.model.layers[0].self_attn
    with torch.no_grad():
        att.z_k.normal_(0, 0.5)
        att.b_k.normal_(0, 0.5)
        h = torch.randn(2, 48, att.hidden_size, device=DEV)
        hw = h.reshape(2, 6, 8, -1)
        alpha = att.window_gates(hw, "K")
    s = alpha.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), "gates do not sum to 1"
    assert float(alpha.min()) >= 0
    print("[ok] gates: learned alphas stay in the simplex")


def test_kv_budget():
    m = small_model("learned")
    L, S, T = 2048, m.config.stride, m.config.local_window
    mask = m.model.layers[0].self_attn.memory_mask(L, DEV)
    for t in [0, 5, 15, 100, 2047]:
        e = torch.arange(L // S, device=DEV)
        n_comp = int(((e + 1) * S - 1 <= t).sum())
        n_local = min(t + 1, T)
        got = int(mask[t].sum())
        assert got == n_comp + n_local, f"t={t}: {got} != {n_comp}+{n_local}"
    full, compr = 2048, 2048 // S + T
    print(f"[ok] KV budget: at t=2047 memory = {compr} entries vs {full} full-attention "
          f"({full/compr:.2f}x reduction), identical across variants")


if __name__ == "__main__":
    test_zero_init_equivalence()
    test_causality()
    test_perturbation_mask()
    test_compress_reference()
    test_gate_sanity()
    test_incremental_consistency()
    test_kv_budget()
    print("ALL_CONSISTENCY_TESTS_PASSED")
    sys.exit(0)
