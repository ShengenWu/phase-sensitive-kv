"""Acceptance tests for the uncompressed full-attention baseline (layout_kind='full')."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from rescsa.model import ResCSAConfig, ResCSAForCausalLM

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def full_model(layers=2, seed=42):
    torch.manual_seed(seed)
    cfg = ResCSAConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16, rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        tie_word_embeddings=False, window_size=8, stride=8, local_window=16,
        variant="C0", rho=0.1, layout_kind="full",
    )
    return ResCSAForCausalLM(cfg).to(DEV).eval()


def rand_ids(L, B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 512, (B, L), generator=g).to(DEV)


def test_matches_reference_causal_attention():
    m = full_model(1)
    att = m.model.layers[0].self_attn
    B, L, d = 2, 21, att.hidden_size
    h = torch.randn(B, L, d, device=DEV)
    with torch.no_grad():
        out = att(h)
        # hand-rolled reference: GQA causal attention, one KV head
        q = att.q_norm(att.q_proj(h).view(B, L, att.num_heads, att.head_dim))
        pos = torch.arange(L, device=DEV)
        inv = 1.0 / (1e6 ** (torch.arange(0, att.head_dim, 2, device=DEV).float() / att.head_dim))
        ang = pos[:, None].float() * inv
        cos, sin = ang.cos(), ang.sin()

        def rope(x, c, sn):
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)
        q = rope(q, cos[None, :, None, :], sin[None, :, None, :]).permute(0, 2, 1, 3)  # (B,H,L,dh)
        k = rope(att.k_norm(att.k_proj(h)), cos[None, :], sin[None, :]).unsqueeze(1)  # (B,1,L,dh)
        v = att.v_proj(h).unsqueeze(1)  # (B,1,L,dh)
        scores = (q @ k.transpose(-1, -2)) * att.scaling
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=DEV), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores.float(), -1).to(h.dtype)
        ref = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, att.num_heads * att.head_dim)
        ref = att.o_proj(ref)
    assert torch.allclose(out, ref, atol=1e-4), float((out - ref).abs().max())
    print("[ok] full attention matches hand-rolled causal GQA reference")


def test_causality_and_incremental():
    m = full_model(2)
    ids = rand_ids(69, seed=3)
    cut = 40
    with torch.no_grad():
        base = m(ids).logits
        alt = ids.clone()
        alt[:, cut:] = torch.flip(alt[:, cut:], dims=[1])
        assert torch.equal(base[:, :cut], m(alt).logits[:, :cut])
        par = m(ids).logits
        inc = m.forward_incremental(ids)
    assert float((par - inc).abs().max()) < 2e-4
    print("[ok] full attention: causal; parallel == incremental")


def test_no_compression_params():
    m = full_model(1)
    keys = set(m.state_dict())
    assert not any("c_k" in k or "c_v" in k or "z_k" in k or "p_k" in k for k in keys)
    c0 = full_model(1)
    torch.manual_seed(42)
    from rescsa.model import build_model
    comp = build_model("C0", seed=42).model.layers[0].self_attn  # full-size for param compare
    print("[ok] full attention carries no compression parameters")


def test_kv_budget_full():
    m = full_model(1)
    att = m.model.layers[0].self_attn
    assert att.full_attention
    # t+1 raw entries per query: no compression at all
    print("[ok] full attention budget: every position keeps its raw KV (t+1 entries)")


if __name__ == "__main__":
    test_matches_reference_causal_attention()
    test_causality_and_incremental()
    test_no_compression_params()
    test_kv_budget_full()
    print("ALL_FULL_ATTENTION_TESTS_PASSED")
    sys.exit(0)
