"""Compressed attention with chunked KV compression (paper reference kernel + variants).

Implements Appendix D.1/D.2 of arXiv 2609.36322 for non-overlapping windows (W == S):

- payload maps C^K, C^V in R^{dh x d}, shared across window offsets
- scalar gates: logits s^B_{j,i} = <Z^B_i, h_{j,i}> + b^B_i with Z^B_i in R^d,
  softmax over the W offsets of the window (learned in C0/C2/C3, fixed 1/W in C1)
- compressed key  k_hat_j = RoPE_{(j-1)S}( RMSNorm_K( K_bar_j ) ),  value v_hat_j = V_bar_j
- local raw window of exactly T=16 tokens: k_u = RoPE_u(RMSNorm_K(W_K h_u)), v_u = W_V h_u
- visibility: every causally closed entry (jS - 1 <= t) plus the local window,
  normalized by ONE joint softmax (SDPA over the concatenated memory axis)
- RMSNorm_K is shared between the local and compressed key paths (single reading of
  the paper's notation; both write "RMSNorm_K")

Variants (see config.py). Mixing for C2/C3 happens before key normalization and RoPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# cuDNN SDPA handles the (L, J+L) additive-bias attention memory-efficiently, but the
# dispatcher skips it unless globally enabled (math backend would retain full attention
# weights: ~5 GB per layer at B=4/L=2048 instead of ~0.3 GB).
torch.backends.cuda.enable_cudnn_sdp(True)


def _ac_dtype(t: torch.Tensor) -> torch.dtype:
    """Target dtype for tensors entering SDPA (cuDNN requires low-precision inputs).

    RMSNorm sits on autocast's fp32 list, so its output would otherwise leak fp32
    into the attention call and force the math backend (fp32 weights, ~GB per layer).
    """
    if torch.is_autocast_enabled():
        return torch.get_autocast_dtype(t.device.type)
    return t.dtype


_MASK_CACHE: dict = {}  # (L, S, T, device, dtype) -> additive attention bias


def build_rope_cache(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """cos/sin tables for split-half (rotate_half) RoPE at the given positions.

    positions: any-shape long tensor. Returns cos, sin of shape (*positions.shape, head_dim // 2),
    computed in fp32 and cast to `dtype`.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    angles = positions[..., None].float() * inv_freq  # (*, hd/2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Split-half rotary embedding; x (..., head_dim), cos/sin (..., head_dim // 2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class ResCSAAttention(nn.Module):
    """Qwen3-style GQA attention with a chunked-compression KV memory.

    Query heads: config.num_attention_heads (16). KV heads: config.num_key_value_heads (1).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.variant = config.variant
        self.W = config.window_size
        self.S = config.stride
        self.T = config.local_window
        self.rho = config.rho
        # getattr preserves the historical meaning of pickled legacy configs.
        self.c3_mode = getattr(config, "c3_mode", "mix")
        self.residual_scale = getattr(config, "residual_scale", 0.1)
        self.residual_scope = getattr(config, "residual_scope", "kv")
        self.record_branch_stats = False
        self.branch_stats = {}
        # uncompressed control (the paper's full-attention baseline): plain causal
        # GQA over raw K/V; no compression parameters are created at all
        self.full_attention = (getattr(config, "layout_kind", "fixed") == "full")

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        if self.num_kv_heads != 1:
            raise ValueError("round-1 kernel implements one KV head per layer")

        # query / local-key / local-value projections (local raw tokens)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if not self.full_attention:
            # compression payload maps, shared across offsets (C^K_i = C^K)
            self.c_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.c_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # Zero initialization gives uniform learned gates; residual mode/init
        # separately determines the complete initial compressed representation.
        if self.variant in {"C0", "C2", "C3"} and not self.full_attention:
            self.z_k = nn.Parameter(torch.zeros(self.W, self.hidden_size))
            self.z_v = nn.Parameter(torch.zeros(self.W, self.hidden_size))
            self.b_k = nn.Parameter(torch.zeros(self.W))
            self.b_v = nn.Parameter(torch.zeros(self.W))
        if self.variant == "C3" and not self.full_attention:
            # P^B: copy/zero initialized by modeling, then trained
            # independently. Construction AND post-init re-randomization are RNG-neutral so
            # that every parameter shared with C0/C2 keeps its paired value.
            rng = torch.get_rng_state()
            self.p_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.p_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            torch.set_rng_state(rng)
            self.p_k._rescsa_paired_neutral = True
            self.p_v._rescsa_paired_neutral = True
            if self.residual_scope == "v":
                self.p_k.requires_grad_(False)

    def apply_residual(self, k_bar, v_bar, mean_h):
        """Shared by parallel and incremental paths; K mixing precedes norm/RoPE."""
        base_scale = 1.0 if self.c3_mode == "add" else 1.0 - self.rho
        weight = self.residual_scale if self.c3_mode == "add" else self.rho
        result = []
        for branch, base in (("k", k_bar), ("v", v_bar)):
            active = branch == "v" or self.residual_scope == "kv"
            residual = weight * getattr(self, "p_" + branch)(mean_h) if active else torch.zeros_like(base)
            scaled_base = base_scale * base if active else base
            if self.record_branch_stats:
                with torch.no_grad():
                    base_norm = scaled_base.float().norm(dim=-1).mean().item()
                    residual_norm = residual.float().norm(dim=-1).mean().item()
                    self.branch_stats[branch] = {"base_norm": base_norm,
                                                 "residual_norm": residual_norm,
                                                 "residual_to_base": residual_norm / max(base_norm, 1e-12)}
            result.append(scaled_base + residual)
        return result

    # ------------------------------------------------------------------ gates

    def window_gates(self, hw: torch.Tensor, branch: str) -> torch.Tensor:
        """hw: (B, J, W, d) hidden states laid out per window. Returns alpha (B, J, W)."""
        if self.variant == "C1":
            return None  # uniform; caller uses 1/W weighting
        z = self.z_k if branch == "K" else self.z_v
        b = self.b_k if branch == "K" else self.b_v
        logits = torch.einsum("bjwd,wd->bjw", hw, z) + b
        alpha = torch.softmax(logits.float(), dim=-1).to(hw.dtype)
        if self.variant == "C2":
            alpha = (1.0 - self.rho) * alpha + self.rho / self.W
        return alpha

    def compress(self, h: torch.Tensor):
        if self.full_attention:
            raise RuntimeError("compress() is unavailable for layout_kind='full'")
        """Compute compressed entries for all full windows of a sequence.

        h: (B, L, d). Returns k_hat (B, J, dh), v_hat (B, J, dh) AFTER key-norm + RoPE
        at entry positions (j-1)S, J = L // S.
        """
        B, L, d = h.shape
        J = L // self.S
        hw = h[:, : J * self.S].reshape(B, J, self.W, d)

        alpha_k = self.window_gates(hw, "K")
        alpha_v = self.window_gates(hw, "V")
        payload_k = self.c_k(hw)  # (B, J, W, dh)
        payload_v = self.c_v(hw)

        if alpha_k is None:  # C1
            k_bar = payload_k.mean(dim=2)
            v_bar = payload_v.mean(dim=2)
        else:
            k_bar = torch.einsum("bjw,bjwd->bjd", alpha_k, payload_k)
            v_bar = torch.einsum("bjw,bjwd->bjd", alpha_v, payload_v)

        if self.variant == "C3":
            mean_h = hw.mean(dim=2)  # (B, J, d)
            k_bar, v_bar = self.apply_residual(k_bar, v_bar, mean_h)

        k_bar = self.k_norm(k_bar)  # mixing happens before normalization and RoPE
        k_bar = k_bar.to(_ac_dtype(k_bar))
        entry_pos = torch.arange(J, device=h.device) * self.S  # (j-1)S, 0-based entries
        cos, sin = build_rope_cache(entry_pos, self.head_dim, self.config.theta, k_bar.dtype)
        k_hat = apply_rope(k_bar, cos, sin)
        return k_hat, v_bar

    # ------------------------------------------------------------------ masks

    def memory_mask(self, L: int, device) -> torch.Tensor:
        if self.full_attention:
            raise RuntimeError("memory_mask() is unavailable for layout_kind='full'")
        """Boolean keep-mask (L, J + L) for the memory axis [entries ; local tokens].

        Entry e (0-based, covering tokens [eS, (e+1)S - 1]) is visible to query t iff
        (e+1)S - 1 <= t. Local token u is visible to query t iff t - T + 1 <= u <= t.
        """
        J = L // self.S
        t = torch.arange(L, device=device)[:, None]
        e = torch.arange(J, device=device)[None, :]
        u = torch.arange(L, device=device)[None, :]
        comp = (e + 1) * self.S - 1 <= t  # (L, J)
        local = (u <= t) & (u > t - self.T)  # (L, L)
        return torch.cat([comp, local], dim=1)

    def bias_mask(self, L: int, device, dtype) -> torch.Tensor:
        """Additive (1, 1, L, J+L) attention bias, cached per (L, device, dtype).

        cuDNN SDPA accepts float biases; bool masks route to the math backend,
        which materializes full attention weights and exhausts memory.
        """
        key = (L, self.S, self.T, str(device), dtype)
        m = _MASK_CACHE.get(key)
        if m is None:
            keep = self.memory_mask(L, device)
            m = torch.zeros(L, L // self.S + L, device=device, dtype=dtype)
            m.masked_fill_(~keep, float("-inf"))
            _MASK_CACHE[key] = m
        return m[None, None]

    # ------------------------------------------------------------------ forward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Parallel (training) forward. hidden_states: (B, L, d)."""
        B, L, d = hidden_states.shape
        J = L // self.S
        dev = hidden_states.device

        if self.full_attention:
            q = self.q_norm(self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim))
            q = q.to(_ac_dtype(q))
            pos = torch.arange(L, device=dev)
            cos, sin = build_rope_cache(pos, self.head_dim, self.config.theta, q.dtype)
            q = apply_rope(q, cos[None, :, None, :], sin[None, :, None, :]).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(B, L, self.head_dim))
            k = apply_rope(k.to(_ac_dtype(k)), cos[None, :, :], sin[None, :, :]).unsqueeze(1)
            v = self.v_proj(hidden_states).view(B, L, self.head_dim).unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0, scale=self.scaling, enable_gqa=True)
            out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
            return self.o_proj(out)

        q = self.q_norm(self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim))
        q = q.to(_ac_dtype(q))
        pos = torch.arange(L, device=dev)
        cos, sin = build_rope_cache(pos, self.head_dim, self.config.theta, q.dtype)
        q = apply_rope(q, cos[None, :, None, :], sin[None, :, None, :])  # (B, L, H, dh)
        q = q.transpose(1, 2)  # (B, H, L, dh)

        k_hat, v_hat = self.compress(hidden_states)  # (B, J, dh)

        k_loc = self.k_norm(self.k_proj(hidden_states).view(B, L, self.head_dim))
        k_loc = k_loc.to(_ac_dtype(k_loc))
        k_loc = apply_rope(k_loc, cos[None, :, :], sin[None, :, :])  # (B, L, dh)
        v_loc = self.v_proj(hidden_states).view(B, L, self.head_dim)

        K = torch.cat([k_hat, k_loc], dim=1).unsqueeze(1)  # (B, 1, J+L, dh)
        V = torch.cat([v_hat, v_loc], dim=1).unsqueeze(1)

        bias = self.bias_mask(L, dev, q.dtype)
        # set_priority: the allowed-set alone keeps the dispatcher's default order,
        # which skips cuDNN and lands on math (full attention weights in memory)
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                         set_priority=True):
            out = F.scaled_dot_product_attention(
                q, K, V, attn_mask=bias,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                scale=self.scaling, enable_gqa=True,
            )  # (B, H, L, dh)
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        return self.o_proj(out)

    # ------------------------------------------------------------------ incremental

    @torch.no_grad()
    def forward_token(self, h_t: torch.Tensor, t: int, state: "ResCSAIncrementalState") -> torch.Tensor:
        """One-token incremental path (tests / short-prompt inference).

        h_t: (B, 1, d) hidden state of token t (0-based). Maintains `state`.
        """
        B = h_t.shape[0]
        dev = h_t.device

        q = self.q_norm(self.q_proj(h_t).view(B, 1, self.num_heads, self.head_dim))
        cos, sin = build_rope_cache(torch.tensor([t], device=dev), self.head_dim, self.config.theta, q.dtype)
        q = apply_rope(q, cos, sin).transpose(1, 2)  # (B, H, 1, dh)

        k_loc = self.k_norm(self.k_proj(h_t).view(B, 1, self.head_dim))
        k_loc = apply_rope(k_loc, cos, sin)  # (B, 1, dh)
        v_loc = self.v_proj(h_t).view(B, 1, self.head_dim)

        if self.full_attention:
            # every token stays in the cache; no window ever closes
            state.max_local = 10**9
            state.push_local(k_loc, v_loc, t)
            K_all = torch.cat(state.local_keys, dim=1).unsqueeze(1)
            V_all = torch.cat(state.local_values, dim=1).unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q, K_all, V_all, attn_mask=None, dropout_p=0.0, scale=self.scaling, enable_gqa=True)
            out = out.transpose(1, 2).reshape(B, 1, self.num_heads * self.head_dim)
            return self.o_proj(out)

        state.push_local(k_loc, v_loc, t)
        state.push_window_hidden(h_t, t)
        if (t + 1) % self.S == 0:  # window [t-W+1, t] just closed
            hw = torch.cat(state.window_hidden, dim=1).view(B, 1, self.W, self.hidden_size)
            alpha_k = self.window_gates(hw, "K")
            alpha_v = self.window_gates(hw, "V")
            payload_k = self.c_k(hw)
            payload_v = self.c_v(hw)
            if alpha_k is None:
                k_bar = payload_k.mean(dim=2)
                v_bar = payload_v.mean(dim=2)
            else:
                k_bar = torch.einsum("bjw,bjwd->bjd", alpha_k, payload_k)
                v_bar = torch.einsum("bjw,bjwd->bjd", alpha_v, payload_v)
            if self.variant == "C3":
                mean_h = hw.mean(dim=2)
                k_bar, v_bar = self.apply_residual(k_bar, v_bar, mean_h)
            k_bar = self.k_norm(k_bar)
            ec, es = build_rope_cache(torch.tensor([t - self.W + 1], device=dev),
                                      self.head_dim, self.config.theta, k_bar.dtype)
            state.push_entry(apply_rope(k_bar, ec, es).squeeze(1), v_bar.squeeze(1))
            state.reset_window_hidden()

        K = torch.stack(state.entry_keys, dim=1) if state.entry_keys else None  # (B, J, dh)
        V = torch.stack(state.entry_values, dim=1) if state.entry_values else None
        Lk = torch.cat(state.local_keys, dim=1)  # (B, T, dh)
        Lv = torch.cat(state.local_values, dim=1)

        if K is not None:
            K_all = torch.cat([K, Lk], dim=1)
            V_all = torch.cat([V, Lv], dim=1)
        else:
            K_all, V_all = Lk, Lv
        K_all = K_all.unsqueeze(1)  # (B, 1, M, dh)
        V_all = V_all.unsqueeze(1)

        # visibility: all closed entries (they are, by construction) + local window
        out = F.scaled_dot_product_attention(
            q, K_all, V_all, attn_mask=None, dropout_p=0.0, scale=self.scaling, enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(B, 1, self.num_heads * self.head_dim)
        return self.o_proj(out)


@dataclass
class ResCSAIncrementalState:
    """Per-layer buffers for the one-token incremental path."""

    max_local: int = 16
    entry_keys: list = field(default_factory=list)
    entry_values: list = field(default_factory=list)
    local_keys: list = field(default_factory=list)
    local_values: list = field(default_factory=list)
    window_hidden: list = field(default_factory=list)

    def push_local(self, k, v, t):
        self.local_keys.append(k)
        self.local_values.append(v)
        while len(self.local_keys) > self.max_local:
            self.local_keys.pop(0)
            self.local_values.pop(0)

    def push_window_hidden(self, h, t):
        self.window_hidden.append(h)

    def reset_window_hidden(self):
        self.window_hidden = []

    def push_entry(self, k_hat, v_hat):
        self.entry_keys.append(k_hat)
        self.entry_values.append(v_hat)
