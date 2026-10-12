"""Phase-sensitive KV model: Qwen3-0.6B backbone with compressed-attention layers.

We reuse Qwen3's embeddings, RMSNorm, SwiGLU MLP, residual structure and tied
lm_head unchanged, so chunked KV compression is the only architectural change
(matching the paper's controlled setup). The custom attention lives in attention.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)

from .attention import ResCSAAttention, ResCSAIncrementalState
from .config import ResCSAConfig


class ResCSAPreTrainedModel(Qwen3PreTrainedModel):
    config_class = ResCSAConfig
    base_model_prefix = "model"
    _no_split_modules = ["ResCSADecoderLayer"]


class ResCSADecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: ResCSAConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = ResCSAAttention(config=config, layer_idx=layer_idx)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # compression handles its own masking/positions; standard masks are not applicable
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class ResCSAModel(Qwen3Model):
    config_class = ResCSAConfig

    def __init__(self, config: ResCSAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ResCSADecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids / inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    @torch.no_grad()
    def forward_incremental(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """One-token-at-a-time path (consistency tests, short-prompt inference)."""
        states = [
            ResCSAIncrementalState(max_local=self.config.local_window)
            for _ in range(self.config.num_hidden_layers)
        ]
        embeds = self.embed_tokens(input_ids)  # (B, L, d)
        L = embeds.shape[1]
        outputs = []
        for t in range(L):
            h = embeds[:, t : t + 1]
            for layer, state in zip(self.layers, states):
                residual = h
                x = layer.input_layernorm(h)
                x = layer.self_attn.forward_token(x, t, state)
                h = residual + x
                residual = h
                h = residual + layer.mlp(layer.post_attention_layernorm(h))
            outputs.append(h)
        return self.norm(torch.cat(outputs, dim=1))


class ResCSAForCausalLM(Qwen3ForCausalLM):
    config_class = ResCSAConfig
    base_model_prefix = "model"

    def __init__(self, config: ResCSAConfig):
        super().__init__(config)
        self.model = ResCSAModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)
        logits = self.lm_head(out.last_hidden_state)
        return CausalLMOutputWithPast(logits=logits)

    @torch.no_grad()
    def forward_incremental(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.lm_head(self.model.forward_incremental(input_ids))


def build_model(variant: str = "learned", seed: int = 42) -> ResCSAForCausalLM:
    """Construct the reference model.

    The same seed produces identical initialization for both variants (gate maps are
    zero-initialized, so "learned" computes the exact uniform average at step 0).
    """
    torch.manual_seed(seed)
    cfg = ResCSAConfig(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=1,  # paper reference: one KV head per layer
        head_dim=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=40960,
        rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        tie_word_embeddings=True,
        attention_bias=False,
        window_size=8,
        stride=8,
        local_window=16,
        variant=variant,
    )
    return ResCSAForCausalLM(cfg)
