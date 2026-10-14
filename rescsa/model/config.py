"""ResCSA configuration: Qwen3 backbone + chunked-KV-compression fields.

Follows arXiv 2609.36322 Appendix D (reference kernel):
- non-overlapping windows W = S = 8
- one KV head per layer (num_key_value_heads=1)
- separate K/V compression branches, scalar gates (dg = 1)
- payload maps shared across offsets, gate maps + biases offset-specific
- exact 16-token raw local window, one joint softmax over compressed+local memory

Variants (round 1):
- C0: learned compression (paper reference)        K_hat = sum_i alpha_i C h_i
- C1: uniform compression, alpha_i = 1/W            (no gate parameters)
- C2: gate smoothing  alpha~ = (1-rho) alpha + rho/W
- C3: independent mean projection, with explicit mode/scope/initialization:
      mix: (1-rho) A + rho P(mean(h)); add: A + residual_scale P(mean(h)).
      Scope is K+V or V-only; P initializes by copying C or with zeros.
"""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class ResCSAConfig(Qwen3Config):
    model_type = "rescsa"

    def __init__(
        self,
        window_size: int = 8,
        stride: int = 8,
        local_window: int = 16,
        variant: str = "C0",
        rho: float = 0.1,
        c3_mode: str = "mix",
        residual_scale: float = 0.1,
        residual_init: str = "copy",
        residual_scope: str = "kv",
        layout_kind: str = "fixed",
        **kwargs,
    ):
        # Reference kernel is defined for W == S only (non-overlapping windows).
        if window_size != stride:
            raise ValueError("ResCSA round 1 supports non-overlapping windows only (W == S)")
        if variant not in {"C0", "C1", "C2", "C3"}:
            raise ValueError(f"unknown variant {variant!r}")
        if not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0,1]")
        if c3_mode not in {"mix", "add"} or residual_init not in {"copy", "zero"}:
            raise ValueError("invalid C3 mode or residual initialization")
        if residual_scope not in {"kv", "v"} or residual_scale < 0:
            raise ValueError("invalid residual scope or scale")
        if layout_kind not in {"fixed", "full"}:
            raise ValueError(f"unknown layout_kind {layout_kind!r}")
        if layout_kind == "full" and variant != "C0":
            # the uncompressed baseline is a pure control; variant axes do not apply
            raise ValueError("layout_kind='full' requires variant='C0'")
        super().__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride
        self.local_window = local_window
        self.variant = variant
        self.rho = rho
        self.c3_mode = c3_mode
        self.residual_scale = residual_scale
        self.residual_init = residual_init
        self.residual_scope = residual_scope
        self.layout_kind = layout_kind

    @property
    def theta(self) -> float:
        """RoPE base frequency, robust across transformers 4.x/5.x config layouts."""
        rp = getattr(self, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            return float(rp["rope_theta"])
        return float(getattr(self, "rope_theta", 10000.0))
