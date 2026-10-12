"""Phase-sensitive KV compression configuration: Qwen3 backbone + chunked-KV compression.

Implements the reference kernel of arXiv 2609.36322 (Appendix D):
- non-overlapping windows W = S = 8
- one KV head per layer (num_key_value_heads=1)
- separate K/V compression branches, scalar gates (dg = 1)
- payload maps shared across offsets, gate maps + biases offset-specific
- exact 16-token raw local window, one joint softmax over compressed+local memory

Variants (both from the paper's own model family):
- "learned": the reference kernel (offset-specific learned gates)  [paper Table 8 "Reference"]
- "uniform": uniform averaging, gate scores fixed at 1/W           [paper Table 8 "Uniform averaging"]
"""

import os

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

DATA_DIR = os.environ.get("PSKV_DATA_DIR", "/root/rescsa/data")


class ResCSAConfig(Qwen3Config):
    model_type = "pskv"

    def __init__(
        self,
        window_size: int = 8,
        stride: int = 8,
        local_window: int = 16,
        variant: str = "learned",
        **kwargs,
    ):
        # The reference kernel is defined for W == S only (non-overlapping windows).
        if window_size != stride:
            raise ValueError("this implementation supports non-overlapping windows only (W == S)")
        if variant not in {"learned", "uniform"}:
            raise ValueError(f"unknown variant {variant!r}")
        super().__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride
        self.local_window = local_window
        self.variant = variant

    @property
    def theta(self) -> float:
        """RoPE base frequency, robust across transformers 4.x/5.x config layouts."""
        rp = getattr(self, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            return float(rp["rope_theta"])
        return float(getattr(self, "rope_theta", 10000.0))
