"""Memory-bounded next-token loss, also used inside the DDP forward."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LOGIT_CHUNK = 4096


def chunked_ce_loss(model, hidden, targets):
    def piece(h, t):
        return F.cross_entropy(model.lm_head(h).float(), t, reduction="sum")

    h = hidden.reshape(-1, hidden.size(-1))
    t = targets.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, t.numel(), LOGIT_CHUNK):
        total = total + checkpoint(piece, h[i:i + LOGIT_CHUNK],
                                   t[i:i + LOGIT_CHUNK], use_reentrant=False)
    return total / t.numel()
