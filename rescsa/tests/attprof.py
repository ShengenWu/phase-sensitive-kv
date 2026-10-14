import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from rescsa.model import build_model
from rescsa.model.attention import apply_rope, build_rope_cache

model = build_model("C0").cuda()
att = model.model.layers[5].self_attn
B, L, d = 4, 2048, 1024
h = torch.randn(B, L, d, device="cuda")


def mem(tag):
    torch.cuda.synchronize()
    print(f"{tag:34s} {torch.cuda.memory_allocated()/1e9:7.3f} GB")


keep = []
with torch.autocast("cuda", dtype=torch.bfloat16):
    mem("start")
    q = att.q_norm(att.q_proj(h).view(B, L, 16, 128)); keep.append(q); mem("q_proj+norm")
    pos = torch.arange(L, device="cuda")
    cos, sin = build_rope_cache(pos, 128, att.config.theta, q.dtype); mem("rope cache")
    q = apply_rope(q, cos[None, :, None, :], sin[None, :, None, :]); keep.append(q); mem("q rope")
    q = q.transpose(1, 2); mem("q transpose")
    k_hat, v_hat = att.compress(h); keep += [k_hat, v_hat]; mem("compress")
    k_loc = att.k_norm(att.k_proj(h).view(B, L, 128)); keep.append(k_loc); mem("k_loc norm")
    k_loc = apply_rope(k_loc, cos[None, :, :], sin[None, :, :]); keep.append(k_loc); mem("k_loc rope")
    v_loc = att.v_proj(h).view(B, L, 128); keep.append(v_loc); mem("v_loc")
    K = torch.cat([k_hat, k_loc], dim=1).unsqueeze(1); keep.append(K); mem("K cat")
    V = torch.cat([v_hat, v_loc], dim=1).unsqueeze(1); keep.append(V); mem("V cat")
    bias = att.bias_mask(L, q.device, q.dtype); mem("bias mask")
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, K, V, attn_mask=bias, dropout_p=0.0,
                                             scale=att.scaling, enable_gqa=True)
    keep.append(out); mem("sdpa out")
    out2 = out.transpose(1, 2).reshape(B, L, 2048); keep.append(out2); mem("out reshape")
    o = att.o_proj(out2); keep.append(o); mem("o_proj")
