import torch

from rescsa.model import build_model


def forward_mem(tag):
    model = build_model("C0").cuda()
    if MODE == "no_mlp":
        for layer in model.model.layers:
            layer.mlp = torch.nn.Identity()
    elif MODE == "no_attn":
        for layer in model.model.layers:
            layer.self_attn = torch.nn.Identity()
    elif MODE == "no_compress":  # keep local attention only
        for layer in model.model.layers:
            att = layer.self_attn
            att.compress = lambda h, a=att: (
                torch.zeros(h.shape[0], h.shape[1] // a.S, a.head_dim,
                            device=h.device, dtype=next(a.parameters()).dtype),
                torch.zeros(h.shape[0], h.shape[1] // a.S, a.head_dim,
                            device=h.device, dtype=next(a.parameters()).dtype),
            )
    ids = torch.randint(0, 151936, (4, 2048), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(ids).last_hidden_state
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    print(f"{tag:14s} retained={(after-before)/1e9:6.2f} GB  peak={torch.cuda.max_memory_allocated()/1e9:6.2f} GB")
    del model, ids, hidden
    torch.cuda.empty_cache()


for MODE in ["full", "no_mlp", "no_attn", "no_compress"]:
    forward_mem(MODE)
