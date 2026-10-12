"""Schematics for the chunked-compression kernel: windowing/phase layout + dataflow.

No data dependency. Run:  python scripts/plot_kernel_diagram.py
Outputs figures/kernel_windowing.png and figures/kernel_dataflow.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

PHASE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#937860", "#DA8BC3", "#8C8C8C"]  # phase 7 = boundary, grey


def kernel_windowing():
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    n, W, T = 26, 8, 16
    for i in range(n):
        ax.add_patch(FancyBboxPatch((i, 1.0), 0.82, 0.5, boxstyle="round,pad=0.02",
                                    fc=PHASE_COLORS[i % 8], ec="black", lw=0.6))
        ax.text(i + 0.41, 1.25, str(i), ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    # window brackets + compressed entries (emitted when a window closes)
    for j in range(n // W):  # full windows only
        x0, x1 = j * W, (j + 1) * W - 1 + 0.82
        ax.plot([x0, x1 + 0.09], [0.82, 0.82], color="black", lw=1.4)
        ax.plot([x0, x0], [0.82, 0.88], color="black", lw=1.4)
        ax.plot([x1 + 0.09, x1 + 0.09], [0.82, 0.88], color="black", lw=1.4)
        cx = (x0 + x1) / 2
        ax.text(cx, 0.64, f"window $j={j+1}$", ha="center", fontsize=9)
        ax.add_patch(FancyBboxPatch((cx - 1.85, 0.02), 3.7, 0.42, boxstyle="round,pad=0.03",
                                    fc="#FFF3CD", ec="#B8860B"))
        ax.text(cx, 0.23, f"$\\hat{{K}}_{{{j+1}}},\\hat{{V}}_{{{j+1}}}$",
                ha="center", va="center", fontsize=9.5)
        ax.text(cx, -0.22, f"closes at {(j+1)*W-1}  \u00b7  RoPE pos {j*W}",
                ha="center", fontsize=8, color="#7A5C00")
    # partial trailing window
    if n % W:
        x0 = (n // W) * W
        ax.plot([x0, n - 1 + 0.91], [0.82, 0.82], color="black", lw=1.4, ls=":")
        ax.text((x0 + n) / 2, 0.64, "partial window\n(no entry yet)", ha="center",
                fontsize=8, color="#555")

    # query t=22 and its visibility
    t = 22
    for u in range(t - T + 1, t + 1):
        ax.add_patch(FancyBboxPatch((u, 0.94), 0.82, 0.62, boxstyle="round,pad=0.02",
                                    fc="none", ec="#2E7D32", lw=1.6, ls="--"))
    ax.add_patch(FancyBboxPatch((t - 0.06, 0.88), 0.94, 0.74, boxstyle="round,pad=0.02",
                                fc="none", ec="#B71C1C", lw=2.2))
    ax.text(t + 0.5, 2.08, "query $q_t$, $t{=}22$", color="#B71C1C", fontsize=9.5,
            ha="center", fontweight="bold")
    ax.annotate("local window $u \\in [t{-}15, t]$: raw $k_u, v_u$\n(dashed green)",
                xy=(3.5, 1.62), xytext=(0.2, 2.9), fontsize=9, ha="left", color="#2E7D32",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#2E7D32"))
    ax.annotate("entries $j$ with $jS{-}1 \\leq t$ are visible\n(entry 3 closes at 23: not yet)",
                xy=(12.2, 0.44), xytext=(14.5, -0.75), fontsize=8.5, ha="left", color="#333",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#333"))

    # phase legend (top right, clear of the token strip)
    for p in range(8):
        ax.add_patch(FancyBboxPatch((17.6 + p * 1.05, 2.62), 0.8, 0.42,
                                    boxstyle="round,pad=0.02",
                                    fc=PHASE_COLORS[p], ec="black", lw=0.5))
        ax.text(18.0 + p * 1.05, 3.14, str(p), ha="center", fontsize=8)
    ax.text(17.4, 2.83, "phase = $pos\\ \\mathrm{mod}\\ S$,  $S{=}8$:", fontsize=9,
            va="center", ha="right")
    ax.set_xlim(-0.8, n + 1.4)
    ax.set_ylim(-1.15, 3.55)
    ax.axis("off")
    ax.set_title("Windows, phases and visibility (W=S=8, local window T=16)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kernel_windowing.png", dpi=200)
    print("wrote figures/kernel_windowing.png")


def kernel_dataflow():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    def box(x, y, w, h, text, fc="#EEF3FB", ec="#4C72B0", fs=9.5, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, color="#333", style="-|>", lw=1.3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=13, color=color, lw=lw))

    # hidden states of one window
    box(0.3, 3.0, 2.4, 2.6, "hidden states\n$h_{j,0} \\ldots h_{j,7}$\n(one window,\n8 offsets)",
        fc="#F5F5F5", ec="#666")
    # gate path
    box(4.0, 4.35, 3.4, 1.15, "gate logits  $s_i = \\langle Z_i, h_{j,i}\\rangle + b_i$\n"
        "$Z_i \\in \\mathbb{R}^{d}$, offset-specific (scalar gate)", fc="#FDECEA", ec="#C44E52")
    box(4.0, 2.95, 3.4, 1.0, "$\\alpha = \\mathrm{softmax}_i(s)$   over the\n"
        "8 offsets   (uniform: $\\alpha_i \\equiv 1/W$)", fc="#FDECEA", ec="#C44E52")
    # payload path
    box(4.0, 1.15, 3.4, 1.4, "payload  $C h_{j,i}$\n$C \\in \\mathbb{R}^{d_h \\times d}$ shared\nacross offsets (per branch)",
        fc="#EAF3EA", ec="#55A868")
    arrow(2.72, 5.05, 3.92, 4.93, "#C44E52")   # -> gate logits
    arrow(2.72, 4.55, 3.92, 3.45, "#C44E52")   # -> softmax box
    arrow(2.72, 3.55, 3.92, 1.85, "#55A868")   # -> payload
    # weighted sums
    box(8.2, 3.3, 3.1, 1.9, "$\\bar{K}_j = \\sum_i \\alpha_i\\, C^K h_{j,i}$\n\n"
        "$\\bar{V}_j = \\sum_i \\alpha_i\\, C^V h_{j,i}$\n(separate K/V branches)", fc="#FFF8E1", ec="#B8860B")
    arrow(7.42, 3.45, 8.12, 4.25, "#C44E52")   # alpha -> weighted sums
    arrow(7.42, 1.85, 8.12, 3.55, "#55A868")   # payload -> weighted sums
    # norm/rope
    box(8.2, 1.55, 3.1, 1.2, "$\\hat{k}_j = \\mathrm{RoPE}_{(j-1)S}\\left(\\mathrm{RMSNorm}_K(\\bar{K}_j)\\right)$\n"
        "$\\hat{v}_j = \\bar{V}_j$\n(no norm / no rotation)", fc="#EEF3FB", fs=9)
    arrow(9.75, 3.28, 9.75, 2.79, "#B8860B")
    # memory axis + joint attention
    box(3.6, 0.0, 5.0, 1.0, "memory axis:  $[$ compressed entries $\\hat{k}_j,\\hat{v}_j$  ;  "
        "raw local $k_u, v_v \\ (|u - t| < 16)\\,]$", fc="#F3E5F5", ec="#8172B3", fs=9)
    arrow(9.0, 1.55, 8.0, 1.0, "#8172B3")
    box(9.3, 0.0, 3.3, 1.0, "one joint softmax\n(SDPA, GQA 16:1)", fc="#F3E5F5", ec="#8172B3")
    arrow(8.6, 0.5, 9.3, 0.5, "#8172B3")
    box(0.3, 0.0, 2.6, 1.0, "query $q_t$ (per head,\nRMSNorm + RoPE)", fc="#F5F5F5", ec="#666", fs=9)
    arrow(2.9, 0.5, 3.6, 0.5, "#8172B3")
    arrow(11.9, 1.0, 11.9, 1.75, "#8172B3")
    ax.text(11.9, 1.95, "$o_t$", fontsize=11, ha="center")
    ax.set_xlim(0, 13.4)
    ax.set_ylim(-0.3, 5.9)
    ax.axis("off")
    ax.set_title("Compression kernel: one entry per window (paper reference, Appendix D)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kernel_dataflow.png", dpi=200)
    print("wrote figures/kernel_dataflow.png")


if __name__ == "__main__":
    kernel_windowing()
    kernel_dataflow()
