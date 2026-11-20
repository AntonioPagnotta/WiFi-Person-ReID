import json
import re
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    ("Original paper model", "../WiPER-Benchmarking_original.ipynb"),
    ("v_2 - Open set before bug fix", "../WiPER-Benchmarking_v2.ipynb"),
    ("v_2.5 - Open set + bug fix", "../WiPER-Benchmarking_v2_5.ipynb"),
    ("v_3 - Open set + HardTripletLoss", "../WiPER-Benchmarking_v3.ipynb"),
]

N_EP = 20


def parse(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    txt = "".join(
        "".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))
    pat = re.compile(
        r"Total samples: (\d+), Rank #1: ([\d.]+)%.*?Rank #5: ([\d.]+)%.*?Rank #10: ([\d.]+)%, mAP:([\d.]+)%\s*\nEpoch: \[(\d+)/\d+\]\| Loss: ([\d.]+)",
        re.S)

    rows = [dict(epoch=int(m[6]), loss=float(m[7]), rank1=float(m[2]), rank5=float(m[3]), rank10=float(m[4]),
                 mAP=float(m[5])) for m in pat.finditer(txt)]
    df = pd.DataFrame(rows).drop_duplicates("epoch").sort_values("epoch")
    return df[df.epoch <= N_EP].reset_index(drop=True)


series = [
    ("rank1", "Rank-1", "#1f77b4", "o"),
    ("rank5", "Rank-5", "#2ca02c", "s"),
    ("rank10", "Rank-10", "#9467bd", "^"),
    ("mAP", "mAP", "#ff7f0e", "D")
]

fig, axes = plt.subplots(2, 2, figsize=(18, 10))

h1, l1, h2, l2 = [], [], [], []

for ax, (name, path) in zip(axes.ravel(), MODELS):
    try:
        d = parse(path)

        for col, lab, c, mk in series:
            ax.plot(d.epoch, d[col], marker=mk, ms=4, lw=1.8, color=c, label=lab)

        met_min = d[['rank1', 'rank5', 'rank10', 'mAP']].min().min()
        met_max = d[['rank1', 'rank5', 'rank10', 'mAP']].max().max()
        margin = (met_max - met_min) * 0.1
        ax.set_ylim(max(0, met_min - margin), min(105, met_max + margin))

        ax.set_ylabel("Metrics (%)")
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(2, N_EP + 1, 2))
        ax.set_xlim(0.5, N_EP + 0.5)
        ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=14, fontweight="bold")

        ax2 = ax.twinx()
        ax2.plot(d.epoch, d.loss, "--", color="#d62728", lw=2, marker="x", ms=4, label="Loss")
        ax2.set_ylabel("Loss", color="#d62728")
        ax2.tick_params(axis="y", colors="#d62728")

        loss_min = d.loss.min()
        loss_max = d.loss.max()
        l_margin = (loss_max - loss_min) * 0.1
        ax2.set_ylim(max(0, loss_min - l_margin), loss_max + l_margin)

        if not h1:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

    except FileNotFoundError:
        ax.set_title(f"{name}\n(File non trovato!)", color="red")
        ax.axis('off')

if h1 and h2:
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.05))

fig.suptitle("WiPER – Metrics and Loss per Model (20 epochs)", fontsize=18, fontweight="bold")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])

output_name = "wiper_per_model_20_epochs_dynamic.png"
fig.savefig(output_name, dpi=160, bbox_inches="tight", facecolor='white')
print(f"Immagine salvata con successo come: {output_name}")

plt.close(fig)