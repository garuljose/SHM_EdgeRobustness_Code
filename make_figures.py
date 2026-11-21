import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

MODEL_ORDER = ["Full-signal baseline", "Naive-impute baseline", "Recon-GRU baseline",
               "Proposed - no pretrain (ablation)", "Proposed (ours)"]
COLORS = {"Full-signal baseline": "#888888", "Naive-impute baseline": "#4C72B0",
          "Recon-GRU baseline": "#DD8452", "Proposed - no pretrain (ablation)": "#C44E52",
          "Proposed (ours)": "#55A868"}
MARKERS = {"Full-signal baseline": "o", "Naive-impute baseline": "s",
           "Recon-GRU baseline": "^", "Proposed - no pretrain (ablation)": "x",
           "Proposed (ours)": "D"}

df = pd.read_csv(f"{OUT}/results_grid.csv")

# ---------------------------------------------------------------- Fig 1: accuracy vs missingness
# single-column width (stacked subplots), sized for an IEEE two-column layout
fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.6), sharex=True)
for ax, ctype in zip(axes, ["MCAR", "MAR", "MNAR"]):
    sub = df[df.ctype == ctype]
    for model in MODEL_ORDER:
        m = sub[sub.model == model].sort_values("rate")
        ax.plot(m.rate * 100, m.accuracy * 100, marker=MARKERS[model], label=model,
                color=COLORS[model], linewidth=1.3, markersize=3.5)
    ax.set_ylabel(f"{ctype}\naccuracy (%)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
axes[-1].set_xlabel("Missing / corrupted rate (%)")
axes[0].legend(fontsize=6.3, loc="lower left", ncol=2, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_accuracy_vs_missingness.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: accuracy vs resource footprint
prof = pd.read_csv(f"{OUT}/resource_profile.csv")
avg_acc = df[df.ctype != "clean"].groupby("model")["accuracy"].mean().reset_index()
name_map = {
    "Full-signal baseline (CNN)": "Full-signal baseline",
    "Recon-GRU baseline": "Recon-GRU baseline",
    "Proposed (ours)": "Proposed (ours)",
}
prof["model_key"] = prof["model"].map(name_map)
merged = prof.merge(avg_acc, left_on="model_key", right_on="model", suffixes=("", "_avg"))

fig, ax = plt.subplots(figsize=(3.4, 2.8))
for _, row in merged.iterrows():
    ax.scatter(row["size_kb_int8"], row["accuracy"] * 100, s=70,
               color=COLORS.get(row["model_key"], "#333"), zorder=3)
    ax.annotate(row["model_key"].replace(" baseline", "").replace(" (ours)", ""),
                (row["size_kb_int8"], row["accuracy"] * 100),
                textcoords="offset points", xytext=(6, 4), fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("Quantized (int8) model size, KB (log scale)")
ax.set_ylabel("Mean accuracy across all\ncorruption conditions (%)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_accuracy_vs_resource.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: ablation (pretrain vs no pretrain)
sub = df[df.model.isin(["Proposed (ours)", "Proposed - no pretrain (ablation)"])]
grp = sub.groupby(["ctype", "model"])["accuracy"].mean().reset_index()
grp = grp[grp.ctype != "clean"]
ctypes = ["MCAR", "MAR", "MNAR"]
x = np.arange(len(ctypes))
width = 0.35
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for i, model in enumerate(["Proposed - no pretrain (ablation)", "Proposed (ours)"]):
    vals = [grp[(grp.ctype == c) & (grp.model == model)]["accuracy"].values[0] * 100 for c in ctypes]
    ax.bar(x + (i - 0.5) * width, vals, width, label=model.replace(" (ablation)", ""),
           color=COLORS[model])
ax.set_xticks(x)
ax.set_xticklabels(ctypes)
ax.set_ylabel("Mean accuracy (%)")
ax.set_ylim(60, 85)
ax.legend(fontsize=6.5, loc="lower right")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_ablation_pretraining.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: natural-frequency shift (physics validation)
from simulate_structure import assemble_matrices, K_NOMINAL
fig, ax = plt.subplots(figsize=(3.4, 2.6))
sevs = [0.0, 0.20, 0.35, 0.50]
for sev in sevs:
    k = K_NOMINAL.copy()
    k[0] *= (1 - sev)
    M, C, K = assemble_matrices(k)
    Minv = np.linalg.inv(M)
    eig = np.linalg.eigvals(Minv @ K)
    freqs = np.sort(np.sqrt(eig.real)) / (2 * np.pi)
    ax.plot(range(1, 5), freqs, marker="o", label=f"{int(sev*100)}% stiffness loss" if sev > 0 else "Healthy")
ax.set_xticks(range(1, 5))
ax.set_xlabel("Mode number")
ax.set_ylabel("Natural frequency (Hz)")
ax.legend(fontsize=6.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_modal_frequency_shift.png", bbox_inches="tight")
plt.close(fig)

print("Saved fig1-4 PNGs to outputs/")

# ---------------------------------------------------------------- summary tables (CSV, for embedding into paper tables)
main_table = df[df.ctype != "clean"].groupby("model")[["accuracy", "f1", "auc"]].mean().reset_index()
clean_table = df[df.ctype == "clean"][["model", "accuracy", "f1", "auc"]]
main_table.to_csv(f"{OUT}/table_main_results.csv", index=False)
clean_table.to_csv(f"{OUT}/table_clean_results.csv", index=False)
print(main_table.round(3))
print(clean_table.round(3))
