"""Aggregate + visualize 5 TRAINED models (08 June 2026 rerun).

conf-3/conf-4 are now REAL trained fixed-2s LoRA models (conf2 LoRA config, adapter
alive) — no longer dead-adapter base-model datapoints. All 5 models treated uniformly.

  baseline  : base MDT (no adapter)        infer repeat + zero
  conf-1    : LoRA MDT conf2, train repeat  infer repeat + zero
  conf-2    : LoRA MDT conf2, train zero    infer zero
  conf-3    : LoRA fixed-2s, train repeat   infer repeat   (epoch_000)
  conf-4    : LoRA fixed-2s, train zero     infer zero     (epoch_047)

Outputs -> outputs/08_june_2026/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs/08_june_2026")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = Path("reports")

# model -> natural full-length report (padding-independent => 1 report per model)
NAT_REPORT = {
    "baseline": "baseline_mdt_protocol_full_fixed_trend_report",
    "conf-1":   "conf2_ep008_mdt_protocol_full_fixed_trend_report",
    "conf-2":   "lora_conf2_zero_trend_report",
    "conf-3":   "conf3_fixed2s_repeat_trend_report",
    "conf-4":   "conf4_fixed2s_zero_trend_report",
}
# (model, infer_padding) -> report holding its fixed-length-by-bin matrix
FIXED_REPORT = {
    ("baseline", "repeat"): "baseline_mdt_protocol_full_fixed_trend_report",
    ("baseline", "zero"):   "baseline_mdt_zeropad_trend_report",
    ("conf-1", "repeat"):   "conf2_ep008_mdt_protocol_full_fixed_trend_report",
    ("conf-1", "zero"):     "conf2_ep008_zeropad_trend_report",
    ("conf-2", "zero"):     "lora_conf2_zero_trend_report",
    ("conf-3", "repeat"):   "conf3_fixed2s_repeat_trend_report",
    ("conf-4", "zero"):     "conf4_fixed2s_zero_trend_report",
}

ORDER = ["baseline", "conf-1", "conf-2", "conf-3", "conf-4"]
COLORS = {"baseline": "#888888", "conf-1": "#1f77b4", "conf-2": "#17becf",
          "conf-3": "#d62728", "conf-4": "#2ca02c"}
LEGEND = {"baseline": "baseline (MDT base)",
          "conf-1": "conf-1 (LoRA MDT, train repeat)",
          "conf-2": "conf-2 (LoRA MDT, train zero)",
          "conf-3": "conf-3 (LoRA fixed-2s, train repeat)",
          "conf-4": "conf-4 (LoRA fixed-2s, train zero)"}
BIN_LABEL = {"observed_short_0p5_1p0": "0.5-1.0s", "observed_short_1p0_1p5": "1.0-1.5s",
             "observed_short_1p5_2p0": "1.5-2.0s", "vad_0p5_1p0": "0.5-1.0s",
             "vad_1p0_1p5": "1.0-1.5s", "vad_1p5_2p0": "1.5-2.0s"}
bins = ["0.5-1.0s", "1.0-1.5s", "1.5-2.0s"]
durs = ["0.5s", "1.0s", "1.5s", "2.0s"]
xd = [0.5, 1.0, 1.5, 2.0]
M3 = {"eer_percent": "eer", "mdr_at_far1_percent": "mdr_at_far1",
      "threshold_free_accuracy_percent": "acc"}

# ---------------- collect NATURAL full-length (overall + by bin) ----------------
nat_rows = []
for model, rep in NAT_REPORT.items():
    df = pd.read_csv(R / rep / "threshold_free_class_accuracy_summary.csv")
    fl = df[df.eval_type.isin(["full_length_overall", "full_length_by_bin"])].copy()
    fl = fl.drop(columns=[c for c in ["eer", "mdr_at_far1", "threshold_free_accuracy"]
                          if c in fl.columns])
    fl["model"] = model
    fl["bin"] = fl.duration_bin_eval.map(lambda v: BIN_LABEL.get(v, "overall"))
    nat_rows.append(fl.rename(columns=M3))
nat = pd.concat(nat_rows, ignore_index=True)

# ---------------- collect FIXED length by bin (per model+infer padding) ----------------
fix_rows = []
for (model, pad), rep in FIXED_REPORT.items():
    df = pd.read_csv(R / rep / "fixed_length_by_bin_summary.csv")
    df = df.drop(columns=[c for c in ["eer", "mdr_at_far1", "threshold_free_accuracy"]
                          if c in df.columns]).rename(columns=M3)
    df["model"], df["infer_padding"] = model, pad
    df["bin"] = df.duration_bin_eval.map(BIN_LABEL)
    fix_rows.append(df[["model", "infer_padding", "dataset_type", "bin", "random_start",
                        "inference_duration", "samples", "eer", "mdr_at_far1", "acc"]])
fix = pd.concat(fix_rows, ignore_index=True).dropna(subset=["bin"])

# ---------------- comparison CSV ----------------
nat_csv = nat[nat.eval_type == "full_length_by_bin"][
    ["model", "dataset_type", "bin", "eer", "mdr_at_far1", "acc"]].copy()
nat_csv.insert(1, "infer_padding", "natural")
nat_csv.insert(4, "random_start", np.nan)
nat_csv.insert(5, "inference_duration", "full")
out = pd.concat([nat_csv.assign(eval_type="full_length_by_bin"),
                 fix.assign(eval_type="fixed_length_by_bin")], ignore_index=True)
for c in ["eer", "mdr_at_far1", "acc"]:
    out[c] = out[c].round(2)
out.to_csv(OUT / "full_vs_fixed_by_bin.csv", index=False)
print("full_vs_fixed_by_bin.csv:", out.shape)

# ================= FIG 1: full-length natural EER by bin =================
def grouped(ax, sub, metric, hb=False):
    x = np.arange(len(bins)); w = 0.16
    for i, m in enumerate(ORDER):
        vals = [sub[(sub.model == m) & (sub.bin == b)][metric].values for b in bins]
        vals = [v[0] if len(v) else np.nan for v in vals]
        bb = ax.bar(x + (i - 2) * w, vals, w, label=LEGEND[m], color=COLORS[m])
        ax.bar_label(bb, fmt="%.2f", fontsize=6.5, rotation=90, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(bins); ax.grid(axis="y", alpha=0.3)
    vmax = sub[metric].max()
    if hb:
        vmin = sub[metric].min()
        ax.set_ylim(max(0, vmin - (vmax - vmin) * 0.5), vmax + (vmax - vmin) * 0.35)
    else:
        ax.set_ylim(0, vmax * 1.35)

fb = nat[nat.eval_type == "full_length_by_bin"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ds in zip(axes, ["observed_short", "vad_short"]):
    grouped(ax, fb[fb.dataset_type == ds], "eer")
    ax.set_title(f"Natural full-length EER by duration bin — {ds}")
    ax.set_ylabel("EER (%)")
axes[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "full_length_eer_by_bin.png", dpi=150); plt.close(fig)

# ================= FIG 2: full-length natural EER/MDR/ACC by bin =================
metrics = [("eer", "EER (%)", False), ("mdr_at_far1", "MDR@FAR=1% (%)", False),
           ("acc", "ACC (%)", True)]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for r, ds in enumerate(["observed_short", "vad_short"]):
    sub = fb[fb.dataset_type == ds]
    for c, (m, title, hb) in enumerate(metrics):
        grouped(axes[r][c], sub, m, hb=hb)
        axes[r][c].set_title(f"{ds} — {title}", fontsize=10)
axes[0][0].legend(fontsize=7, loc="upper right")
fig.suptitle("Natural full-length by duration bin — EER / MDR@FAR=1% / ACC", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "full_length_metrics_by_bin.png", dpi=150); plt.close(fig)

# ================= FIG 3: full-length natural overall =================
fo = nat[nat.eval_type == "full_length_overall"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for r, ds in enumerate(["observed_short", "vad_short"]):
    sub = fo[fo.dataset_type == ds]
    for c, (m, title, hb) in enumerate(metrics):
        vals = [sub[sub.model == cfg][m].values[0] for cfg in ORDER]
        bb = axes[r][c].bar(range(len(ORDER)), vals, color=[COLORS[x] for x in ORDER])
        axes[r][c].bar_label(bb, fmt="%.2f", fontsize=8)
        axes[r][c].set_title(f"{ds} — {title}", fontsize=10)
        axes[r][c].set_xticks(range(len(ORDER))); axes[r][c].set_xticklabels(ORDER, fontsize=8)
        axes[r][c].grid(axis="y", alpha=0.3)
        if hb:
            axes[r][c].set_ylim(min(vals) - 1, max(vals) + 0.5)
fig.suptitle("Natural full-length overall — EER / MDR@FAR=1% / ACC", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "full_length_overall_metrics.png", dpi=150); plt.close(fig)

# ================= FIG 4: fixed-length EER trend vs duration (7 lines) =================
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for r, ds in enumerate(["observed_short", "vad_short"]):
    for c, rs in enumerate([False, True]):
        ax = axes[r][c]
        sub = fix[(fix.dataset_type == ds) & (fix.random_start == rs)]
        # pool 3 bins -> mean per duration (overall-ish trend), per (model,padding)
        for (m, p), g in sub.groupby(["model", "infer_padding"]):
            s = g.groupby("inference_duration").eer.mean().reindex(durs)
            ax.plot(xd, s.values, marker="o", color=COLORS[m],
                    ls="--" if p == "zero" else "-", lw=1.6, ms=4,
                    label=f"{m}|{p}")
        ax.set_title(f"{ds} — random_start={rs}", fontsize=10)
        ax.grid(alpha=0.3)
        if c == 0: ax.set_ylabel("mean EER over bins (%)")
        if r == 1: ax.set_xlabel("fixed inference duration (s)")
axes[0][0].legend(fontsize=7, ncol=2)
fig.suptitle("Fixed-length EER vs duration (solid=repeat infer, dashed=zero infer)", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "fixed_length_eer_trend.png", dpi=150); plt.close(fig)

# ================= FIG 5: natural vs best-fixed per bin (uniform 5 groups) =================
fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
for r, ds in enumerate(["observed_short", "vad_short"]):
    for c, b in enumerate(bins):
        ax = axes[r][c]
        nat_v, fix_v, fix_lab = [], [], []
        for m in ORDER:
            nv = fb[(fb.dataset_type == ds) & (fb.bin == b) & (fb.model == m)].eer.values
            nat_v.append(nv[0] if len(nv) else np.nan)
            fsub = fix[(fix.dataset_type == ds) & (fix.bin == b) & (fix.model == m)]
            i = fsub.eer.idxmin()
            fix_v.append(fsub.loc[i, "eer"])
            fix_lab.append(f"{fsub.loc[i,'inference_duration']}|{fsub.loc[i,'infer_padding']}"
                           f"|rs={'T' if fsub.loc[i,'random_start'] else 'F'}")
        x = np.arange(len(ORDER))
        b1 = ax.bar(x - 0.2, nat_v, 0.38, label="natural full-length", color="#bbbbbb")
        b2 = ax.bar(x + 0.2, fix_v, 0.38, label="best fixed (crop orig)", color="#2ca02c")
        ax.bar_label(b1, fmt="%.2f", fontsize=7)
        ax.bar_label(b2, fmt="%.2f", fontsize=8)
        for xi, lab in zip(x, fix_lab):
            ax.text(xi + 0.2, 0.05, lab, fontsize=6, ha="center", va="bottom", rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=9)
        ax.set_title(f"{ds} — true duration {b}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, np.nanmax(nat_v + fix_v) * 1.3)
axes[0][0].legend(fontsize=8)
fig.suptitle("Natural full-length vs best fixed-length (crop from original utterance) — "
             "EER by true-duration bin\n(all 5 trained models, fixed = min EER over "
             "duration x padding x random_start)", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "full_vs_fixed_best_by_bin.png", dpi=150); plt.close(fig)

print("figures:", sorted(p.name for p in FIG.glob("*.png")))
