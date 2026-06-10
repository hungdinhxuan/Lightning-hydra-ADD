#!/usr/bin/env python3
"""Aggregate speechcommand (all-bonafide) score files into report + figures.

All clips are bonafide -> no EER. Metric = bonafide detection rate (higher =
better; a clip below threshold is a FALSE spoof alarm). Reports the rate at
three operating points and the raw-logit score distribution.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCORE_DIR = "logs/results/speechcommand_bonafide"
OUT_DIR = "outputs/10_june_2026"
FIG_DIR = os.path.join(OUT_DIR, "figures")

# old EER thresholds on raw bonafide-logit scale (from full_length_overall_summary)
THR = {
    "baseline": dict(obs=-1.1484, vad=-0.4980),
    "conf-1":   dict(obs=-0.7812, vad=-0.2559),
    "conf-2":   dict(obs=-0.5977, vad=0.0454),
    "conf-3":   dict(obs=-1.1250, vad=-0.5664),
    "conf-4":   dict(obs=-0.1021, vad=1.1016),
}
ORDER = ["baseline", "conf-1", "conf-2", "conf-3", "conf-4"]
COLORS = {"baseline": "#888888", "conf-1": "#1f77b4", "conf-2": "#17becf",
          "conf-3": "#d62728", "conf-4": "#2ca02c"}


def load(name):
    rows = []
    with open(os.path.join(SCORE_DIR, f"{name}.txt")) as f:
        for ln in f:
            p, s0, s1 = ln.rsplit(maxsplit=2)
            rows.append((p, float(s0), float(s1)))
    df = pd.DataFrame(rows, columns=["path", "spoof", "bona"])
    return df


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summ = []
    scores = {}
    for name in ORDER:
        df = load(name)
        scores[name] = df
        bona, spoof = df["bona"].values, df["spoof"].values
        n = len(df)
        # @0.5 softmax == argmax bonafide == bona > spoof
        rate_05 = float((bona > spoof).mean())
        rate_obs = float((bona >= THR[name]["obs"]).mean())
        rate_vad = float((bona >= THR[name]["vad"]).mean())
        summ.append(dict(
            model=name, n=n,
            mean_bona=bona.mean(), median_bona=float(np.median(bona)), std_bona=bona.std(),
            bona_rate_0p5_pct=100 * rate_05,
            bona_rate_eer_obs_pct=100 * rate_obs,
            bona_rate_eer_vad_pct=100 * rate_vad,
            thr_obs=THR[name]["obs"], thr_vad=THR[name]["vad"],
        ))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(OUT_DIR, "speechcommand_bonafide_summary.csv"), index=False)
    print(sdf.to_string(index=False))

    # ---- fig 1: bonafide detection rate bars (3 operating points) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ORDER)); w = 0.26
    for i, (col, lab) in enumerate([("bona_rate_0p5_pct", "@0.5 (argmax)"),
                                    ("bona_rate_eer_obs_pct", "@EER-thr observed"),
                                    ("bona_rate_eer_vad_pct", "@EER-thr vad")]):
        vals = [sdf[sdf.model == m][col].iloc[0] for m in ORDER]
        ax.bar(x + (i - 1) * w, vals, w, label=lab)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("Bonafide detection rate (%)")
    ax.set_title("SpeechCommand (all-bonafide) — bonafide detection rate by operating point")
    ax.set_ylim(min(90, sdf[["bona_rate_0p5_pct", "bona_rate_eer_obs_pct", "bona_rate_eer_vad_pct"]].values.min() - 2), 100.5)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "bonafide_detection_rate.png"), dpi=130)
    plt.close(fig)

    # ---- fig 2: bona-logit score distribution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-8, 8, 80)
    for m in ORDER:
        ax.hist(scores[m]["bona"].values, bins=bins, histtype="step", lw=1.8,
                color=COLORS[m], label=m, density=True)
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("bonafide logit (higher = more bonafide)")
    ax.set_ylabel("density")
    ax.set_title("SpeechCommand — bonafide-logit distribution (all clips are bonafide)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "bona_logit_distribution.png"), dpi=130)
    plt.close(fig)

    print("\nfigures:", FIG_DIR)


if __name__ == "__main__":
    main()
