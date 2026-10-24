"""
Figures of the revised evaluation (eight single-attacker attacks, coordinated attacks), from saved results only:
  roc_dense.png            ROC curves of every CPM-level detector (pooled test folds, averaged over runs): (a) single-attacker
                           attacks, (b) coordinated attacks, retrained, (c) coordinated attacks, trained on (a);
                           AUC printed in each panel in the method's color
  recovery_per_attack.png  AP@0.5 per attack: no defense, PILD dropping CPMs, PILD filtering objects, the two
                           oracles; benign cooperation and ego-only perception as reference marks
  auc_vs_window_dense.png  validation AUC vs. window length W for the temporal models (W with results only)
Colors: reference categorical palette in fixed order (validated); markers / hatch as second encoding.
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#9a9890"

# every CPM-level detector, one color and marker each, in the same order in every panel (as attackers_vs_drop.png)
METHODS = [("defenses_W20", "plausibility", "Plausibility rule"), ("defenses_W20", "consistency", "Consistency rule"),
           ("baselines_W20", "rule-nn_med", "Association-jump rule"), ("temporal_W20_global", "LSTM", "LSTM"),
           ("temporal_W20_global", "GRU", "GRU"), ("temporal_W20_global", "BiLSTM", "BiLSTM"),
           ("temporal_W20_global", "CNN-LSTM", "CNN-LSTM"), ("baselines_W20", "GBDT-rules", "GBDT"),
           ("defenses_W20", "PILD", "PILD")]
TRANSFER = {"LSTM": ("temporal_W20_transfer", "LSTM-transfer"), "GRU": ("temporal_W20_transfer", "GRU-transfer"),
            "BiLSTM": ("temporal_W20_transfer", "BiLSTM-transfer"), "CNN-LSTM": ("temporal_W20_transfer", "CNN-LSTM-transfer"),
            "GBDT-rules": ("defenses_W20", "GBDT-transfer"), "PILD": ("defenses_W20", "PILD-transfer")}
COLORS = [C[1], C[2], "#52514e", C[3], C[4], C[5], C[6], C[7], C[0]]  # PILD in the first palette color
MARK = "osD^vP*Xh"
MARK_AT = np.array([2, 5, 10, 20, 40, 70])  # FPR (%) at which markers are drawn


def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)


def pooled(work, d, model, seed=0):
    meta = pd.read_pickle(work / "cpm_meta.pkl")
    ends = np.load(work / d / "window_ends.npy")
    y = meta["label"].to_numpy()[ends]
    ys, ps = [], []
    for f in sorted(glob.glob(str(work / d / f"{model}_fold*_seed{seed}_probs.npz"))):
        z = np.load(f)
        ys.append(y[z["test_idx"]]); ps.append(z["p_test"])
    return (np.concatenate(ys), np.concatenate(ps)) if ys else (None, None)


def roc_panel(ax, work, transfer, title):
    """ROC curve of every detector; transfer: detectors trained on the single-attacker attacks only (rules unchanged)."""
    for i, (d, m, label) in enumerate(METHODS):
        if transfer and m in TRANSFER:
            d, m = TRANSFER[m]
        # vertical averaging over runs: TPR of each run on a common FPR grid, then the mean (rules: a single run)
        grid, tprs, aucs = np.linspace(0, 1, 501), [], []
        for seed in range(3):
            y, p = pooled(work, d, m, seed)
            if y is None:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            tprs.append(np.interp(grid, fpr, tpr)); aucs.append(roc_auc_score(y, p))
        if not tprs:
            continue
        x, t = 100 * grid, 100 * np.mean(tprs, axis=0)
        t[0] = 0.0
        ax.plot(x, t, color=COLORS[i], lw=1.8 if label == "PILD" else 1.2, zorder=3 if label == "PILD" else 2)
        ax.plot(MARK_AT, np.interp(MARK_AT, x, t), ls="none", marker=MARK[i], ms=3.5, color=COLORS[i],
                zorder=3 if label == "PILD" else 2)
        ax.text(0.98, 0.04 + 0.075 * (len(METHODS) - 1 - i), f"{np.mean(aucs):.3f}", color=COLORS[i],
                fontsize=6.5, ha="right", transform=ax.transAxes)
    ax.plot([0, 100], [0, 100], color=NEUTRAL, lw=0.8, ls=(0, (1, 2)))
    ax.axvline(5, color=INK2, lw=0.6, ls="-.")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("False-positive rate (%)")
    ax.set_title(title, color=INK); style(ax)


def fusion_table(path):
    d = pd.read_csv(path)
    d["c"] = d["condition"].str.replace(r":s\d+$", "", regex=True)
    return d.groupby(["subset", "c"])["AP50"].mean().unstack(0)


def bars(ax, t, attacks, series, title):
    x = np.arange(len(attacks))
    n = len(series)
    w = 0.8 / n
    for i, (cond, label, color, hatch) in enumerate(series):
        v = [t.loc[cond, f"only {a}"] for a in attacks]
        ax.bar(x - 0.4 + w * (i + 0.5), v, w * 0.92, color=color if not hatch else "white",
               edgecolor=color, hatch=hatch, lw=0.8, label=label)
    for j, a in enumerate(attacks):
        ax.hlines(t.loc["benign", f"only {a}"], j - 0.44, j + 0.44, color=INK, lw=1.4,
                  label="Benign cooperation" if j == 0 else None)
        ax.hlines(t.loc["ego_only", f"only {a}"], j - 0.44, j + 0.44, color=INK2, lw=1.0, ls=(0, (3, 2)),
                  label="Ego only" if j == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(attacks)
    ax.set_ylabel("AP@0.5 (%)"); ax.set_ylim(0, 80)
    ax.set_title(title, color=INK); style(ax); ax.grid(axis="x", visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="eight-attack work dir")
    ap.add_argument("--coord-work", required=True)
    ap.add_argument("--fusion", required=True, help="eight-attack fusion_report.csv")
    ap.add_argument("--coord-fusion", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--roc-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.titlesize": 8.5, "hatch.linewidth": 0.6})

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.9), dpi=300, sharey=True)
    roc_panel(ax[0], Path(a.work), False, "(a) Single-attacker attacks")
    roc_panel(ax[1], Path(a.coord_work), False, "(b) Coordinated, retrained")
    roc_panel(ax[2], Path(a.coord_work), True, "(c) Coordinated, trained on (a)")
    ax[0].set_ylabel("True-positive rate (%)")
    handles = [plt.Line2D([], [], color=COLORS[i], marker=MARK[i], ms=3.5, lw=1.2, label=l)
               for i, (_, _, l) in enumerate(METHODS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(out / "roc_dense.png"); plt.close(fig)
    if a.roc_only:
        return

    t1, t2 = fusion_table(a.fusion), fusion_table(a.coord_fusion)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.9), dpi=300, gridspec_kw={"width_ratios": [2, 1.15]})
    s1 = [("attack", "No defense", NEUTRAL, None), ("def:PILD", "PILD, drop CPM", C[0], None),
          ("ofilt:PILD", "PILD, filter objects", C[1], None), ("oracle", "Oracle, drop CPM", C[0], "////"),
          ("ofilt:oracle-object", "Oracle, filter objects", C[1], "////")]
    bars(ax[0], t1, ["CF", "CS", "CSD", "ISP", "GT", "SD", "RP", "PF"], s1, "(a) Single-attacker attacks")
    s2 = [("attack", "No defense", NEUTRAL, None), ("def:PILD-transfer", "PILD$^\\ast$, drop CPM", C[0], "...."),
          ("ofilt:PILD-transfer", "PILD$^\\ast$, filter objects", C[1], "...."), ("def:PILD", "PILD, drop CPM", C[0], None),
          ("ofilt:PILD", "PILD, filter objects", C[1], None), ("oracle", "Oracle, drop CPM", C[0], "////"),
          ("ofilt:oracle-object", "Oracle, filter objects", C[1], "////")]
    bars(ax[1], t2, ["CG", "CSUP", "CCSD", "CISP"], s2, "(b) Coordinated attacks")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(out / "recovery_per_attack.png"); plt.close(fig)

    work = Path(a.work)
    meta = pd.read_pickle(work / "cpm_meta.pkl")
    rows = []
    for W in [3, 5, 10, 20]:
        d = work / f"temporal_W{W}_global"
        if not (d / "window_ends.npy").exists():
            continue
        ends = np.load(d / "window_ends.npy"); y = meta["label"].to_numpy()[ends]
        for f in sorted(glob.glob(str(d / "*_probs.npz"))):
            z = np.load(f)
            rows.append({"W": W, "model": Path(f).name.split("_fold")[0],
                         "val_auc": roc_auc_score(y[z["val_idx"]], z["p_val"])})
    if rows:
        df = pd.DataFrame(rows).groupby(["model", "W"])["val_auc"].agg(["mean", "std", "count"]).reset_index()
        df.to_csv(out / "auc_vs_window_dense.csv", index=False)
        fig, axw = plt.subplots(figsize=(3.5, 2.6), dpi=300)
        idx = {m: i for i, (_, m, _) in enumerate(METHODS)}  # same color and marker per model as in roc_dense.png
        order = [m for _, m, _ in METHODS if m in set(df["model"])]
        for j, m in enumerate(order):
            g, color, mk = df[df["model"] == m], COLORS[idx[m]], MARK[idx[m]]
            xw = g["W"].to_numpy(float) * (1 + 0.035 * (j - 1.5))  # small offset so error bars do not overlap
            axw.errorbar(xw, 100 * g["mean"], yerr=100 * g["std"], marker=mk, ms=4, capsize=2.5, lw=1.2,
                         color=color, label=m)
        axw.set_xscale("log"); axw.minorticks_off()
        axw.set_xticks([3, 5, 10, 20]); axw.set_xticklabels(["3", "5", "10", "20"])
        axw.set_xlabel("Window length $W$ (CPMs)"); axw.set_ylabel("Validation AUC (%)")
        style(axw); axw.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
        fig.tight_layout(); fig.savefig(out / "auc_vs_window_dense.png"); plt.close(fig)
        print(df.round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
