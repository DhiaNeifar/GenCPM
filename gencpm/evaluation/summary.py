"""
Summarize CPM-level detection results of every detector in one or more result dirs of a work dir.

For each (detector, fold, seed): AUC, PR-AUC, TPR / FPR at the validation-chosen FPR<=5% threshold,
and per-attack AUC (attack vs benign) and recall at that threshold. Averaged over folds x seeds.
Object-level gates (defenses_W*/objgate_*.npz) are summarized as object AUC and object TPR/FPR.

Writes <work>/detection_summary.csv and prints a table.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SHORT = {"AddObjectAttack": "CF", "RemoveObjectAttack": "CS", "DriftAttack": "CSD", "WhiteNoiseAttack": "ISP",
         "GhostTrackAttack": "GT", "StealthDriftAttack": "SD", "ReplayAttack": "RP", "EvasiveGhostAttack": "EG", "PoseFalsificationAttack": "PF",
         "ColludingGhostAttack": "CG", "CoordinatedSuppressionAttack": "CSUP", "CoordinatedDriftAttack": "CCSD",
         "CoordinatedISPAttack": "CISP"}


def thr_at_fpr(y, p, fpr_max=0.05):
    fpr, tpr, th = roc_curve(y, p)
    ok = np.where(fpr <= fpr_max)[0]
    return th[ok[np.argmax(tpr[ok])]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dirs", nargs="+", default=["temporal_W20_global", "baselines_W20", "defenses_W20"])
    ap.add_argument("--exclude", nargs="*", default=["rule-count_delta", "rule-uncorroborated", "rule-yaw_med",
                                                     "rule-nn_max"])
    args = ap.parse_args()
    wd = Path(args.work_dir)
    meta = pd.read_pickle(wd / "cpm_meta.pkl")
    attacks = [a for a in SHORT if a in set(meta["attack_type"])]
    rows = []
    for d in args.dirs:
        dd = wd / d
        if not dd.exists():
            continue
        ends = np.load(dd / "window_ends.npy")
        y = meta["label"].to_numpy()[ends]
        atk = meta["attack_type"].to_numpy()[ends]
        for f in sorted(glob.glob(str(dd / "*_probs.npz"))):
            name = Path(f).name.split("_fold")[0]
            if name in args.exclude:
                continue
            z = np.load(f)
            yv, pv, yt, pt, at = y[z["val_idx"]], z["p_val"], y[z["test_idx"]], z["p_test"], atk[z["test_idx"]]
            thr = thr_at_fpr(yv, pv)
            flag = pt >= thr
            tp, fp = int((flag & (yt == 1)).sum()), int((flag & (yt == 0)).sum())
            prec = tp / max(1, tp + fp)
            rec = float(flag[yt == 1].mean())
            r = {"detector": name, "auc": roc_auc_score(yt, pt), "pr_auc": average_precision_score(yt, pt),
                 "tpr@5": rec, "fpr@5": float(flag[yt == 0].mean()), "precision@5": prec,
                 "f1@5": 2 * prec * rec / max(1e-12, prec + rec), "fnr@5": 1 - rec}
            thr1 = thr_at_fpr(yv, pv, 0.01)  # stricter operating point: FPR <= 1% on validation
            r["tpr@1"] = float((pt[yt == 1] >= thr1).mean())
            r["fpr@1"] = float((pt[yt == 0] >= thr1).mean())
            if "p_test_benign" in z:
                pb = z["p_test_benign"]
                r["fpr_benign_version@5"] = float((pb[yt == 1] >= thr).mean())
            for a in attacks:
                m = (at == a) | (yt == 0)
                r[f"auc_{SHORT[a]}"] = roc_auc_score(yt[m], pt[m]) if (at == a).any() else np.nan
                r[f"rec_{SHORT[a]}"] = float((pt[at == a] >= thr).mean()) if (at == a).any() else np.nan
            rows.append(r)
    df = pd.DataFrame(rows).groupby("detector").agg(["mean", "std"])
    flat = df.xs("mean", axis=1, level=1).copy()
    flat["auc_std"] = df[("auc", "std")]
    flat = flat.sort_values("auc")
    flat.to_csv(wd / "detection_summary.csv")

    # object-level gates
    orows = []
    fz = wd / "defense_features.npz"
    if fz.exists():
        z = np.load(fz)
        lab, orow = z["obj_label"], z["obj_row"]
        for f in sorted(glob.glob(str(wd / "defenses_W*" / "objgate_*.npz"))):
            g = np.load(f)
            m = ~np.isnan(g["score"])
            s, l = g["score"][m], lab[m]
            thr = g["thr_row"][orow[m]]
            flag = s >= thr
            orows.append({"object_gate": Path(f).stem.replace("objgate_", ""), "obj_auc": roc_auc_score(l, s),
                          "obj_tpr": float(flag[l == 1].mean()), "obj_fpr": float(flag[l == 0].mean())})
    odf = pd.DataFrame(orows)
    if len(odf):
        odf.to_csv(wd / "object_gate_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(flat.round(3).to_string())
    if len(odf):
        print("\nObject-level gates (test folds):\n" + odf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
