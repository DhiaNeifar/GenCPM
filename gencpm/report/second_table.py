"""
Cross-detector table: the same attack sets and defenses with CPMs generated from PointPillars and from SECOND
detections. On SECOND, "transferred" detectors are trained on PointPillars CPMs only.
Writes one LaTeX table (input from paper/revision_v2.tex).
"""

import argparse
from pathlib import Path

import pandas as pd

from gencpm.report.common import table

DET = [("PILD", "PILD, same detector"), ("PILD-transfer", "PILD, trained on PointPillars"),
       ("GBDT-rules", "GBDT, same detector"), ("GBDT-transfer", "GBDT, trained on PointPillars"),
       ("CNN-LSTM", "CNN--LSTM$^\\dagger$")]
FUS = [("ego_only", "Ego only (no cooperation)"), ("benign", "Benign cooperation"), ("attack", "Attack, no defense"),
       ("oracle", "Oracle: drop malicious CPMs"), ("ofilt:oracle-object", "Oracle: drop manipulated objects"),
       ("def:plausibility", "Plausibility, drop CPM"), ("robosac", "ROBOSAC consensus"),
       ("def:CNN-LSTM", "CNN--LSTM$^\\dagger$, drop CPM"),
       ("def:GBDT-rules", "GBDT, same detector, drop CPM"), ("def:GBDT-transfer", "GBDT, trained on PointPillars, drop CPM"),
       ("def:PILD", "PILD, same detector, drop CPM"), ("def:PILD-transfer", "PILD, trained on PointPillars, drop CPM"),
       ("ofilt:PILD", "PILD, same detector, drop objects"),
       ("ofilt:PILD-transfer", "PILD, trained on PointPillars, drop objects")]


def fusion(dirs):
    d = pd.concat([pd.read_csv(Path(x) / "fusion_report.csv") for x in dirs if (Path(x) / "fusion_report.csv").exists()])
    d = d.drop_duplicates(["subset", "condition"])
    d = d[d["subset"] == "attacked(any)"].copy()
    d["c"] = d["condition"].str.replace(r":s\d+$", "", regex=True)
    return d.groupby("c")["AP50"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp-work", required=True, help="PointPillars work dir (eight attacks)")
    ap.add_argument("--second-work", required=True, help="SECOND work dir (eight attacks)")
    ap.add_argument("--pp-fusion", required=True, help="comma-separated PointPillars fusion dirs")
    ap.add_argument("--second-fusion", required=True, help="comma-separated SECOND fusion dirs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = ["pp", "second"]
    det = {m: pd.read_csv(Path(getattr(a, f"{m}_work")) / "detection_summary.csv", index_col=0) for m in cols}
    fus = {m: fusion(getattr(a, f"{m}_fusion").split(",")) for m in cols}

    body = [r"\multicolumn{3}{l}{\textit{CPM-level detection, AUC}} \\"]
    for key, name in DET:
        cells = [f"{det[m].loc[key, 'auc']:.3f}" if key in det[m].index else "--" for m in cols]
        body.append(f"\\quad {name} & " + " & ".join(cells) + r" \\")
    body += [r"\midrule", r"\multicolumn{3}{l}{\textit{AP@0.5 on attacked frames, share of the loss recovered}} \\"]
    for key, name in FUS:
        cells = []
        for m in cols:
            f = fus[m]
            if key not in f.index:
                cells.append("--")
                continue
            v, b, at = f[key], f["benign"], f["attack"]
            rec = "" if key in ("ego_only", "benign", "attack") else f" ({100 * (v - at) / (b - at):.0f}\\%)"
            cells.append(f"{v:.1f}{rec}")
        body.append(f"\\quad {name} & " + " & ".join(cells) + r" \\")
    head = [r"& PointPillars & SECOND \\"]
    cap = ("Cross-detector evaluation. CPMs generated from PointPillars and from SECOND detections, with the same attack "
           "schedules. Same detector: trained and tested on CPMs of that detector. Trained on PointPillars: applied to "
           "SECOND CPMs without retraining. $^\\dagger$on SECOND, trained on PointPillars CPMs. Rules are "
           "calibrated on benign traffic of each detector; ROBOSAC is evaluated with PointPillars only.")
    Path(a.out).write_text(table("table", cap, "tab:rev_second", "lcc", head, body, resize=True) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
