"""
Main result tables of Sec. VI (generated from result files, no hand-typed numbers). tab:impact: AP@0.5 per attack under
ego-only perception, benign cooperation, no defense, PILD, and the oracles, for the single-attacker and the coordinated
collections. tab:results: one row per defense with its detection metrics (AUC, precision, recall, F1, FPR, FNR, recall
at 1% FPR, object-level AUC), the AP@0.5 and loss recovered when flagged CPMs are dropped or flagged objects filtered,
and its cost on benign traffic.
"""

import argparse
from pathlib import Path

import pandas as pd

from gencpm.report.common import fv, load_fusion, num, pct, table

SINGLE = ["CF", "CS", "CSD", "ISP", "GT", "SD", "RP", "PF"]
COORD = ["CG", "CSUP", "CCSD", "CISP"]
ROWS = [  # (label, drop condition, filter condition, detector key)
    ("Plausibility rule", "def:plausibility", "ofilt:plausibility", "plausibility"),
    ("Consistency rule", "def:consistency", "ofilt:consistency", "consistency"),
    ("Association-jump rule", None, None, "rule-nn_med"),
    ("LSTM", "def:LSTM", None, "LSTM"), ("GRU", "def:GRU", None, "GRU"), ("BiLSTM", "def:BiLSTM", None, "BiLSTM"),
    ("CNN--LSTM", "def:CNN-LSTM", None, "CNN-LSTM"), ("GBDT", "def:GBDT-rules", None, "GBDT-rules"),
    ("ROBOSAC$^\\dagger$", "robosac", None, None),
    ("\\textbf{PILD}", "def:PILD", "ofilt:PILD", "PILD"),
]
TRANSFER = [(f"{n}$^\\ast$", f"def:{k}-transfer", f"ofilt:{k}-transfer" if k == "PILD" else None, f"{k}-transfer")
            for n, k in [("LSTM", "LSTM"), ("GRU", "GRU"), ("BiLSTM", "BiLSTM"), ("CNN--LSTM", "CNN-LSTM"),
                         ("GBDT", "GBDT"), ("PILD", "PILD")]]
def _obj(work):
    f = Path(work) / "object_gate_summary.csv"
    if not f.exists():
        return {}
    o = pd.read_csv(f)
    o["det"] = o["object_gate"].str.replace(r"_seed\d+$", "", regex=True)
    return {k: (v["obj_auc"].mean(), v["obj_auc"].std(ddof=0) if len(v) > 1 else float("nan"),
                v["obj_tpr"].mean(), v["obj_fpr"].mean()) for k, v in o.groupby("det")}


def _pm(m, s):
    return f"{m:.3f}" if s != s else f"{m:.3f}$\\pm${s:.3f}"


def _ci(g, cond, s="attacked(any)"):
    """AP@0.5 on all attacked frames with the half-width of its 95% CI."""
    v, lo, hi = fv(g, s, cond), fv(g, s, cond, "lo"), fv(g, s, cond, "hi")
    if v != v:
        return "--", v
    return (num(v, 1) if lo != lo else f"{v:.1f}\\,{{\\tiny$\\pm${(hi - lo) / 2:.1f}}}"), v


def impact_table(g1, g2, caption, label):
    """Tab. impact: AP@0.5 per attack for the reference conditions, the best detector (PILD), and the oracles,
    single-attacker attacks on the left and coordinated attacks on the right, on the same rows."""
    rows = [("Ego-only perception", "ego_only", "ego_only"), ("Benign cooperation", "benign", "benign"),
            ("No defense", "attack", "attack"), None,
            ("PILD, drop CPMs", "def:PILD", "def:PILD"), ("PILD, filter objects", "ofilt:PILD", "ofilt:PILD"),
            ("Oracle, drop CPMs", "oracle", "oracle"), ("Oracle, filter objects", "ofilt:oracle-object", "ofilt:oracle-object")]
    n1, n2 = len(SINGLE) + 1, len(COORD) + 1
    head = [r"& \multicolumn{%d}{c}{Single-attacker attacks} & \multicolumn{%d}{c}{Coordinated attacks} \\" % (n1, n2),
            r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (n1 + 1, n1 + 2, n1 + n2 + 1),
            "& " + " & ".join(SINGLE + ["All"] + COORD + ["All"]) + r" \\"]
    body = []
    for r in rows:
        if r is None:
            body.append(r"\midrule")
            continue
        name, c1, c2 = r
        cells = [num(fv(g1, f"only {a}", c1), 1) for a in SINGLE] + [_ci(g1, c1)[0]]
        cells += [num(fv(g2, f"only {a}", c2), 1) for a in COORD] + [_ci(g2, c2)[0]]
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    return table("table*", caption, label, "l" + "c" * (n1 + n2), head, body, size=r"\footnotesize")


def defense_table(g1, d1, o1, g2, d2, o2, caption, label):
    """Tab. results: one row per defense. Detection metrics at the validation operating point, then AP@0.5 on all
    attacked frames (with the share of the attack-induced loss recovered) when flagged CPMs are dropped and when
    flagged objects are filtered, and the AP@0.5 of the same defense on the benign version of these frames."""
    ncol = 1 + 8 + 2 + 2
    head = [r"& \multicolumn{8}{c}{Detection} & \multicolumn{2}{c}{AP@0.5 under attack (loss rec.)} & \multicolumn{2}{c}{Benign AP@0.5} \\",
            r"\cmidrule(lr){2-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
            r"Defense & AUC & Prec. & Rec. & F1 & FPR & FNR & Rec.\,1\% & Obj.\ AUC & Drop CPMs & Filter objects & Drop & Filter \\"]
    body = []
    s = "attacked(any)"

    def mitig(g, cond, robosac=False):
        if cond is None:
            return "--", "--"
        txt, v = _ci(g, cond)
        att, ben = (fv(g, s, "attack@robosac"), fv(g, s, "benign@robosac")) if robosac else (fv(g, s, "attack"), fv(g, s, "benign"))
        rec = "--" if v != v else f"{round(100 * (v - att) / (ben - att)) + 0:d}"
        bv = fv(g, s, "brobosac" if robosac else "b" + cond)
        return (txt if v != v else f"{txt} ({rec})"), num(bv, 1)

    def det_cells(d, o, key):
        if key and key in d.index:
            r = d.loc[key]
            cells = [_pm(r["auc"], r["auc_std"]), pct(r["precision@5"]), pct(r["tpr@5"]), pct(r["f1@5"]),
                     pct(r["fpr@5"]), pct(r["fnr@5"]), pct(r["tpr@1"])]
        else:
            cells = ["--"] * 7
        ob = o.get(key) if key else None
        return cells + ["--" if ob is None else f"{ob[0]:.3f}"]

    def block(g, d, o, title, rows, first=False):
        body.append(("" if first else r"\midrule ") + r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, title))
        for name, dk, fk, key in rows:
            dr, db = mitig(g, dk, robosac=(dk == "robosac"))
            fr, fb = mitig(g, fk)
            body.append(f"{name} & " + " & ".join(det_cells(d, o, key) + [dr, fr, db, fb]) + r" \\")

    oracle = [("Oracle", "oracle", "ofilt:oracle-object", None)]
    block(g1, d1, o1, "Single-attacker attacks", ROWS + oracle, first=True)
    block(g2, d2, o2, "Coordinated attacks, detectors retrained on the coordinated collection", ROWS + oracle)
    block(g2, d2, o2, "Coordinated attacks, detectors trained on the single-attacker attacks only", TRANSFER)
    return table("table*", caption, label, "l" + "c" * (ncol - 1), head, body, size=r"\scriptsize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--coord-work", required=True)
    ap.add_argument("--fusion", nargs="+", required=True)
    ap.add_argument("--coord-fusion", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d1 = pd.read_csv(Path(a.work) / "detection_summary.csv", index_col=0)
    d2 = pd.read_csv(Path(a.coord_work) / "detection_summary.csv", index_col=0)
    g1, g2 = load_fusion(a.fusion), load_fusion(a.coord_fusion)
    o1, o2 = _obj(a.work), _obj(a.coord_work)
    cap1 = ("Attack impact. AP@0.5 (\\%) on the ego frames in which a CPM of the given attack is fused; All: every "
            "frame with a malicious CPM, with the half-width of the 95\\% episode-bootstrap confidence interval. PILD: "
            "best detector of Table~\\ref{tab:results}, retrained on the coordinated collection for coordinated attacks. "
            "Drop CPMs: flagged CPMs are discarded; filter objects: flagged objects are removed; oracle: perfect detection.")
    cap2 = ("Detection and mitigation. Detection: AUC as mean$\\pm$standard deviation; precision, recall, F1, FPR, and "
            "FNR in \\% at the validation operating point, false-positive rate at most 5\\%; Rec.\\,1\\%: recall at a "
            "false-positive rate of at most 1\\%; Obj.\\ AUC: object-level AUC of detectors that score objects. AP@0.5 "
            "(\\%) on all attacked frames with the 95\\% CI half-width and, in parentheses, the share of the "
            "attack-induced loss recovered; without defense 21.6 and 55.2. Benign: AP@0.5 of the defense on the benign "
            "version of the same frames, 75.2 and 76.6 without defense. $^\\dagger$consensus selection, evaluated on a "
            "random quarter of the attacked frames and compared with its own baselines. $^\\ast$trained on the "
            "single-attacker attacks only.")
    out = ["% Auto-generated by gencpm.report.main_tables -- do not edit by hand.",
           impact_table(g1, g2, cap1, "tab:impact"),
           defense_table(g1, d1, o1, g2, d2, o2, cap2, "tab:results")]
    Path(a.out).write_text("\n\n".join(out) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
