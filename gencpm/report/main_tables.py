"""
Main result table of Sec. VI, tab:results (generated from result files, no hand-typed numbers): for the single-attacker
and the coordinated collections, AP@0.5 per attack under ego-only perception, benign cooperation, no defense, and every
defense (drop flagged CPMs; filter flagged objects where the detector scores objects; the two oracles), the benign
cost and loss recovered of each defense, and its detection metrics (AUC, precision, recall, F1, FPR, FNR, recall at
1% FPR).
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
NA = 8  # attack columns
NCOL = 1 + NA + 3 + 7


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


def results_table(g1, d1, o1, g2, d2, o2, caption, label):
    """One table for attack impact, detection, and mitigation. Rows: reference conditions and defenses (each
    detector drops flagged CPMs and, when it scores objects, also filters flagged objects). Columns: AP@0.5 per
    attack and on all attacked frames (with the 95% CI half-width), AP@0.5 of the defense on the benign version of
    the same frames, the share of the attack-induced loss recovered, and the detection metrics."""
    head = [r"& \multicolumn{%d}{c}{AP@0.5 on attacked frames (\%%)} & Benign & Loss & \multicolumn{7}{c}{Detection} \\" % (NA + 1),
            r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (NA + 2, NA + 5, NCOL),
            None,
            r"& & & & & & & & & & AP (\%) & rec. (\%) & AUC & Prec. & Rec. & F1 & FPR & FNR & Rec.\,1\% \\"]
    body = []

    def attack_names(attacks):
        cells = [a for a in attacks] + [""] * (NA - len(attacks))
        return "Defense & " + " & ".join(cells) + r" & All \\"

    def ap_cells(g, attacks, cond, base=("attack", "benign")):
        s = "attacked(any)"
        cells = [num(fv(g, f"only {a}", cond), 1) if cond else "--" for a in attacks] + [""] * (NA - len(attacks))
        v, lo, hi = (fv(g, s, cond), fv(g, s, cond, "lo"), fv(g, s, cond, "hi")) if cond else (float("nan"),) * 3
        cells.append("--" if v != v else (num(v, 1) if lo != lo else f"{v:.1f}\\,{{\\tiny$\\pm${(hi - lo) / 2:.1f}}}"))
        return cells, v

    def block(g, d, o, attacks, title, rows):
        body.append((r"\midrule " if body else "") + r"\multicolumn{%d}{l}{\textit{%s}} \\" % (NCOL, title))
        body.append(attack_names(attacks).replace("Defense", ""))
        body.append(r"\midrule")
        s = "attacked(any)"
        for name, cond in [("Ego-only perception", "ego_only"), ("Benign cooperation", "benign"), ("No defense", "attack")]:
            cells, _ = ap_cells(g, attacks, cond)
            body.append(f"{name} & " + " & ".join(cells + [""] * 9) + r" \\")
        body.append(r"\midrule")

        def emit(name, cond, bcond, rec_key, det_cells):
            cells, v = ap_cells(g, attacks, cond)
            att, ben = (fv(g, s, "attack@robosac"), fv(g, s, "benign@robosac")) if cond == "robosac" else \
                       (fv(g, s, "attack"), fv(g, s, "benign"))
            bv = fv(g, s, bcond) if bcond else float("nan")
            rec = "--" if v != v else f"{round(100 * (v - att) / (ben - att)) + 0:d}"
            body.append(f"{name} & " + " & ".join(cells + [num(bv, 1) if bcond else "--", rec] + det_cells) + r" \\")

        for name, dk, fk, key in rows:
            if key and key in d.index:
                r = d.loc[key]
                det = [_pm(r["auc"], r["auc_std"]), pct(r["precision@5"]), pct(r["tpr@5"]), pct(r["f1@5"]),
                       pct(r["fpr@5"]), pct(r["fnr@5"]), pct(r["tpr@1"])]
            else:
                det = ["--"] * 7
            bdk = None if dk is None else ("brobosac" if dk == "robosac" else "b" + dk)
            emit(name + (", drop CPMs" if fk else ""), dk, bdk, key, det)
            if fk:
                ob = o.get(key)
                odet = ["--"] * 7 if ob is None else [_pm(ob[0], ob[1]), "--", pct(ob[2]), "--", pct(ob[3]), pct(1 - ob[2]), "--"]
                emit("\\quad filter objects$^\\ddagger$", fk, "b" + fk, key, odet)
        body.append(r"\midrule")
        emit("Oracle, drop CPMs", "oracle", None, None, ["--"] * 7)
        emit("\\quad filter objects", "ofilt:oracle-object", None, None, ["--"] * 7)

    block(g1, d1, o1, SINGLE, "Single-attacker attacks", ROWS)
    block(g2, d2, o2, COORD, "Coordinated attacks, detectors retrained on the coordinated collection", ROWS)
    body.append(r"\midrule \multicolumn{%d}{l}{\textit{Coordinated attacks, detectors trained on the single-attacker attacks only}} \\" % NCOL)
    # transfer rows reuse the coordinated block's frames
    s = "attacked(any)"
    for name, dk, fk, key in TRANSFER:
        for nm, cond, ob in [(name + (", drop CPMs" if fk else ""), dk, False), ("\\quad filter objects$^\\ddagger$", fk, True)]:
            if cond is None:
                continue
            cells = [num(fv(g2, f"only {a}", cond), 1) for a in COORD] + [""] * (NA - len(COORD))
            v, lo, hi = fv(g2, s, cond), fv(g2, s, cond, "lo"), fv(g2, s, cond, "hi")
            cells.append(f"{v:.1f}\\,{{\\tiny$\\pm${(hi - lo) / 2:.1f}}}")
            att, ben = fv(g2, s, "attack"), fv(g2, s, "benign")
            cells += [num(fv(g2, s, "b" + cond), 1), f"{round(100 * (v - att) / (ben - att)) + 0:d}"]
            if ob:
                x = o2.get(key)
                cells += ["--"] * 7 if x is None else [_pm(x[0], x[1]), "--", pct(x[2]), "--", pct(x[3]), pct(1 - x[2]), "--"]
            elif key in d2.index:
                r = d2.loc[key]
                cells += [_pm(r["auc"], r["auc_std"]), pct(r["precision@5"]), pct(r["tpr@5"]), pct(r["f1@5"]),
                          pct(r["fpr@5"]), pct(r["fnr@5"]), pct(r["tpr@1"])]
            else:
                cells += ["--"] * 7
            body.append(f"{nm} & " + " & ".join(cells) + r" \\")
    head = [h for h in head if h is not None]
    return table("table*", caption, label, "l" + "c" * (NCOL - 1), head, body, size=r"\scriptsize", resize=True)


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
    cap = ("Attack impact, detection, and mitigation. AP@0.5 (\\%) on the ego frames in which a CPM of the given attack "
           "is fused; All: every frame with a malicious CPM, with the half-width of the 95\\% episode-bootstrap "
           "confidence interval. Drop CPMs: flagged CPMs are discarded; filter objects: flagged objects are removed; "
           "oracle: perfect detection. Benign AP: AP@0.5 of the defense on the benign version of the same frames, "
           "75.2 and 76.6 without defense. Loss rec.: share of the attack-induced AP@0.5 loss recovered on all "
           "attacked frames. Detection: AUC as mean$\\pm$standard deviation; precision, "
           "recall, F1, FPR, and FNR in \\% at the validation operating point, false-positive rate at most 5\\%; "
           "Rec.\\,1\\%: recall at a false-positive rate of at most 1\\%. CPM-level detection, except "
           "$^\\ddagger$object-level detection. $^\\dagger$consensus selection, evaluated on a random quarter of the "
           "attacked frames and compared with its own baselines. $^\\ast$trained on the single-attacker attacks only.")
    out = ["% Auto-generated by gencpm.report.main_tables -- do not edit by hand.",
           results_table(g1, d1, o1, g2, d2, o2, cap, "tab:results")]
    Path(a.out).write_text("\n\n".join(out) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
