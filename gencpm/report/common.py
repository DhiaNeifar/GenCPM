"""Helpers shared by the table generators: reading fusion reports and LaTeX formatting."""

from pathlib import Path
import pandas as pd


def pct(x, nd=1):
    return "--" if pd.isna(x) else f"{100 * x:.{nd}f}"


def num(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def table(env, caption, label, colspec, header_lines, body, size=None, resize=False):
    out = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if size:
        out.append(size)
    if resize:
        out.append(r"\resizebox{\%s}{!}{%%" % ("textwidth" if env == "table*" else "columnwidth"))
    out += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + header_lines + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]
    if resize:
        out.append("}")
    out.append(rf"\end{{{env}}}")
    return "\n".join(out)


def load_fusion(dirs):
    dfs = []
    for d in dirs:
        f = Path(d) / "fusion_report.csv"
        if f.exists():
            x = pd.read_csv(f)
            if (x["condition"] == "robosac").any():
                # ROBOSAC is evaluated on its own subset of frames: keep its baselines for its recovery share
                x["condition"] = x["condition"].replace({"attack": "attack@robosac", "benign": "benign@robosac"})
            dfs.append(x)
    df = pd.concat(dfs).drop_duplicates(subset=["subset", "condition"])
    df["cond"] = df["condition"].str.replace(r":s\d+$", "", regex=True)
    return df.groupby(["subset", "cond"]).agg(AP30=("AP30", "mean"), AP50=("AP50", "mean"), AP70=("AP70", "mean"),
                                             lo=("AP50_ci_low", "mean"), hi=("AP50_ci_high", "mean"),
                                             n=("samples", "first")).reset_index()


def fv(g, subset, cond, col="AP50"):
    m = (g["subset"] == subset) & (g["cond"] == cond)
    return g.loc[m, col].iloc[0] if m.any() else float("nan")
