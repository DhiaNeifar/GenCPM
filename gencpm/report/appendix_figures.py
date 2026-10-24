"""
Appendix figures from the attack sweeps (`python -m gencpm sweep report` of configs/sweeps/*.json):
  param_sensitivity.png              one panel per attack (eight single-attacker attacks with one attacker, four
                                     coordinated attacks with one group of three colluders): AP@0.3/0.5/0.7 under
                                     attack vs. each parameter of the attack, parameters side by side, the others at
                                     their defaults
  attackers_vs_drop.png              relative AP@0.5 drop vs. number of attackers k, independent and coordinated
Error bars and bands: 95% episode-bootstrap CI. Reference lines: benign cooperation and attackers excluded,
on the same frames. Colors: validated reference categorical palette, fixed order; markers as second encoding.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9a9890"
MARK = "osD^vP*X"
NAMES = {"CF": "CF", "CS": "CS", "CSD": "CSD", "ISP": "ISP", "GT": "GT", "SD": "SD", "RP": "RP", "PF": "PF",
         "CG": "CG", "CSUP": "CSUP", "CCSD": "CCSD", "CISP": "CISP"}
LABELS = {"num_objects": "Fabricated objects per CPM", "radius_max": "Max. placement radius (m)",
          "remove_fraction": "Removed fraction", "step_m": "Drift step per CPM (m)",
          "yaw_step_deg": "Yaw step per CPM (deg)", "targets": "Targeted objects", "noise_m": "Mean position noise (m)",
          "yaw_noise_deg": "Mean yaw noise (deg)", "num_ghosts": "Ghost vehicles", "lane_offset_m": "Lateral offset (m)",
          "longitudinal_m": "Longitudinal range (m)", "rate_mps": "Drift rate (m/s)", "cap_m": "Max. drift (m)",
          "delay_s": "Replay delay (s)", "offset_m": "Pose offset (m)", "yaw_deg": "Pose rotation (deg)"}
DEFAULTS = {("CF", "num_objects"): 25, ("CF", "radius_max"): 40, ("CS", "remove_fraction"): 0.95,
            ("CSD", "step_m"): 2.0, ("CSD", "yaw_step_deg"): 10, ("CSD", "targets"): 10000,
            ("ISP", "noise_m"): 3.0, ("ISP", "yaw_noise_deg"): 15, ("ISP", "targets"): 10000,
            ("GT", "num_ghosts"): 2, ("GT", "lane_offset_m"): 3.5, ("SD", "rate_mps"): 1.5, ("SD", "cap_m"): 8.0,
            ("RP", "delay_s"): 2.0, ("PF", "offset_m"): 4.0, ("PF", "yaw_deg"): 5.0,
            ("CG", "num_ghosts"): 2, ("CG", "lane_offset_m"): 3.5, ("CG", "longitudinal_m"): 10.0,
            ("CSUP", "targets"): 2, ("CCSD", "rate_mps"): 1.5, ("CCSD", "targets"): 4, ("CCSD", "cap_m"): 8.0,
            ("CISP", "noise_m"): 3.0, ("CISP", "yaw_noise_deg"): 15}
SINGLE = [("CF", "num_objects"), ("CF", "radius_max"), ("CS", "remove_fraction"), ("CSD", "step_m"),
          ("CSD", "yaw_step_deg"), ("CSD", "targets"), ("ISP", "noise_m"), ("ISP", "yaw_noise_deg"),
          ("ISP", "targets"), ("GT", "num_ghosts"), ("GT", "lane_offset_m"), ("SD", "rate_mps"), ("SD", "cap_m"),
          ("RP", "delay_s"), ("PF", "offset_m"), ("PF", "yaw_deg")]
COORD = [("CG", "num_ghosts"), ("CG", "lane_offset_m"), ("CG", "longitudinal_m"), ("CSUP", "targets"),
         ("CCSD", "rate_mps"), ("CCSD", "targets"), ("CCSD", "cap_m"), ("CISP", "noise_m"), ("CISP", "yaw_noise_deg")]
IND_ATT = ["CF", "CS", "CSD", "ISP", "GT", "SD", "RP", "PF"]
COORD_ATT = ["CG", "CSUP", "CCSD", "CISP"]


def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)


SHORT = {"num_objects": "Objects", "radius_max": "Radius (m)", "remove_fraction": "Removed fraction",
         "step_m": "Step (m)", "yaw_step_deg": "Yaw step (deg)", "targets": "Targets", "noise_m": "Noise (m)",
         "yaw_noise_deg": "Yaw noise (deg)", "num_ghosts": "Ghosts", "lane_offset_m": "Lateral (m)",
         "longitudinal_m": "Longitudinal (m)", "rate_mps": "Rate (m/s)", "cap_m": "Max. drift (m)",
         "delay_s": "Delay (s)", "offset_m": "Offset (m)", "yaw_deg": "Rotation (deg)"}
THR = [("50", "AP@0.5", C[0], "o"), ("30", "AP@0.3", C[1], "s"), ("70", "AP@0.7", C[2], "D")]


def param_grid(df, path):
    """One panel per attack; each parameter of the attack is a group of x positions, with a gap between groups."""
    attacks = IND_ATT + COORD_ATT
    fig, axes = plt.subplots(4, 3, figsize=(7.16, 4.8), dpi=300, sharey=True)
    for ax, atk in zip(axes.ravel(), attacks):
        prms = [p for a, p in SINGLE + COORD if a == atk]
        x0, ticks, labels = 0, [], []
        for prm in prms:
            g = df[(df["attack"] == atk) & (df["param"] == prm)].sort_values("value")
            xs = np.arange(len(g)) + x0
            d = DEFAULTS.get((atk, prm))
            for key, name, col, mk in THR:
                ax.plot(xs, g[f"ap{key}"], marker=mk, ms=3, lw=1.2, color=col, label=name)
                if key == "50":
                    ax.vlines(xs, g["lo"], g["hi"], color=col, lw=0.8)
            if d is not None and np.isclose(g["value"], d).any():
                ax.axvspan(xs[0] + np.argmax(np.isclose(g["value"], d)) - 0.2,
                           xs[0] + np.argmax(np.isclose(g["value"], d)) + 0.2, color=GRID, lw=0, zorder=0)
            ticks += list(xs); labels += ["all" if v >= 10000 else f"{v:g}".replace("0.", ".", 1) if 0 < v < 1 else f"{v:g}"
                       for v in g["value"]]
            ax.text(xs.mean(), -0.30, SHORT[prm], transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6, color=INK2)
            if prm != prms[-1]:
                ax.axvline(xs[-1] + 1, color=GRID, lw=0.8)
            x0 = xs[-1] + 2
        for key, name, col, mk in THR:
            ax.axhline(df.loc[df["attack"] == atk, f"benign{key}"].mean(), color=col, lw=0.8, ls=(0, (1, 1.5)))
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.3)
        ax.set_xlim(-0.6, x0 - 1.4); ax.set_ylim(0, 90)
        who = "one attacker" if atk in IND_ATT else "three colluders"
        ax.set_title(f"{NAMES[atk]}, {who}", color=INK, fontsize=7.5, pad=2)
        style(ax); ax.grid(axis="x", visible=False); ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("AP under attack (%)", fontsize=6.5)
    h = [plt.Line2D([], [], color=c, marker=m, ms=3, lw=1.2, label=n) for _, n, c, m in sorted(THR, key=lambda t: t[0])]
    h.append(plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="Benign cooperation, same color per AP"))
    h.append(plt.Rectangle((0, 0), 1, 1, color=GRID, label="Default value"))
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.6, w_pad=0.5); fig.savefig(path); plt.close(fig)


def attackers_figure(rep, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), dpi=300, sharey=True)
    for ax, atts, title in ((axes[0], IND_ATT, "(a) Independent attackers"), (axes[1], COORD_ATT, "(b) Colluding attackers")):
        for i, a in enumerate(atts):
            g = rep[(rep["group"] == "nattack") & (rep["attack"] == a)].sort_values("k")
            ax.fill_between(g["k"], g["drop50_ci_low"], g["drop50_ci_high"], color=C[i], alpha=0.15, lw=0)
            ax.plot(g["k"], g["drop50_pct"], marker=MARK[i], ms=3.5, lw=1.3, color=C[i], label=a)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Number of attackers $k$")
        ax.set_title(title, color=INK); style(ax); ax.legend(fontsize=6.5, frameon=False, ncol=2)
    axes[0].set_ylabel("Relative AP@0.5 drop (%)"); axes[0].set_ylim(0, 85)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--coord-report", default=None, help="coordinated sweep with shared targets (replaces those rows)")
    ap.add_argument("--coord-configs", default=None)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.titlesize": 8})
    rep = pd.read_csv(a.report)
    cfgs = json.loads(Path(a.configs).read_text())
    if a.coord_report:  # coordinated attacks: use the shared-target sweep
        crep = pd.read_csv(a.coord_report)
        rep = pd.concat([rep[~rep["attack"].isin(COORD_ATT)], crep], ignore_index=True)
        cfgs = [c for c in cfgs if c["attack"] not in COORD_ATT] + json.loads(Path(a.coord_configs).read_text())
    by = rep.set_index("name")
    rows = []
    for c in cfgs:
        if c["group"] != "param" or c["name"] not in by.index:
            continue
        r = by.loc[c["name"]]
        for sw in c.get("sweeps", []):
            rows.append({"attack": c["attack"], "k": c["k"], "param": sw["param"], "value": sw["value"],
                         "ap": r["attack_AP50"], "lo": r["attack_AP50_ci_low"], "hi": r["attack_AP50_ci_high"],
                         "benign": r["benign_AP50"], "no_att": r["no_att_AP50"],
                         **{f"ap{t}": r[f"attack_AP{t}"] for t in ("30", "50", "70")},
                         **{f"benign{t}": r[f"benign_AP{t}"] for t in ("30", "50", "70")}})
    df = pd.DataFrame(rows)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "param_sensitivity_dense.csv", index=False)
    param_grid(df, out / "param_sensitivity.png")
    attackers_figure(rep, out / "attackers_vs_drop.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
