"""
Build the coordinated-attack collection directly in benchmark cache format
(cpm_meta.pkl, cpm_objects.npz, cpm_boxes.npz), from the frames cache.

Attacker budget matched to the single-attacker collection: every CAV may attack. Per episode, all CAVs are
partitioned into colluding groups of `group_size` CAVs that are often close to each other
(scenarios.Episode.form_groups). Each group attacks in back-to-back intervals of `interval_s` seconds: in each
interval it is active with probability `duty` and then runs one coordinated attack drawn uniformly, shared by
the whole group: colluding ghosts, coordinated suppression, coordinated drift, coordinated ISP. The default
duty, 1 - exp(-0.2 * 5) = 0.632, equals the share of time a CAV is under attack in the single-attacker
collection (Poisson starts at 0.2/s, mean duration 5 s). `repeats` independent realizations are drawn.

Targets (`--common-targets`, default on): a group only targets vehicles that every colluder detects, and places
ghosts within sensing range of every colluder, so that all colluders can carry out the shared plan.
Labels (`--label-rule`, default `content`): a CPM is malicious only if its content differs from the sender's own
detections; `schedule` marks every CPM sent in an active interval as malicious.
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from gencpm.attacks.coordinated import Episode, _seed, generate

NAMES = {"CG": "ColludingGhostAttack", "CSUP": "CoordinatedSuppressionAttack",
         "CCSD": "CoordinatedDriftAttack", "CISP": "CoordinatedISPAttack"}
G = {}


def _changed(ids, boxes, nid, nbox):
    if len(ids) != len(nid) or set(np.asarray(ids).tolist()) != set(np.asarray(nid).tolist()):
        return True
    a, b = np.asarray(boxes)[np.argsort(ids)], np.asarray(nbox)[np.argsort(nid)]
    return bool(np.abs(a - b).max() > 1e-9) if len(a) else False


def episode_rows(name):
    a, fm, fa = G["args"], G["fm"], G["fa"]
    ep = Episode(fm, fa, name)
    groups = ep.form_groups(a.group_size)
    iv = int(round(a.interval_s / 0.05))
    rows = []
    for rep in range(a.repeats):
        run = f"r{rep:03d}"
        seed_r = f"{a.seed}-{run}"
        gens, choice = {}, {}
        for gi, grp in enumerate(groups):
            gseed = f"{seed_r}-g{gi}"
            gens[gi] = {t: generate(ep, t, {}, grp, gseed, interval_s=a.interval_s, duty=1.0, common=a.common_targets)
                        for t in NAMES} \
                if len(grp) >= 2 else {}
            rng = random.Random(_seed(name, gseed, "mix"))
            choice[gi] = [rng.choice(list(NAMES)) if rng.random() < a.duty else None
                          for _ in range((ep.ticks[-1] - ep.ticks[0]) // iv + 1)]
        group_of = {c: gi for gi, grp in enumerate(groups) for c in grp}
        for c in ep.cavs:
            gi = group_of[c]
            for t in ep.ticks:
                if (c, t) not in ep.row:
                    continue
                typ = choice[gi][(t - ep.ticks[0]) // iv]
                ids, boxes, _ = ep.dets(c, t)
                label, atk = 0, ""
                if typ is not None and gens[gi] and (c, t) in gens[gi][typ]:
                    g = gens[gi][typ][(c, t)]
                    if a.label_rule == "schedule" or _changed(ids, boxes, g["ids"], g["boxes"]):
                        label, atk = 1, NAMES[typ]
                    ids, boxes = g["ids"], g["boxes"]
                pose = ep.pose(c, t)
                rows.append(({"episode": name, "town": name.split("/")[0], "run_id": run, "vehicle_id": c,
                              "timestamp": t, "label": label, "attack_type": atk, "group": gi,
                              "ego_x": float(pose[0]), "ego_y": float(pose[1]), "n_obj": len(ids), "n_box": len(ids)},
                             boxes[:, [0, 1, 10]].astype(np.float32) if len(boxes) else np.zeros((0, 3), np.float32),
                             np.asarray(ids, np.int64), np.asarray(boxes, np.float64).reshape(-1, 12)))
    return name, groups, rows


def init(args, fw):
    G["args"] = args
    G["fm"] = pd.read_pickle(fw / "frames_meta.pkl")
    G["fa"] = dict(np.load(fw / "frames_arrays.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--duty", type=float, default=1 - np.exp(-0.2 * 5.0))
    ap.add_argument("--interval-s", type=float, default=5.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--label-rule", choices=["schedule", "content"], default="content")
    ap.add_argument("--common-targets", action=argparse.BooleanOptionalAction, default=True,
                    help="colluders target only vehicles that every group member detects (ghosts within range of all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    fw, out = Path(a.frames_work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("frames_meta.pkl", "frames_arrays.npz"):
        if not (out / f).exists():
            os.symlink(os.path.relpath(fw / f, out), out / f)
    names = sorted(pd.read_pickle(fw / "frames_meta.pkl")["episode"].unique())
    import multiprocessing as mp
    from tqdm import tqdm
    rows, summary = [], {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=init, initargs=(a, fw)) as ex:
        for name, groups, r in tqdm(ex.map(episode_rows, names), total=len(names), desc="episodes"):
            rows += r
            summary[name] = {"groups": groups}
    meta = pd.DataFrame([r[0] for r in rows])
    order = meta.sort_values(["episode", "run_id", "vehicle_id", "timestamp"]).index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)

    def ragged(k, width, dt):
        parts = [rows[i][k] for i in order]
        off = np.zeros(len(parts) + 1, np.int64)
        off[1:] = np.cumsum([len(p) for p in parts])
        flat = np.concatenate(parts) if parts else np.zeros((0, width) if width else (0,), dt)
        return flat.astype(dt), off

    o, oo = ragged(1, 3, np.float32)
    i_, io = ragged(2, 0, np.int64)
    b, _ = ragged(3, 12, np.float64)
    meta.to_pickle(out / "cpm_meta.pkl")
    np.savez(out / "cpm_objects.npz", objects=o, offsets=oo)
    np.savez(out / "cpm_boxes.npz", ids=i_, boxes=b, offsets=io)
    s = {"num_cpms": len(meta), "label_counts": meta["label"].value_counts().to_dict(),
         "attack_counts": meta["attack_type"].value_counts().to_dict(), "group_size": a.group_size,
         "duty": a.duty, "interval_s": a.interval_s, "repeats": a.repeats, "label_rule": a.label_rule,
         "common_targets": a.common_targets,
         "seed": a.seed, "episodes": summary}
    (out / "cache_summary.json").write_text(json.dumps(s, indent=1, default=int))
    print(json.dumps({k: v for k, v in s.items() if k != "episodes"}, indent=1, default=int))


if __name__ == "__main__":
    main()
