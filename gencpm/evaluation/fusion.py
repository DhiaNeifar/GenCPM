"""
End-to-end late-fusion evaluation on AdverCPM episodes (benign / attack / defense / oracle).

Inputs are the caches built by `python -m gencpm frames|cache|train|fit` in a work directory:
    <work>/frames_meta.pkl, frames_arrays.npz   raw perception per (episode, CAV, timestamp):
                                                lidar pose, GT boxes, benign PointPillars boxes + scores
    <work>/cpm_meta.pkl, cpm_boxes.npz          transmitted CPM content per (episode, run, CAV, timestamp)
    <work>/temporal_W*/<model>_fold*_seed*_probs.npz   out-of-fold detector probabilities

For every (episode, run, timestamp, ego CAV):
    participants = CAVs within COM_RANGE of the ego (OpenCOOD late-fusion rule), ego included.
    The ego always fuses its own benign detections; every other participant contributes its CPM.
    GT = union of all participants' vehicles (de-duplicated by id), as in OpenCOOD late fusion.

Conditions (others' contribution):
    ego_only        no cooperation (lower bound)
    benign          benign detections of every participant (no attack anywhere)
    attack          transmitted CPMs as recorded in AdverCPM (malicious ones manipulated)
    oracle          transmitted CPMs, malicious ones dropped (upper bound of drop-gating)
    def:<M>:s<S>    transmitted CPMs, dropped if detector M (seed S) flags the sender's window
    bdef:<M>:s<S>   benign detections, dropped if detector M flags the *benign* window (FP cost)

Detector thresholds are selected on each fold's validation episodes (never on the evaluated
episode): rule `maxF1` or `FPR<=5%`. The detector probabilities are out-of-fold, so every episode
is evaluated with a model that never saw it.

Scores: transmitted objects that exist in the sender's own detections keep their detector score.
Fabricated objects (CF) receive U(0.8, 0.99), seeded per CPM, matching the attacker model used in
the original OpenCPM experiment.

Fusion runs in the C++ core (gencpm/evaluation/csrc/fusion_core.cpp) with the conventions of OpenCOOD late fusion:
box projection to the ego frame, removal of boxes longer or wider than 6 m or outside the height window, rotated NMS,
evaluation-range mask, and greedy TP/FP matching at IoU 0.3/0.5/0.7.

Output: one npz per episode with per-detection (sample, condition, score, tp@.3/.5/.7) and per-sample
(GT count, subset flags). AP is computed by `report` (global score sort = standard VOC AP; OpenCOOD's
order-dependent default is reported as a secondary number).
"""

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from gencpm.evaluation.geometry import COMM_RANGE_M, EVAL_RANGE_XY, boxes_in_range, pose_to_matrix

ATTACK_SHORT = {"AddObjectAttack": "CF", "RemoveObjectAttack": "CS", "DriftAttack": "CSD",
                "WhiteNoiseAttack": "ISP", "GhostTrackAttack": "GT", "StealthDriftAttack": "SD",
                "ReplayAttack": "RP", "EvasiveGhostAttack": "EG", "PoseFalsificationAttack": "PF",
                "ColludingGhostAttack": "CG", "CoordinatedSuppressionAttack": "CSUP",
                "CoordinatedDriftAttack": "CCSD", "CoordinatedISPAttack": "CISP"}
IOUS = (0.3, 0.5, 0.7)

# Globals populated in worker processes (fork) to avoid pickling large arrays per task.
G = {}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def select_thresholds(y_val, p_val):
    from sklearn.metrics import precision_recall_curve, roc_curve
    pr, rc, th = precision_recall_curve(y_val, p_val)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    out = {"maxF1": float(th[np.argmax(f1)])}
    fpr, tpr, th2 = roc_curve(y_val, p_val)
    ok = np.where(fpr <= 0.05)[0]
    out["FPR<=5%"] = float(th2[ok[np.argmax(tpr[ok])]]) if len(ok) else float("inf")
    return out


def fake_scores(key: str, n: int) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).uniform(0.8, 0.99, size=n).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# setup: load caches and build gating tables
# ----------------------------------------------------------------------------------------------

def load_inputs(args):
    wd = Path(args.work_dir)
    fm = pd.read_pickle(wd / "frames_meta.pkl")
    fa = dict(np.load(wd / "frames_arrays.npz"))
    cm = pd.read_pickle(wd / "cpm_meta.pkl")
    cb = dict(np.load(wd / "cpm_boxes.npz"))

    # Gate tables: for every (detector, seed) -> prob per CPM row (attacked stream) and per CPM row
    # computed on the benign version of the window, plus the validation-selected threshold.
    gates = {}
    labels = cm["label"].to_numpy()
    for tdir_name, model, seed in [(d, m, s) for d in args.temporal_dir for m in args.detectors for s in args.seeds]:
        tdir = wd / tdir_name
        if not sorted(tdir.glob(f"{model}_fold*_seed{seed}_probs.npz")):
            continue
        ends = np.load(tdir / "window_ends.npy")
        p_att = np.full(len(cm), np.nan, np.float32)
        p_ben = np.full(len(cm), np.nan, np.float32)
        thr_row = np.full(len(cm), np.nan, np.float32)
        files = sorted(tdir.glob(f"{model}_fold*_seed{seed}_probs.npz"))
        for f in files:
            z = np.load(f)
            thr = select_thresholds(labels[ends[z["val_idx"]]], z["p_val"])[args.threshold_rule]
            rows = ends[z["test_idx"]]
            p_att[rows] = z["p_test"]
            if "p_test_benign" in z:
                p_ben[rows] = z["p_test_benign"]
            thr_row[rows] = thr
        gates[f"{model}:s{seed}"] = (p_att, p_ben, thr_row)
    if args.detectors != ["none"]:
        found = {k.split(":s")[0] for k in gates}
        missing = [m for m in args.detectors if m not in found]
        if missing:  # never skip a requested detector silently
            raise SystemExit(f"[run] no predictions found for detectors {missing} in {args.temporal_dir}")
    # Object-level gates: per transmitted object a score (aligned with cpm_boxes), per benign-version
    # object a score (aligned with the defense features' benign offsets), and a per-CPM threshold.
    ogates = {}
    if args.object_gates:
        dz = np.load(Path(args.defense_features))
        b_off = dz["b_offsets"]
        lab = dz["obj_label"].astype(np.float32)
        ogates["oracle-object"] = (lab, np.zeros(len(dz["b_obj_row"]), np.float32),
                                   np.full(len(cm), 0.5, np.float32), b_off)
        for path in args.object_gates:
            g = dict(np.load(path))
            name = Path(path).stem.replace("objgate_", "").replace("_seed", ":s")
            ogates[name] = (g["score"], g["score_benign"], g["thr_row"], b_off)
    return fm, fa, cm, cb, gates, ogates


def init_worker(args_dict):
    import torch
    torch.set_num_threads(1)  # one process per core; avoid thread oversubscription
    from gencpm.evaluation import fusion_core
    G.update({"COM_RANGE": args_dict.get("comm_range") or COMM_RANGE_M, "order": args_dict.get("box_order") or "hwl",
              "nms": args_dict.get("nms_thresh") or 0.15, "fc": fusion_core,
              "range_xy": tuple(args_dict.get("range_xy") or EVAL_RANGE_XY)})
    rx, ry = G["range_xy"]
    z0, z1 = args_dict.get("z_window") or (-3.0, 1.0)
    fusion_core.set_filters(rx, ry, z0, z1, args_dict.get("max_extent") or 6.0)


# ----------------------------------------------------------------------------------------------
# per-episode evaluation
# ----------------------------------------------------------------------------------------------

def project(ids, boxes, ego_pose):
    """World boxes -> (N, 8, 3) corners in the ego frame."""
    if len(ids) == 0:
        return np.zeros((0, 8, 3), np.float32)
    world_to_ego = np.linalg.inv(pose_to_matrix(list(ego_pose)))
    return G["fc"].project(np.ascontiguousarray(boxes, np.float64), world_to_ego, G["order"] == "hwl")


def fuse_boxes(sources):
    """Late fusion: concatenate, drop large / abnormal-height boxes, rotated NMS, range mask.
    Returns (corners (N,8,3) numpy, scores (N,)) or (None, None) when nothing survives."""
    corners = [c for c, s in sources if len(c)]
    if not corners:
        return None, None
    box, score = G["fc"].fuse_boxes(np.concatenate(corners), np.concatenate([s for c, s in sources if len(c)]), G["nms"])
    return (box, score) if len(box) else (None, None)


def fuse_and_score(sources, gt_corners):
    """sources: list of (corners (N,8,3), scores (N,)). Returns (scores_sorted, tp(3, n))."""
    corners = [c for c, s in sources if len(c)]
    gt = gt_corners.numpy() if gt_corners is not None else np.zeros((0, 8, 3), np.float32)
    if not corners:
        return np.zeros(0, np.float32), np.zeros((len(IOUS), 0), np.int8)
    return G["fc"].fuse_and_score(np.concatenate(corners), np.concatenate([s for c, s in sources if len(c)]),
                                  gt, G["nms"], np.array(IOUS, np.float64))


# ----------------------------------------------------------------------------------------------
# ROBOSAC-style consensus (Li et al., ICCV 2023), adapted to late fusion
# ----------------------------------------------------------------------------------------------
ROBO_R = 50.0        # agreement is measured within this radius of the ego (its own sensing region)
ROBO_IOU = 0.3       # BEV IoU for two boxes to agree
ROBO_EXHAUSTIVE = 5  # exhaustive subset search up to this many collaborators, random sampling beyond
ROBO_SAMPLES = 32    # sampled subsets per size when sampling


def _near(corners):
    if corners is None or not len(corners):
        return np.zeros((0, 8, 3), np.float32)
    c = corners[:, :4, :2].mean(1)
    return corners[np.hypot(c[:, 0], c[:, 1]) <= ROBO_R]


def agreement(ego_c, fused_c):
    """Jaccard index between the ego's own boxes and the fused boxes within ROBO_R (greedy BEV matching)."""
    iou_matrix = G["fc"].iou_matrix
    a, b = _near(ego_c), _near(fused_c)
    if len(a) + len(b) == 0:
        return 1.0
    m = 0
    if len(a) and len(b):
        iou = np.nan_to_num(iou_matrix(a, b))
        for i in np.argsort(-iou.max(1)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= ROBO_IOU:
                m += 1
                iou[:, j] = -1
    return m / (len(a) + len(b) - m)


def robosac_select(ego_src, srcs, tau, rng_key):
    """Largest subset of collaborators whose fusion with the ego agrees with the ego's own result
    (agreement >= tau); ties broken by agreement. srcs: list of (corners, scores) per collaborator.
    Returns a tuple of booleans (kept) aligned with srcs."""
    from itertools import combinations
    n = len(srcs)
    if n == 0 or not np.isfinite(tau):
        return tuple(True for _ in srcs)
    ego_c, _ = fuse_boxes([ego_src])
    rng = np.random.default_rng(int(hashlib.sha256(rng_key.encode()).hexdigest()[:16], 16))
    for k in range(n, 0, -1):
        subsets = list(combinations(range(n), k))
        budget = G.get("robo_samples", ROBO_SAMPLES)
        if n > ROBO_EXHAUSTIVE and len(subsets) > budget:
            subsets = [subsets[i] for i in rng.choice(len(subsets), budget, replace=False)]
        best, best_j = None, -1.0
        for sub in subsets:
            fc, _ = fuse_boxes([ego_src] + [srcs[i] for i in sub])
            j = agreement(ego_c, fc)
            if j >= tau and j > best_j:
                best, best_j = sub, j
        if best is not None:
            return tuple(i in best for i in range(n))
    return tuple(False for _ in srcs)


def calib_episode(task):
    """Agreement of the full benign collaborator set with the ego, per (timestamp, ego) with >= 1 other."""
    episode, out_path = task
    fm, fa = G["fm"], G["fa"]
    f_rows = np.where(fm["episode"].to_numpy() == episode)[0]
    frames = {(fm.at[r, "vehicle_id"], int(fm.at[r, "timestamp"])): r for r in f_rows}
    cavs = sorted({k[0] for k in frames})
    js = []
    for ts in sorted({k[1] for k in frames})[::G.get("ts_stride", 1)]:
        present = [c for c in cavs if (c, ts) in frames]
        for ego in present:
            ego_pose = fa["poses"][frames[(ego, ts)]]
            others = [c for c in present if c != ego and math.hypot(
                fa["poses"][frames[(c, ts)]][0] - ego_pose[0], fa["poses"][frames[(c, ts)]][1] - ego_pose[1]) <= G["COM_RANGE"]]
            if not others:
                continue
            srcs = []
            for c in [ego] + others:
                r = frames[(c, ts)]
                a, b = fa["pred_offsets"][r], fa["pred_offsets"][r + 1]
                srcs.append((project(fa["pred_ids"][a:b], fa["pred_boxes"][a:b], ego_pose), fa["pred_scores"][a:b]))
            ego_c, _ = fuse_boxes(srcs[:1])
            fc, _ = fuse_boxes(srcs)
            js.append(agreement(ego_c, fc))
    np.save(out_path, np.array(js, np.float32))
    return episode, len(js)


def eval_episode(task):
    import torch
    episode, run_ids, cond_names, out_path = task[:4]
    chunk = task[4] if len(task) > 4 else (0, 1)  # (k, K): evaluate the k-th of K contiguous time chunks
    fm, fa, cm, cb, gates = G["fm"], G["fa"], G["cm"], G["cb"], G["gates"]
    ogates = G["ogates"]

    f_rows = np.where(fm["episode"].to_numpy() == episode)[0]
    frames = {(fm.at[r, "vehicle_id"], int(fm.at[r, "timestamp"])): r for r in f_rows}
    c_rows = np.where(cm["episode"].to_numpy() == episode)[0]
    cpm_at = {(cm.at[r, "run_id"], cm.at[r, "vehicle_id"], int(cm.at[r, "timestamp"])): r for r in c_rows}
    cavs = sorted({k[0] for k in frames})
    stamps = sorted({k[1] for k in frames})
    stamps = stamps[::G.get("ts_stride", 1)]
    stamps = [int(t) for t in np.array_split(np.array(stamps), chunk[1])[chunk[0]]]

    def benign_of(r):
        a, b = fa["pred_offsets"][r], fa["pred_offsets"][r + 1]
        return fa["pred_ids"][a:b], fa["pred_boxes"][a:b], fa["pred_scores"][a:b]

    def transmitted_of(crow, frow, run, cav, ts):
        a, b = cb["offsets"][crow], cb["offsets"][crow + 1]
        ids, boxes = cb["ids"][a:b], cb["boxes"][a:b]
        bid, _, bsc = benign_of(frow)
        lookup = dict(zip(bid.tolist(), bsc.tolist()))
        sc = np.array([lookup.get(int(i), np.nan) for i in ids], np.float32)
        missing = np.isnan(sc)
        if missing.any():
            sc[missing] = fake_scores(f"{episode}|{run}|{cav}|{ts}", int(missing.sum()))
        return ids, boxes, sc

    det = defaultdict(list)       # per detection: sample, cond, score, tp30, tp50, tp70
    samples = []                  # per sample metadata
    for run in run_ids:
        for ts in stamps:
            present = [c for c in cavs if (c, ts) in frames]
            for ego in present:
                ego_row = frames[(ego, ts)]
                ego_pose = fa["poses"][ego_row]
                parts = [c for c in present
                         if math.hypot(fa["poses"][frames[(c, ts)]][0] - ego_pose[0],
                                       fa["poses"][frames[(c, ts)]][1] - ego_pose[1]) <= G["COM_RANGE"]]
                others = [c for c in parts if c != ego]
                # GT: union over participants, dedup by id
                gt = OrderedDict()
                for c in parts:
                    r = frames[(c, ts)]
                    a, b = fa["gt_offsets"][r], fa["gt_offsets"][r + 1]
                    for i, bx in zip(fa["gt_ids"][a:b], fa["gt_boxes"][a:b]):
                        gt.setdefault(int(i), bx)
                gtc = project(np.array(list(gt.keys())), np.array(list(gt.values())), ego_pose)
                gt_t = torch.from_numpy(gtc).float()
                if len(gtc):
                    gt_t = gt_t[boxes_in_range(gt_t, G["range_xy"])]
                n_gt = int(gt_t.shape[0])

                bid, bbox, bsc = benign_of(ego_row)
                ego_src = (project(bid, bbox, ego_pose), bsc)
                ben_src, tx_src, mal, fam = {}, {}, {}, {}
                for c in others:
                    fr = frames[(c, ts)]
                    i, bx, sc = benign_of(fr)
                    ben_src[c] = (project(i, bx, ego_pose), sc)
                    crow = cpm_at.get((run, c, ts))
                    if crow is None:  # no CPM logged (should not happen) -> treat as silent
                        tx_src[c], mal[c], fam[c] = (np.zeros((0, 8, 3), np.float32), np.zeros(0, np.float32)), 0, ""
                        continue
                    i, bx, sc = transmitted_of(crow, fr, run, c, ts)
                    tx_src[c] = (project(i, bx, ego_pose), sc)
                    mal[c] = int(cm.at[crow, "label"])
                    fam[c] = cm.at[crow, "attack_type"]

                # source selection per condition: tuple of 'b'/'t'/'-' per other participant
                choice = {"ego_only": tuple("-" for _ in others),
                          "benign": tuple("b" for _ in others),
                          "attack": tuple("t" for _ in others),
                          "oracle": tuple("-" if mal[c] else "t" for c in others)}
                for gname, (p_att, p_ben, thr) in gates.items():
                    sel_d, sel_b = [], []
                    for c in others:
                        crow = cpm_at.get((run, c, ts))
                        pa = p_att[crow] if crow is not None else np.nan
                        pb = p_ben[crow] if crow is not None else np.nan
                        th = thr[crow] if crow is not None else np.nan
                        # warm-up (no full window yet) -> include, as in the original experiment
                        sel_d.append("-" if (not np.isnan(pa) and pa >= th) else "t")
                        sel_b.append("-" if (not np.isnan(pb) and pb >= th) else "b")
                    choice[f"def:{gname}"] = tuple(sel_d)
                    choice[f"bdef:{gname}"] = tuple(sel_b)
                if "robosac" in cond_names:
                    tau = G["robo_tau"].get(episode, np.nan)
                    key = f"{episode}|{run}|{ts}|{ego}"
                    kt = robosac_select(ego_src, [tx_src[c] for c in others], tau, key)
                    kb = robosac_select(ego_src, [ben_src[c] for c in others], tau, key)
                    choice["robosac"] = tuple("t" if k else "-" for k in kt)
                    choice["brobosac"] = tuple("b" if k else "-" for k in kb)
                for gname in ogates:
                    choice[f"ofilt:{gname}"] = tuple(f"o|{gname}" for _ in others)
                    choice[f"bofilt:{gname}"] = tuple(f"p|{gname}" for _ in others)

                def source(c, s):
                    if s == "b":
                        return ben_src[c]
                    if s == "t":
                        return tx_src[c]
                    kind, gname = s.split("|", 1)
                    score, score_b, thr, b_off = ogates[gname]
                    crow = cpm_at.get((run, c, ts))
                    corners, sc = tx_src[c] if kind == "o" else ben_src[c]
                    if crow is None or np.isnan(thr[crow]) or len(sc) == 0:
                        return corners, sc
                    if kind == "o":
                        osc = score[cb["offsets"][crow]:cb["offsets"][crow + 1]]
                    else:
                        osc = score_b[b_off[crow]:b_off[crow + 1]]
                    keep = ~(osc >= thr[crow])  # NaN score (no evidence) -> keep
                    return corners[keep], sc[keep]

                sid = len(samples)
                fams = sorted({fam[c] for c in others if mal[c]})
                samples.append({"run": run, "ts": ts, "ego": ego, "n_gt": n_gt, "n_others": len(others),
                                "n_mal": int(sum(mal.values())), "families": ",".join(fams)})
                memo = {}
                for ci, cname in enumerate(cond_names):
                    sel = choice[cname]
                    if sel not in memo:
                        srcs = [ego_src] + [source(c, s) for c, s in zip(others, sel) if s != "-"]
                        memo[sel] = fuse_and_score(srcs, gt_t)
                    sc, tp = memo[sel]
                    n = len(sc)
                    if n:
                        det["sample"].append(np.full(n, sid, np.int32))
                        det["cond"].append(np.full(n, ci, np.int16))
                        det["score"].append(sc)
                        det["tp"].append(tp.T)
    s = pd.DataFrame(samples, columns=["run", "ts", "ego", "n_gt", "n_others", "n_mal", "families"])
    np.savez_compressed(
        out_path,
        sample=np.concatenate(det["sample"]) if det["sample"] else np.zeros(0, np.int32),
        cond=np.concatenate(det["cond"]) if det["cond"] else np.zeros(0, np.int16),
        score=np.concatenate(det["score"]) if det["score"] else np.zeros(0, np.float32),
        tp=np.concatenate(det["tp"]) if det["tp"] else np.zeros((0, 3), np.int8),
        s_run=s["run"].to_numpy().astype(str), s_ts=s["ts"].to_numpy(), s_ego=s["ego"].to_numpy().astype(str),
        s_ngt=s["n_gt"].to_numpy(), s_nothers=s["n_others"].to_numpy(), s_nmal=s["n_mal"].to_numpy(),
        s_fam=s["families"].to_numpy().astype(str), cond_names=np.array(cond_names))
    return episode, len(samples)


def cmd_run(args):
    fm, fa, cm, cb, gates, ogates = load_inputs(args)
    G.update({"fm": fm, "fa": fa, "cm": cm, "cb": cb, "gates": gates, "ogates": ogates})
    conds = ["ego_only", "benign", "attack", "oracle"] + [f"{p}:{g}" for g in gates for p in ("def", "bdef")] \
        + [f"{p}:{g}" for g in ogates for p in ("ofilt", "bofilt")]
    robo_tau = {}
    if args.robosac_calib:
        robo_tau = robosac_thresholds(cm, Path(args.robosac_calib), args)
        conds += ["robosac", "brobosac"]
    G["robo_tau"] = robo_tau
    G["ts_stride"] = args.ts_stride
    G["robo_samples"] = args.robosac_samples
    episodes = sorted(fm["episode"].unique())
    if args.episodes:
        episodes = [e for e in episodes if e in set(args.episodes)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"conditions": conds}, indent=2))
    todo = [e for e in episodes if args.overwrite or not (out / (e.replace("/", "__") + ".npz")).exists()]
    parts_dir = out / "_parts"
    parts_dir.mkdir(exist_ok=True)
    # One task per (episode, run, time chunk): small tasks balance the load across workers.
    tasks = []
    def part_ok(path):  # a finished part from an earlier, interrupted run (same conditions) is kept
        try:
            z = np.load(path)
            return list(z["cond_names"]) == list(conds)
        except Exception:
            return False
    for e in todo:
        for run in args.runs:
            for k in range(args.chunks):
                part = parts_dir / f"{e.replace('/', '__')}__{run}__{k:03d}.npz"
                if part.exists() and part_ok(part):
                    continue
                tasks.append((e, [run], conds, str(part), (k, args.chunks)))
    tasks.sort(key=lambda t: -int((fm["episode"] == t[0]).sum()))  # largest episodes first
    print(f"[run] {len(todo)} episodes x runs {args.runs} x {args.chunks} chunks = {len(tasks)} tasks, "
          f"{len(conds)} conditions, {args.workers} workers")
    from tqdm import tqdm
    import multiprocessing as mp
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork"), initializer=init_worker,
                             initargs=(vars(args),)) as ex:
        for _ in tqdm(ex.map(eval_episode, tasks), total=len(tasks), desc="tasks", unit="task", mininterval=30):
            pass
    for e in todo:  # merge parts into the per-episode file, in the original (run, time) order
        merge_parts([parts_dir / f"{e.replace('/', '__')}__{run}__{k:03d}.npz" for run in args.runs
                     for k in range(args.chunks)], out / (e.replace("/", "__") + ".npz"))
    print(f"[run] merged {len(todo)} episodes into {out}")


def merge_parts(paths, out_path):
    zs = [dict(np.load(p, allow_pickle=False)) for p in paths]
    off, sample = 0, []
    for z in zs:
        sample.append(z["sample"] + off)
        off += len(z["s_run"])
    cat = lambda k: np.concatenate([z[k] for z in zs])
    np.savez_compressed(out_path, sample=np.concatenate(sample).astype(np.int32), cond=cat("cond"), score=cat("score"),
                        tp=cat("tp"), s_run=cat("s_run"), s_ts=cat("s_ts"), s_ego=cat("s_ego"), s_ngt=cat("s_ngt"),
                        s_nothers=cat("s_nothers"), s_nmal=cat("s_nmal"), s_fam=cat("s_fam"), cond_names=zs[0]["cond_names"])
    for p in paths:
        Path(p).unlink()


def robosac_thresholds(cm, calib_dir, args):
    """Per episode: tau = ROBO_FPR quantile of benign full-set agreement over the training folds of
    the episode's test fold (same town-stratified folds as the detectors)."""
    from gencpm.data.cache import fold_roles, make_folds
    fold_of = make_folds(cm, args.folds, args.split_seed)
    J = {e: np.load(calib_dir / (e.replace("/", "__") + ".npy")) for e in fold_of}
    taus, out = {}, {}
    for k in range(args.folds):
        test_f, val_f = fold_roles(k, args.folds)
        tr = np.concatenate([J[e] for e, f in fold_of.items() if f not in (test_f, val_f)])
        taus[k] = float(np.quantile(tr, args.robosac_fpr))
    for e, f in fold_of.items():
        out[e] = taus[f]
    print("[robosac] tau per test fold:", {k: round(v, 3) for k, v in taus.items()})
    return out


def cmd_robosac_calib(args):
    fm = pd.read_pickle(Path(args.work_dir) / "frames_meta.pkl")
    fa = dict(np.load(Path(args.work_dir) / "frames_arrays.npz"))
    G.update({"fm": fm, "fa": fa, "ts_stride": args.ts_stride})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(e, str(out / (e.replace("/", "__") + ".npy"))) for e in sorted(fm["episode"].unique())
             if not (out / (e.replace("/", "__") + ".npy")).exists()]
    from tqdm import tqdm
    import multiprocessing as mp
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork"), initializer=init_worker,
                             initargs=(vars(args),)) as ex:
        for ep, n in tqdm(ex.map(calib_episode, tasks), total=len(tasks), desc="robosac calib", unit="ep"):
            pass
    print("[robosac-calib] done:", out)


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------

def voc_ap(tp, score, n_gt, global_sort=True):
    if n_gt == 0:
        return np.nan
    if global_sort:
        o = np.argsort(-score, kind="stable")
        tp = tp[o]
    tpc = np.cumsum(tp)
    fpc = np.cumsum(1 - tp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1)
    # OpenCOOD voc_ap (all-point interpolation)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # == OpenCOOD's backward max loop
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def _report_ci_worker(key):
    return G["report_ci_for"](key)


def _boot_ap_ci(tp, ep, ngt, Wb, kblock=200_000, bblock=250):
    """Episode-bootstrap 95% CI of AP@0.5 from detections sorted by descending score (tp flags, episode index),
    exact and computed only at true-positive ranks (see cmd_report_legacy.ci_for)."""
    pos = np.flatnonzero(tp)
    if len(pos) == 0:
        return (np.nan, np.nan)
    ep_pos = ep[pos]
    n_ep = len(ngt)
    T = [np.flatnonzero((ep == e) & (tp == 1)) for e in range(n_ep)]
    F = [np.flatnonzero((ep == e) & (tp == 0)) for e in range(n_ep)]
    out = []
    for b0 in range(0, len(Wb), bblock):
        W = Wb[b0:b0 + bblock]
        n = W @ ngt
        env_next = np.zeros(len(W))
        ap = np.zeros(len(W))
        for k1 in range(len(pos), 0, -kblock):
            k0 = max(0, k1 - kblock)
            pk = pos[k0:k1]
            tcnt = np.stack([np.searchsorted(t, pk, "right") for t in T]).astype(float)
            fcnt = np.stack([np.searchsorted(f, pk, "right") for f in F]).astype(float)
            tpc, fpc = W @ tcnt, W @ fcnt
            prec = tpc / np.maximum(tpc + fpc, 1)
            env = np.maximum.accumulate(np.concatenate([env_next[:, None], prec[:, ::-1]], axis=1), axis=1)
            env = env[:, 1:][:, ::-1]
            env_next = env[:, 0]
            ap += np.sum(W[:, ep_pos[k0:k1]] * env, axis=1)
        ap = 100 * ap / np.maximum(n, 1e-12)
        ap[n == 0] = np.nan
        out.append(ap)
    return tuple(np.nanpercentile(np.concatenate(out), [2.5, 97.5]))


def _boot_ap_ci_torch(tp, ep, ngt, Wb, kblock=1_000_000, bblock=250, dev="cuda"):
    """GPU version of _boot_ap_ci (same computation in float64; CIs agree to about 1e-14)."""
    import torch
    pos = np.flatnonzero(tp)
    if len(pos) == 0:
        return (np.nan, np.nan)
    n_ep = len(ngt)
    T = [torch.from_numpy(np.flatnonzero((ep == e) & (tp == 1))).to(dev) for e in range(n_ep)]
    F = [torch.from_numpy(np.flatnonzero((ep == e) & (tp == 0))).to(dev) for e in range(n_ep)]
    pos_t = torch.from_numpy(pos).to(dev); ep_pos = torch.from_numpy(ep[pos].astype(np.int64)).to(dev)
    ngt_t = torch.from_numpy(np.asarray(ngt, np.float64)).to(dev)
    out = []
    for b0 in range(0, len(Wb), bblock):
        W = torch.from_numpy(Wb[b0:b0 + bblock]).to(dev)
        n = W @ ngt_t
        env_next = torch.zeros(len(W), dtype=torch.float64, device=dev)
        ap = torch.zeros(len(W), dtype=torch.float64, device=dev)
        for k1 in range(len(pos), 0, -kblock):
            k0 = max(0, k1 - kblock)
            pk = pos_t[k0:k1]
            tcnt = torch.stack([torch.searchsorted(t, pk, right=True) for t in T]).double()
            fcnt = torch.stack([torch.searchsorted(f, pk, right=True) for f in F]).double()
            tpc, fpc = W @ tcnt, W @ fcnt
            prec = tpc / torch.clamp(tpc + fpc, min=1)
            env = torch.cat([env_next[:, None], prec.flip(1)], 1).cummax(1).values[:, 1:].flip(1)
            env_next = env[:, 0]
            ap += (W[:, ep_pos[k0:k1]] * env).sum(1)
            del tcnt, fcnt, tpc, fpc, prec, env
        ap = 100 * ap / torch.clamp(n, min=1e-12)
        ap[n == 0] = float("nan")
        out.append(ap.cpu().numpy())
    return tuple(np.nanpercentile(np.concatenate(out), [2.5, 97.5]))


def _report_cond(ci):
    """All subsets of one condition: AP@0.3/0.5/0.7 (global score order), AP@0.5 in OpenCOOD order, and the
    bootstrap CI for the bootstrap subsets. Reads only this condition's detections (split files). One stable
    sort per condition; a subset is a mask on the sorted detections, which keeps the order of tied scores of a
    stable sort of the subset alone."""
    R = G["report"]
    base = Path(R["split_dir"]) / f"c{ci:03d}"
    tp = np.fromfile(f"{base}_tp.bin", np.int8).reshape(-1, 3)
    sc = np.fromfile(f"{base}_score.bin", np.float32)
    ep = np.fromfile(f"{base}_ep.bin", np.int16).astype(np.int64)
    smp = np.fromfile(f"{base}_sample.bin", np.int32)
    gid = R["sample_off"][ep] + smp                    # global sample index of every detection
    del smp
    o = np.argsort(-sc, kind="stable")
    tp_s, sc_s, ep_s, gid_s = tp[o], sc[o], ep[o], gid[o]
    del o
    res = {}
    for name in R["subset_names"]:
        gm = R["sub_global"][name]                     # mask over all samples
        ms, mu = gm[gid_s], gm[gid]
        ngt_e = R["ngt_ep"][name]
        ngt = int(ngt_e.sum())
        a = tp_s[ms]
        aps = [100 * voc_ap(a[:, j], sc_s[ms], ngt, False) for j in range(3)]
        ap_oc = 100 * voc_ap(tp[mu][:, 1], sc[mu], ngt, False)
        row = {"AP30": aps[0], "AP50": aps[1], "AP70": aps[2], "AP50_opencood_order": ap_oc}
        if R["Wb"] is not None and name in R["boot_subsets"] and not R["conds"][ci].startswith(("bdef:", "bofilt:", "brobosac")):
            if R.get("gpu_lock") is not None:  # one worker at a time on the GPU
                with R["gpu_lock"]:
                    import torch
                    row["AP50_ci_low"], row["AP50_ci_high"] = _boot_ap_ci_torch(a[:, 1], ep_s[ms], ngt_e, R["Wb"])
                    torch.cuda.empty_cache()  # hand the GPU memory back before the next worker takes the lock
            else:
                row["AP50_ci_low"], row["AP50_ci_high"] = _boot_ap_ci(a[:, 1], ep_s[ms], ngt_e, R["Wb"])
        res[name] = row
    return ci, res


def cmd_report(args):
    """Report with bounded memory: the per-episode detections are split once by condition into flat files
    (one episode in memory at a time); workers then read one condition each."""
    import shutil
    from tqdm import tqdm
    out = Path(args.out_dir)
    files = sorted(out.glob("Town*.npz"))
    eps = [f.stem.replace("__", "/") for f in files]
    split_dir = Path(args.split_dir or (out / "_bycond"))
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    meta, conds, handles = [], None, None
    for e, f in enumerate(tqdm(files, desc="split by condition", unit="episode")):
        d = dict(np.load(f, allow_pickle=False))
        if conds is None:
            conds = list(d["cond_names"])
            handles = {(ci, k): open(split_dir / f"c{ci:03d}_{k}.bin", "wb")
                       for ci in range(len(conds)) for k in ("tp", "score", "ep", "sample")}
        meta.append({k: d[k] for k in ("s_fam", "s_nmal", "s_nothers", "s_ngt")})
        o = np.argsort(d["cond"], kind="stable")       # original order within a condition
        cuts = np.searchsorted(d["cond"][o], np.arange(len(conds) + 1))
        for ci in range(len(conds)):
            idx = o[cuts[ci]:cuts[ci + 1]]
            d["tp"][idx].astype(np.int8).tofile(handles[(ci, "tp")])
            d["score"][idx].astype(np.float32).tofile(handles[(ci, "score")])
            np.full(len(idx), e, np.int16).tofile(handles[(ci, "ep")])
            d["sample"][idx].astype(np.int32).tofile(handles[(ci, "sample")])
        del d
    for h in handles.values():
        h.close()

    def subsets(m):
        fam, nm = m["s_fam"], m["s_nmal"]
        sub = {"all": np.ones(len(nm), bool), "cooperative(>=1 other in range)": m["s_nothers"] > 0,
               "attacked(any)": nm > 0}
        for k, v in ATTACK_SHORT.items():
            sub[f"only {v}"] = (nm > 0) & (fam == k)
        sub["1 attacker"], sub["2 attackers"], sub[">=3 attackers"] = nm == 1, nm == 2, nm >= 3
        return sub

    present = set().union(*[set(m["s_fam"].tolist()) for m in meta])
    subset_names = (["all", "cooperative(>=1 other in range)", "attacked(any)"]
                    + [f"only {v}" for k, v in ATTACK_SHORT.items() if k in present]
                    + ["1 attacker", "2 attackers", ">=3 attackers"])
    subs_of = [subsets(m) for m in meta]
    sample_off = np.zeros(len(meta) + 1, np.int64)
    sample_off[1:] = np.cumsum([len(m["s_nmal"]) for m in meta])
    sub_global = {n: np.concatenate([subs_of[e][n] for e in range(len(meta))]) for n in subset_names}
    ngt_ep = {n: np.array([int(meta[e]["s_ngt"][subs_of[e][n]].sum()) for e in range(len(meta))], float)
              for n in subset_names}
    Wb = None
    if args.bootstrap:
        rng = np.random.default_rng(0)
        boots = [rng.integers(0, len(meta), len(meta)) for _ in range(args.bootstrap)]
        Wb = np.zeros((len(boots), len(meta)))
        for b, pick in enumerate(boots):
            np.add.at(Wb[b], pick, 1.0)
    G["report"] = {"split_dir": str(split_dir), "sample_off": sample_off, "sub_global": sub_global, "ngt_ep": ngt_ep,
                   "conds": conds, "subset_names": subset_names, "Wb": Wb, "boot_subsets": set(args.bootstrap_subsets)}
    gpu_lock = None
    if args.bootstrap and args.boot_device == "cuda":
        import multiprocessing as mp
        gpu_lock = mp.get_context("fork").Lock()  # created before the workers fork; CUDA is initialized in them only
    G["report"]["gpu_lock"] = gpu_lock
    results = {}
    workers = max(1, getattr(args, "report_workers", 1))
    bar = tqdm(total=len(conds), desc="conditions", unit="cond")
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
                for ci, res in ex.map(_report_cond, range(len(conds))):
                    results[ci] = res
                    bar.update()
        except Exception as exc:
            print(f"report workers failed ({type(exc).__name__}: {exc}), computing the remaining conditions serially", flush=True)
    for ci in range(len(conds)):
        if ci not in results:
            results[ci] = _report_cond(ci)[1]
            bar.update()
    bar.close()
    if not args.keep_split:
        shutil.rmtree(split_dir)
    rows = []
    for name in subset_names:
        n_samples = int(sub_global[name].sum())
        for ci, cond in enumerate(conds):
            rows.append({"subset": name, "condition": cond, "samples": n_samples, **results[ci][name]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "fusion_report.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print(f"\nepisodes={len(eps)}  wrote {out / 'fusion_report.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--work-dir", required=True, help="GenCPM benchmark_out directory")
    r.add_argument("--temporal-dir", nargs="+", default=["temporal_W20_global"],
                   help="one or more result dirs with <model>_fold*_seed*_probs.npz (CPM-level gates)")
    r.add_argument("--detectors", nargs="+", default=["LSTM", "GRU", "BiLSTM", "CNN-LSTM"])
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--object-gates", nargs="*", default=[], help="objgate_*.npz files (object-level filtering)")
    r.add_argument("--defense-features", default=None, help="defense_features.npz (benign-version offsets)")
    r.add_argument("--threshold-rule", choices=["maxF1", "FPR<=5%"], default="maxF1")
    r.add_argument("--runs", nargs="+", default=["r000", "r001"])
    r.add_argument("--episodes", nargs="*", default=None)
    r.add_argument("--box-order", choices=["hwl", "lwh"], default="hwl", help="box size order of the detector")
    r.add_argument("--nms-thresh", type=float, default=0.15, help="rotated-NMS IoU threshold of late fusion")
    r.add_argument("--comm-range", type=float, default=COMM_RANGE_M, help="communication range (m)")
    r.add_argument("--range-xy", type=float, nargs=2, default=[140.0, 40.0], metavar=("X", "Y"),
                   help="evaluation range |x|<=X, |y|<=Y in the ego frame (default 140 x 40 m, as in OpenCOOD)")
    r.add_argument("--z-window", type=float, nargs=2, default=[-3.0, 1.0], metavar=("ZMIN", "ZMAX"),
                   help="predicted boxes must lie within this height window in the ego frame (default -3..1 m)")
    r.add_argument("--max-extent", type=float, default=6.0, help="drop predicted boxes longer/wider than this (m)")
    r.add_argument("--out-dir", default="results/advercpm_fusion")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--chunks", type=int, default=4, help="time chunks per episode and run (load balancing only)")
    r.add_argument("--ts-stride", type=int, default=1, help="evaluate every k-th timestamp (1 = all, 10 Hz)")
    r.add_argument("--robosac-samples", type=int, default=ROBO_SAMPLES,
                   help="ROBOSAC: sampled subsets per size when there are more than ROBO_EXHAUSTIVE collaborators")
    r.add_argument("--robosac-calib", default=None, help="dir from robosac-calib; adds robosac/brobosac conditions")
    r.add_argument("--robosac-fpr", type=float, default=0.05, help="benign quantile for the agreement threshold")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--split-seed", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--out-dir", default="results/advercpm_fusion")
    p.add_argument("--bootstrap", type=int, default=0, help="episode bootstrap resamples for AP50 CI")
    p.add_argument("--report-workers", type=int, default=1, help="processes, one condition at a time (results identical)")
    p.add_argument("--split-dir", default="", help="where the per-condition files go (default <out-dir>/_bycond)")
    p.add_argument("--keep-split", action="store_true", help="keep the per-condition files")
    p.add_argument("--boot-device", choices=["cuda", "cpu"], default="cuda", help="bootstrap CIs on the GPU (default) or CPU")
    p.add_argument("--bootstrap-subsets", nargs="+", default=["attacked(any)"],
                   help="subsets for which the AP50 bootstrap CI is computed")
    c = sub.add_parser("robosac-calib", help="benign agreement per sample for the ROBOSAC threshold")
    c.add_argument("--work-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--box-order", choices=["hwl", "lwh"], default="hwl", help="box size order of the detector")
    c.add_argument("--nms-thresh", type=float, default=0.15, help="rotated-NMS IoU threshold of late fusion")
    c.add_argument("--comm-range", type=float, default=COMM_RANGE_M, help="communication range (m)")
    c.add_argument("--range-xy", type=float, nargs=2, default=[140.0, 40.0], metavar=("X", "Y"),
                   help="evaluation range |x|<=X, |y|<=Y in the ego frame (default 140 x 40 m, as in OpenCOOD)")
    c.add_argument("--z-window", type=float, nargs=2, default=[-3.0, 1.0], metavar=("ZMIN", "ZMAX"),
                   help="predicted boxes must lie within this height window in the ego frame (default -3..1 m)")
    c.add_argument("--max-extent", type=float, default=6.0, help="drop predicted boxes longer/wider than this (m)")
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    c.add_argument("--ts-stride", type=int, default=1)
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report, "robosac-calib": cmd_robosac_calib}[args.cmd](args)


if __name__ == "__main__":
    main()
