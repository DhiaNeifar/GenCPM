"""Caches of the released perception frames and of the generated CPMs, content labels, and the episode folds.

  frames  parse every <ts>_preds.yaml (+ _score.npy) once: detections, scores, ground truth, poses -> frames_meta.pkl, frames_arrays.npz
  cache   parse every CPM YAML once into a meta table and ragged object arrays, with content labels: a CPM is malicious
          only if what it transmits differs from what its sender perceived

Splitting rule: the unit is the episode; both attack realizations of an episode always fall in the same fold."""

from __future__ import annotations
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml


YAML_LOADER = getattr(yaml, "CUnsafeLoader", yaml.UnsafeLoader)


ATTACK_TO_PAPER = {
    "AddObjectAttack": "CF",
    "RemoveObjectAttack": "CS",
    "DriftAttack": "CSD",
    "WhiteNoiseAttack": "ISP",
}


BOX_FIELDS = ("location", "center", "extent", "angle")  # OpenCOOD object format, 3 floats each


def boxes_from_opencood_dict(objs: dict) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCOOD-style {id: {location, center, extent, angle}} -> (ids int64 (N,), boxes float64 (N, 12))."""
    ids = np.zeros(len(objs), dtype=np.int64)
    boxes = np.zeros((len(objs), 12), dtype=np.float64)
    for i, (k, o) in enumerate(objs.items()):
        ids[i] = int(k)
        boxes[i] = [float(v) for f in BOX_FIELDS for v in o[f]]
    return ids, boxes


def _parse_cpm(path: str) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        cpm = yaml.load(f, Loader=YAML_LOADER)
    p = Path(path)
    stem = p.stem.split("_")
    run_id = stem[0] if stem[0].startswith("r") and stem[0][1:].isdigit() else "r000"
    attacks = [a for a in (cpm.get("attacks") or []) if a]
    attack_types = sorted({a.get("attack_type") for a in attacks})
    gp = cpm.get("global_position") or {}
    objs = cpm.get("detected_vehicles") or []
    arr = np.zeros((len(objs), 3), dtype=np.float32)
    for i, v in enumerate(objs):
        loc = v.get("location") or {}
        ori = v.get("orientation") or {}
        arr[i] = (loc.get("x", 0.0), loc.get("y", 0.0), ori.get("yaw", 0.0))
    # Full transmitted boxes (what a receiver fuses). Same order as detected_vehicles.
    ids, boxes = boxes_from_opencood_dict(cpm.get("detected_objects") or {})
    meta = {
        "episode": p.parent.relative_to(p.parent.parent.parent).as_posix(),  # Town/scenario
        "town": p.parent.parent.name,
        "run_id": run_id,
        "vehicle_id": str(cpm.get("vehicle_id")),
        "timestamp": int(cpm.get("timestamp")),
        "label": int(cpm.get("cpm_type") == "malicious"),
        "attack_type": ",".join(attack_types) if attack_types else "",
        "ego_x": float(gp.get("x", 0.0)),
        "ego_y": float(gp.get("y", 0.0)),
        "n_obj": len(objs),
        "n_box": len(ids),
    }
    return meta, arr, ids, boxes


def _ragged(parts: List[np.ndarray], order: np.ndarray, width: int, dtype) -> Tuple[np.ndarray, np.ndarray]:
    parts = [parts[i] for i in order]
    offsets = np.zeros(len(parts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(a) for a in parts])
    shape = (0, width) if width else (0,)
    flat = np.concatenate(parts, axis=0) if parts else np.zeros(shape, dtype)
    return flat.astype(dtype), offsets


def content_labels(meta, box_ids, boxes, box_offsets, fm, fa):
    """1 where the transmitted CPM differs from what its sender perceived: a different set of objects, any box
    field changed, or a different reported sender position. Exact comparison: benign CPMs reproduce the sender's
    detections and pose bit for bit."""
    fidx = pd.Series(np.arange(len(fm)), index=pd.MultiIndex.from_arrays(
        [fm["episode"], fm["vehicle_id"].astype(str), fm["timestamp"].astype(int)]))
    fr = fidx.loc[list(zip(meta["episode"], meta["vehicle_id"].astype(str), meta["timestamp"].astype(int)))].to_numpy()
    po = fa["pred_offsets"]
    changed = np.ones(len(meta), bool)
    same_n = np.diff(box_offsets) == np.diff(po)[fr]
    for r in np.flatnonzero(same_n):
        a, b = box_offsets[r], box_offsets[r + 1]
        f = fr[r]
        ci, pi = box_ids[a:b], fa["pred_ids"][po[f]:po[f + 1]]
        if not np.array_equal(np.sort(ci), np.sort(pi)):
            continue
        if b > a and not np.array_equal(boxes[a:b][np.argsort(ci, kind="stable")],
                                        fa["pred_boxes"][po[f]:po[f + 1]][np.argsort(pi, kind="stable")]):
            continue
        changed[r] = False
    moved = (meta["ego_x"].to_numpy() != fa["poses"][fr, 0]) | (meta["ego_y"].to_numpy() != fa["poses"][fr, 1])
    return (changed | moved).astype(int)


def cmd_cache(args):
    from tqdm import tqdm

    root = Path(args.cpm_root)
    files = sorted(str(f) for f in root.glob("Town*/*/*.yaml") if not f.name.startswith("simulation_config"))
    print(f"[cache] {len(files):,} CPM files under {root}")
    metas, arrays, id_parts, box_parts = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for m, a, ids, boxes in tqdm(ex.map(_parse_cpm, files, chunksize=256), total=len(files),
                                     desc="parse CPMs", unit="cpm", mininterval=10):
            metas.append(m)
            arrays.append(a)
            id_parts.append(ids)
            box_parts.append(boxes)
    meta = pd.DataFrame(metas)
    # Canonical order makes every later stage deterministic.
    order = meta.sort_values(["episode", "run_id", "vehicle_id", "timestamp"]).index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    objects, offsets = _ragged(arrays, order, 3, np.float32)
    box_ids, box_offsets = _ragged(id_parts, order, 0, np.int64)
    boxes, _ = _ragged(box_parts, order, 12, np.float64)

    wd = Path(args.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    meta["scheduled_label"] = meta["label"]
    meta["scheduled_attack"] = meta["attack_type"]
    if args.label_rule == "content":
        if not ((wd / "frames_meta.pkl").exists() and (wd / "frames_arrays.npz").exists()):
            raise SystemExit("[cache] content labels need the frames cache (frames_meta.pkl, frames_arrays.npz) in the work dir")
        lab = content_labels(meta, box_ids, boxes, box_offsets, pd.read_pickle(wd / "frames_meta.pkl"),
                             dict(np.load(wd / "frames_arrays.npz")))
        wrong = int(((meta["scheduled_label"] == 0) & (lab == 1)).sum())
        if wrong:
            raise SystemExit(f"[cache] {wrong} CPMs scheduled benign differ from the sender's detections: check the data")
        flipped = int(((meta["scheduled_label"] == 1) & (lab == 0)).sum())
        print(f"[cache] content labels: {flipped:,} CPMs sent during an attack interval but unchanged are benign "
              f"({100 * flipped / max(int(meta['scheduled_label'].sum()), 1):.2f}% of the scheduled ones)")
        meta["label"] = lab
        meta.loc[lab == 0, "attack_type"] = ""
    meta.to_pickle(wd / "cpm_meta.pkl")
    np.savez(wd / "cpm_objects.npz", objects=objects, offsets=offsets)
    np.savez(wd / "cpm_boxes.npz", ids=box_ids, boxes=boxes, offsets=box_offsets)
    summary = {
        "num_cpms": int(len(meta)),
        "num_episodes": int(meta["episode"].nunique()),
        "num_unique_frames": int(meta[["episode", "vehicle_id", "timestamp"]].drop_duplicates().shape[0]),
        "runs": sorted(meta["run_id"].unique().tolist()),
        "label_counts": meta["label"].value_counts().to_dict(),
        "attack_counts": meta["attack_type"].value_counts().to_dict(),
        "episodes_per_town": meta.groupby("town")["episode"].nunique().to_dict(),
    }
    (wd / "cache_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))


def _parse_frame(preds_path: str):
    p = Path(preds_path)
    with open(p, "r") as f:
        d = yaml.load(f, Loader=YAML_LOADER)
    det = d.get("detected_objects") or {}
    pred_ids, pred_boxes = boxes_from_opencood_dict(det)
    # inference_yaml_preds.py assigns ids in detection order, and _score.npy follows the same order.
    order = np.argsort(pred_ids, kind="stable")
    pred_ids, pred_boxes = pred_ids[order], pred_boxes[order]
    score_path = p.with_name(p.name.replace("_preds.yaml", "_score.npy"))
    if score_path.exists():
        scores = np.load(score_path).reshape(-1).astype(np.float32)
    elif len(pred_ids) == 0:
        scores = np.zeros(0, np.float32)
    else:
        raise FileNotFoundError(f"{score_path} missing but {len(pred_ids)} detections present")
    if len(scores) != len(pred_ids):
        raise ValueError(f"{p}: {len(scores)} scores vs {len(pred_ids)} detections")
    gt = {int(k): v for k, v in (d.get("vehicles") or {}).items()}
    gt_ids, gt_boxes = boxes_from_opencood_dict(gt)
    ep = p.parent.parent
    meta = {"episode": f"{ep.parent.name}/{ep.name}", "vehicle_id": p.parent.name,
            "timestamp": int(p.name.split("_")[0]), "road_type": str(d.get("road_type")),
            "ego_speed": float(d.get("ego_speed", 0.0))}
    pose = np.asarray(d["lidar_pose"], dtype=np.float64)
    return meta, pose, pred_ids, pred_boxes, scores, gt_ids, gt_boxes


def cmd_frames(args):
    from tqdm import tqdm

    root = Path(args.data_root)
    files = sorted(str(f) for f in root.glob("Town*/*/*/*_preds.yaml"))
    print(f"[frames] {len(files):,} frames under {root}")
    metas, poses, pid, pbox, psc, gid, gbox = [], [], [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for m, pose, a, b, s, c, g in tqdm(ex.map(_parse_frame, files, chunksize=128), total=len(files),
                                           desc="parse frames", unit="frame", mininterval=10):
            metas.append(m); poses.append(pose); pid.append(a); pbox.append(b); psc.append(s)
            gid.append(c); gbox.append(g)
    meta = pd.DataFrame(metas)
    order = meta.sort_values(["episode", "vehicle_id", "timestamp"]).index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    pred_ids, pred_off = _ragged(pid, order, 0, np.int64)
    pred_boxes, _ = _ragged(pbox, order, 12, np.float64)
    pred_scores, _ = _ragged(psc, order, 0, np.float32)
    gt_ids, gt_off = _ragged(gid, order, 0, np.int64)
    gt_boxes, _ = _ragged(gbox, order, 12, np.float64)
    wd = Path(args.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    meta.to_pickle(wd / "frames_meta.pkl")
    np.savez(wd / "frames_arrays.npz", poses=np.stack([poses[i] for i in order]),
             pred_ids=pred_ids, pred_boxes=pred_boxes, pred_scores=pred_scores, pred_offsets=pred_off,
             gt_ids=gt_ids, gt_boxes=gt_boxes, gt_offsets=gt_off)
    print(f"[frames] {len(meta):,} frames, {len(pred_ids):,} detections, {len(gt_ids):,} GT boxes; "
          f"episodes={meta['episode'].nunique()}")


def load_cache(work_dir: Path):
    meta = pd.read_pickle(work_dir / "cpm_meta.pkl")
    z = np.load(work_dir / "cpm_objects.npz")
    return meta, z["objects"], z["offsets"]


def make_folds(meta: pd.DataFrame, n_folds: int, split_seed: int) -> Dict[str, int]:
    """Town-stratified assignment of episodes to folds (round-robin after a seeded shuffle)."""
    rng = random.Random(split_seed)
    fold_of = {}
    offset = 0
    for town, eps in sorted(meta.groupby("town")["episode"].unique().items()):
        eps = sorted(eps)
        rng.shuffle(eps)
        for i, ep in enumerate(eps):
            fold_of[ep] = (i + offset) % n_folds
        offset += len(eps)
    return fold_of


def fold_roles(k: int, n_folds: int) -> Tuple[int, int]:
    """Test fold k, validation fold k+1 (mod n), all others train."""
    return k, (k + 1) % n_folds
