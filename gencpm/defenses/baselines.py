"""Rule baselines and gradient-boosted trees on hand-crafted window features (count changes, association jumps, heading
changes, uncorroborated objects), evaluated on the same folds as every other detector. The trees of every fold and run
are saved as baselines_W<W>/GBDT-rules_fold<k>_seed<s>.pkl; with --pretrained DIR they are loaded instead of trained."""

from __future__ import annotations
import pickle
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from gencpm.data.cache import fold_roles, load_cache, make_folds
from gencpm.defenses.temporal import window_index


def frame_rule_features(meta, objects, offsets, corroboration_radius: float, sensing_range: float):
    """Per-CPM hand-crafted features that need no labels.

    count, count_delta       object count and |change| vs previous CPM of the same stream
    nn_med, nn_max           median / max distance of each object to its nearest object in the
                             previous CPM of the same stream (temporal plausibility; IDs are not
                             persistent, so nearest-neighbour association is used)
    yaw_med                  median |yaw change| to that nearest neighbour (deg, wrapped)
    uncorroborated           fraction of this CPM's objects that lie within sensing range of another
                             CAV (same episode, run, timestamp) but have no object of that CAV within
                             `corroboration_radius` (cross-vehicle consistency); NaN if none checkable
    """
    n = len(meta)
    out = np.full((n, 6), np.nan, dtype=np.float32)
    counts = meta["n_obj"].to_numpy()
    out[:, 0] = counts
    stream_key = meta["episode"] + "|" + meta["run_id"] + "|" + meta["vehicle_id"]
    prev = np.full(n, -1, dtype=np.int64)
    same = stream_key.to_numpy()[1:] == stream_key.to_numpy()[:-1]
    prev[1:][same] = np.arange(n - 1)[same]

    def objs(i):
        return objects[offsets[i]:offsets[i + 1]]

    for i in tqdm(range(n), desc="rule features", unit="cpm", mininterval=5):
        j = prev[i]
        if j < 0:
            continue
        out[i, 1] = abs(int(counts[i]) - int(counts[j]))
        a, b = objs(i), objs(j)
        if len(a) and len(b):
            d = np.sqrt(((a[:, None, :2] - b[None, :, :2]) ** 2).sum(-1))
            nn = d.argmin(1)
            dmin = d[np.arange(len(a)), nn]
            dyaw = np.abs((a[:, 2] - b[nn, 2] + 180.0) % 360.0 - 180.0)
            out[i, 2], out[i, 3], out[i, 4] = np.median(dmin), dmin.max(), np.median(dyaw)

    ex = meta["ego_x"].to_numpy()
    ey = meta["ego_y"].to_numpy()
    groups = meta.groupby(["episode", "run_id", "timestamp"], sort=False).indices
    for _, idx in tqdm(groups.items(), total=len(groups), desc="cross-sender features", unit="frame", mininterval=5):
        if len(idx) < 2:
            continue
        for i in idx:
            a = objs(i)
            if not len(a):
                continue
            checkable = np.zeros(len(a), bool)
            confirmed = np.zeros(len(a), bool)
            for j in idx:
                if j == i:
                    continue
                in_range = np.hypot(a[:, 0] - ex[j], a[:, 1] - ey[j]) <= sensing_range
                if not in_range.any():
                    continue
                checkable |= in_range
                b = objs(j)
                if len(b):
                    d = np.sqrt(((a[:, None, :2] - b[None, :, :2]) ** 2).sum(-1)).min(1)
                    confirmed |= in_range & (d <= corroboration_radius)
            if checkable.any():
                out[i, 5] = 1.0 - confirmed[checkable].mean()
    return out


RULE_NAMES = ["count", "count_delta", "nn_med", "nn_max", "yaw_med", "uncorroborated"]


def save_gbdt(out, k, seed, clf):
    with open(Path(out) / f"GBDT-rules_fold{k}_seed{seed}.pkl", "wb") as f:
        pickle.dump(clf, f)


def load_gbdt(src, k, seed):
    with open(Path(src) / f"GBDT-rules_fold{k}_seed{seed}.pkl", "rb") as f:
        return pickle.load(f)


def cmd_baselines(args):
    from sklearn.ensemble import HistGradientBoostingClassifier

    wd = Path(args.work_dir)
    meta, objects, offsets = load_cache(wd)
    fold_of = make_folds(meta, args.folds, args.split_seed)
    meta["fold"] = meta["episode"].map(fold_of)
    feat_path = wd / "rule_features.npy"
    if feat_path.exists() and not args.overwrite:
        R = np.load(feat_path)
    else:
        t0 = time.time()
        R = frame_rule_features(meta, objects, offsets, args.corroboration_radius, args.sensing_range)
        np.save(feat_path, R)
        print(f"[baselines] rule features computed in {time.time() - t0:.0f}s")

    # Same features on the benign version of every stream (transmitter's own detections), used to
    # measure the false-positive cost of gating on unattacked traffic.
    R_b = None
    if (wd / "frames_meta.pkl").exists():
        fmeta = pd.read_pickle(wd / "frames_meta.pkl")
        fr = np.load(wd / "frames_arrays.npz")
        key = fmeta["episode"] + "|" + fmeta["vehicle_id"] + "|" + fmeta["timestamp"].astype(str)
        pos = pd.Series(np.arange(len(fmeta)), index=key.to_numpy())
        rows = pos.loc[(meta["episode"] + "|" + meta["vehicle_id"] + "|" + meta["timestamp"].astype(str)).to_numpy()].to_numpy()
        po, pb = fr["pred_offsets"], fr["pred_boxes"]
        parts = [pb[po[r]:po[r + 1]][:, [0, 1, 10]].astype(np.float32) for r in rows]
        b_off = np.zeros(len(parts) + 1, dtype=np.int64)
        b_off[1:] = np.cumsum([len(a) for a in parts])
        b_obj = np.concatenate(parts, axis=0)
        meta_b = meta.copy()
        meta_b["n_obj"] = np.diff(b_off)
        R_b = frame_rule_features(meta_b, b_obj, b_off, args.corroboration_radius, args.sensing_range)
        ben = meta["label"].to_numpy() == 0
        print(f"[baselines] benign-version features: max |diff| on benign CPMs = "
              f"{np.nanmax(np.abs(np.nan_to_num(R[ben]) - np.nan_to_num(R_b[ben]))):.2e}")

    # Evaluate on the same window set as the temporal models (score = last frame of the window).
    ends = window_index(meta, args.seq_len)
    wfold = meta["fold"].to_numpy()[ends]
    y = meta["label"].to_numpy()[ends]
    out = wd / f"baselines_W{args.seq_len}"
    out.mkdir(parents=True, exist_ok=True)

    # Window-level hand features: last-frame rules + their max over the window.
    steps = np.arange(-args.seq_len + 1, 1)
    Rw = R[ends[:, None] + steps[None, :]]  # (n, W, 6)
    def window_feats(Rw_):
        with np.errstate(all="ignore"):
            return np.concatenate([Rw_[:, -1, :], np.nanmax(Rw_, axis=1), np.nanmean(Rw_, axis=1)], axis=1)
    Xh = window_feats(Rw)
    Rw_b = R_b[ends[:, None] + steps[None, :]] if R_b is not None else None
    Xh_b = window_feats(Rw_b) if R_b is not None else None

    for k in tqdm(range(args.folds), desc=f"{args.cmd} folds", unit="fold"):
        test_f, val_f = fold_roles(k, args.folds)
        tr = np.where((wfold != test_f) & (wfold != val_f))[0]
        va = np.where(wfold == val_f)[0]
        te = np.where(wfold == test_f)[0]
        # Unsupervised single-rule scores (higher = more suspicious); NaN -> 0 (no evidence).
        for r, name in enumerate(RULE_NAMES):
            if name == "count":
                continue  # not a monotone suspicion score on its own
            s = np.nan_to_num(Rw[:, -1, r], nan=0.0)
            extra = {"p_test_benign": np.nan_to_num(Rw_b[te, -1, r], nan=0.0)} if Rw_b is not None else {}
            np.savez(out / f"rule-{name}_fold{k}_seed0_probs.npz", val_idx=va, test_idx=te,
                     p_val=s[va], p_test=s[te], **extra)
        for seed in args.seeds:
            if args.pretrained:
                clf = load_gbdt(args.pretrained, k, seed)
            else:
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                     early_stopping=True, random_state=seed)
                clf.fit(Xh[tr], y[tr])
                save_gbdt(out, k, seed, clf)
            extra = {"p_test_benign": clf.predict_proba(Xh_b[te])[:, 1]} if Xh_b is not None else {}
            np.savez(out / f"GBDT-rules_fold{k}_seed{seed}_probs.npz", val_idx=va, test_idx=te,
                     p_val=clf.predict_proba(Xh[va])[:, 1], p_test=clf.predict_proba(Xh[te])[:, 1], **extra)
        print(f"  fold {k} done")
    np.save(out / "window_ends.npy", ends)
