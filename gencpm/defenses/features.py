"""Receiver-side CPM defenses for AdverCPM (Step 2): local plausibility, cross-sender consistency,
and a physics-informed learned detector (PILD). All checks use only information a receiver has:
the content and arrival order of CPMs. Sender-provided track ids are not trusted; the receiver
associates objects across consecutive CPMs of the same sender itself.

Stages (work dir = output of `python -m gencpm frames` and `cache`):
  features  per-object and per-CPM features, for the transmitted CPMs and for the benign version
            of every CPM (same frames, no attack) -> defense_features.npz
  fit       per fold: calibrate plausibility/consistency on benign *training* objects, train PILD
            (3 seeds); write CPM-level probabilities (same format as the temporal models, evaluated on
            the same W-windows) and object-level gates for late fusion -> defenses_W<W>/

Per-object features (receiver view, sender s, time t; previous CPM of s at t' < t):
  0 nn_prev     distance (m) to the nearest object in s's previous CPM
  1 speed       nn_prev / (t - t'), associated objects only  [m/s]
  2 cv_resid    |p_t - (p_{t'} + v_{t'} (t - t'))|, constant-velocity residual over the associated chain
  3 yaw_rate    heading change mod 180 deg / (t - t')      [deg/s] (detector heading is front/back ambiguous)
  4 head_mis    angle between motion direction and reported heading (mod 180), if speed > 2 m/s
  5 is_new      1 if no object of the previous CPM within the association gate
  6 dist_sender distance from the sender (m)
  7 len, 8 wid  reported half-length / half-width (m)
  9 n_check     other CAVs within R_CHECK of the object (they should be able to see it)
 10 n_confirm   of those, how many report an object within D_MATCH
 11 unconf      (n_check - n_confirm) / n_check
 12 nn_other    distance to the nearest object reported by those CAVs
 13 age         consecutive associated CPMs (track length) at the receiver
 14 max_speed   max implied speed over the associated chain
Per-CPM features:
  0 n_obj, 1 count_delta, 2 vanish_near (objects of the previous CPM within NEAR_M of the sender
  with no counterpart now), 3 omission (objects reported by other CAVs within NEAR_M of the
  sender, and within D_MATCH of nothing in s's CPM), 4 n_others_near (CAVs within R_CHECK)"""

from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from gencpm.data.cache import load_cache


DT_TICK = 0.05          # seconds per simulator tick


GATE_BASE_M = 2.0       # receiver association gate between consecutive CPMs of one sender:


V_MAX = 40.0            #   gate = GATE_BASE_M + V_MAX * dt  (6 m at 10 Hz)


R_CHECK = 40.0          # range within which another CAV is expected to detect a vehicle


D_MATCH = 2.5           # two reports refer to the same object if closer than this


NEAR_M = 30.0           # "near the sender" radius for vanish/omission checks


OBJ_FEATS = ["nn_prev", "speed", "cv_resid", "yaw_rate", "head_mis", "is_new", "dist_sender", "len", "wid",
             "n_check", "n_confirm", "unconf", "nn_other", "age", "max_speed"]


CPM_FEATS = ["n_obj", "count_delta", "vanish_near", "omission", "n_others_near"]


PLAUS_OBJ = ["speed", "cv_resid", "yaw_rate", "head_mis", "len", "wid", "max_speed"]  # upper-tail checks


PLAUS_CPM = ["count_delta", "vanish_near"]


CONS_OBJ = ["nn_other"]


CONS_CPM = ["omission"]


def _wrap_deg(a):
    return np.abs((a + 180.0) % 360.0 - 180.0)


def compute_features(meta: pd.DataFrame, ids, boxes, offsets):
    """ids/boxes/offsets: ragged per-CPM content (OpenCOOD 12-float boxes). Returns
    (obj_feats (n_obj, 15) float32, cpm_feats (n_cpm, 5) float32)."""
    n = len(meta)
    xy = boxes[:, 0:2].astype(np.float64)
    yaw = boxes[:, 10].astype(np.float64)
    ext = boxes[:, 6:8].astype(np.float64)
    F = np.full((len(ids), len(OBJ_FEATS)), np.nan, dtype=np.float32)
    C = np.full((n, len(CPM_FEATS)), np.nan, dtype=np.float32)
    sx = meta["ego_x"].to_numpy(np.float64)
    sy = meta["ego_y"].to_numpy(np.float64)
    ts = meta["timestamp"].to_numpy()
    stream = (meta["episode"] + "|" + meta["run_id"] + "|" + meta["vehicle_id"]).to_numpy()
    prev = np.full(n, -1, dtype=np.int64)
    same = stream[1:] == stream[:-1]
    prev[1:][same] = np.arange(n - 1)[same]

    # chain state per object row: index of associated object in previous CPM (-1 if new)
    link = np.full(len(ids), -1, dtype=np.int64)
    age = np.zeros(len(ids), dtype=np.int32)
    vel = np.full((len(ids), 2), np.nan)
    maxsp = np.zeros(len(ids), dtype=np.float32)

    F[:, 7] = ext[:, 0]
    F[:, 8] = ext[:, 1]
    for i in tqdm(range(n), desc="object features", unit="cpm", mininterval=5):
        a, b = offsets[i], offsets[i + 1]
        C[i, 0] = b - a
        if b > a:
            F[a:b, 6] = np.hypot(xy[a:b, 0] - sx[i], xy[a:b, 1] - sy[i])
        j = prev[i]
        if j < 0:
            continue
        pa, pb = offsets[j], offsets[j + 1]
        C[i, 1] = abs((b - a) - (pb - pa))
        dt = max(1, ts[i] - ts[j]) * DT_TICK
        gate = GATE_BASE_M + V_MAX * dt
        if b == a or pb == pa:
            if b > a:
                F[a:b, 5] = 1.0
            if pb > pa:  # everything near the sender vanished
                C[i, 2] = float((np.hypot(xy[pa:pb, 0] - sx[i], xy[pa:pb, 1] - sy[i]) <= NEAR_M).sum())
            continue
        d = np.sqrt(((xy[a:b, None, :] - xy[None, pa:pb, :]) ** 2).sum(-1))
        nn = d.argmin(1)
        dmin = d[np.arange(b - a), nn]
        matched = dmin <= gate
        F[a:b, 0] = dmin
        F[a:b, 1] = np.where(matched, dmin / dt, np.nan)
        F[a:b, 5] = (~matched).astype(np.float32)
        dyaw = _wrap_deg(yaw[a:b] - yaw[pa + nn])
        dyaw = np.minimum(dyaw, 180.0 - dyaw)
        F[a:b, 3] = np.where(matched, dyaw / dt, np.nan)
        disp = xy[a:b] - xy[pa + nn]
        sp = dmin / dt
        mdir = np.degrees(np.arctan2(disp[:, 1], disp[:, 0]))
        mis = _wrap_deg(mdir - yaw[a:b])
        mis = np.minimum(mis, 180.0 - mis)  # heading ambiguity (front/back)
        F[a:b, 4] = np.where(matched & (sp > 2.0), mis, np.nan)
        # chain bookkeeping and constant-velocity residual
        for k in range(b - a):
            if not matched[k]:
                continue
            p = pa + nn[k]
            link[a + k] = p
            age[a + k] = age[p] + 1
            v_new = disp[k] / dt
            if not np.isnan(vel[p, 0]):
                pred = xy[p] + vel[p] * dt
                F[a + k, 2] = float(np.hypot(*(xy[a + k] - pred)))
            vel[a + k] = v_new
            maxsp[a + k] = max(maxsp[p], sp[k])
        # objects near the sender in the previous CPM with no counterpart now
        dprev = d.min(0)
        near_prev = np.hypot(xy[pa:pb, 0] - sx[i], xy[pa:pb, 1] - sy[i]) <= NEAR_M
        C[i, 2] = float((near_prev & (dprev > gate)).sum())
    F[:, 13] = age
    F[:, 14] = maxsp

    # cross-sender consistency at the same timestamp (same episode and run)
    for _, idx in meta.groupby(["episode", "run_id", "timestamp"], sort=False).indices.items():
        if len(idx) < 2:
            C[idx, 4] = 0
            continue
        for i in idx:
            a, b = offsets[i], offsets[i + 1]
            others = [j for j in idx if j != i]
            C[i, 4] = float(sum(np.hypot(sx[j] - sx[i], sy[j] - sy[i]) <= R_CHECK for j in others))
            n_check = np.zeros(b - a)
            n_conf = np.zeros(b - a)
            nn_other = np.full(b - a, np.inf)
            omission = 0
            for j in others:
                ja, jb = offsets[j], offsets[j + 1]
                if b > a:
                    in_rng = np.hypot(xy[a:b, 0] - sx[j], xy[a:b, 1] - sy[j]) <= R_CHECK
                    if in_rng.any():
                        n_check += in_rng
                        if jb > ja:
                            dd = np.sqrt(((xy[a:b, None, :] - xy[None, ja:jb, :]) ** 2).sum(-1)).min(1)
                            n_conf += in_rng & (dd <= D_MATCH)
                            nn_other = np.where(in_rng, np.minimum(nn_other, dd), nn_other)
                # objects other CAVs report near the sender that the sender does not report
                if jb > ja:
                    near = np.hypot(xy[ja:jb, 0] - sx[i], xy[ja:jb, 1] - sy[i]) <= NEAR_M
                    if near.any():
                        if b > a:
                            dd = np.sqrt(((xy[ja:jb][near][:, None, :] - xy[None, a:b, :]) ** 2).sum(-1)).min(1)
                            omission += int((dd > D_MATCH).sum())
                        else:
                            omission += int(near.sum())
            C[i, 3] = omission
            if b > a:
                F[a:b, 9] = n_check
                F[a:b, 10] = n_conf
                F[a:b, 11] = np.where(n_check > 0, (n_check - n_conf) / np.maximum(n_check, 1), np.nan)
                F[a:b, 12] = np.where(np.isfinite(nn_other), nn_other, np.nan)
    return F, C


def object_labels(meta, cb, fr, rows):
    """1 if a transmitted object is fabricated or displaced w.r.t. the sender's own detection."""
    lab = np.zeros(len(cb["ids"]), dtype=np.int8)
    co, po = cb["offsets"], fr["pred_offsets"]
    mal = np.where(meta["label"].to_numpy() == 1)[0]
    for i in mal:
        a, b = co[i], co[i + 1]
        if b == a:
            continue
        r = rows[i]
        pids = fr["pred_ids"][po[r]:po[r + 1]]
        pbox = fr["pred_boxes"][po[r]:po[r + 1]]
        pos = {int(k): j for j, k in enumerate(pids)}
        for k in range(a, b):
            j = pos.get(int(cb["ids"][k]))
            if j is None or np.abs(cb["boxes"][k, :3] - pbox[j, :3]).max() > 1e-3 \
                    or abs(cb["boxes"][k, 10] - pbox[j, 10]) > 1e-3:
                lab[k] = 1
    return lab


def cmd_features(args):
    wd = Path(args.work_dir)
    meta, _, _ = load_cache(wd)
    cb = dict(np.load(wd / "cpm_boxes.npz"))
    fm = pd.read_pickle(wd / "frames_meta.pkl")
    fr = dict(np.load(wd / "frames_arrays.npz"))
    key = fm["episode"] + "|" + fm["vehicle_id"] + "|" + fm["timestamp"].astype(str)
    pos = pd.Series(np.arange(len(fm)), index=key.to_numpy())
    rows = pos.loc[(meta["episode"] + "|" + meta["vehicle_id"] + "|" + meta["timestamp"].astype(str)).to_numpy()].to_numpy()

    t0 = time.time()
    F, C = compute_features(meta, cb["ids"], cb["boxes"], cb["offsets"])
    print(f"[features] transmitted: {len(F):,} objects, {len(C):,} CPMs ({time.time() - t0:.0f}s)")
    # benign version of every CPM: each sender transmits its own detections
    po = fr["pred_offsets"]
    parts_i = [fr["pred_ids"][po[r]:po[r + 1]] for r in rows]
    parts_b = [fr["pred_boxes"][po[r]:po[r + 1]] for r in rows]
    b_off = np.zeros(len(rows) + 1, dtype=np.int64)
    b_off[1:] = np.cumsum([len(p) for p in parts_i])
    b_ids, b_boxes = np.concatenate(parts_i), np.concatenate(parts_b)
    t0 = time.time()
    Fb, Cb = compute_features(meta, b_ids, b_boxes, b_off)
    print(f"[features] benign version: {len(Fb):,} objects ({time.time() - t0:.0f}s)")
    lab = object_labels(meta, cb, fr, rows)
    obj_row = np.repeat(np.arange(len(meta)), np.diff(cb["offsets"]))
    np.savez(wd / "defense_features.npz", F=F, C=C, Fb=Fb, Cb=Cb, obj_label=lab, obj_row=obj_row,
             b_offsets=b_off, b_obj_row=np.repeat(np.arange(len(meta)), np.diff(b_off)))
    by = pd.Series(lab).groupby(meta["attack_type"].to_numpy()[obj_row]).mean()
    print("[features] manipulated-object share by CPM attack type:", by.round(3).to_dict())
