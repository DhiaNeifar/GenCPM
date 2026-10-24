"""
Controlled attack scenarios generated in memory from the cached perception data (frames cache of
`python -m gencpm frames`). Used for the parameter sweeps, the number-of-attackers study, and the coordinated
attack set. Attackers are a fixed set of CAVs per episode; attacks run in back-to-back intervals of
`interval_s` seconds (state resets each interval, as in the released schedules).

Uncoordinated (one attacker acts alone; reuses the GenCPM attack classes, so behaviour matches the
released data): CF (AddObjectAttack), CS (RemoveObjectAttack), CSD (DriftAttack), ISP (WhiteNoiseAttack).

Coordinated (all colluders follow one shared plan; real vehicles are identified by matching each
colluder's detections to the vehicle positions, which the colluders can exchange):
  CG    colluding ghosts      the same fabricated vehicles (lane next to a real vehicle, constant
                              velocity) are reported by every colluder within sensing range
  CSUP  coordinated suppress. every colluder removes its detections of the same target vehicles
  CCSD  coordinated drift     every colluder shifts its detections of the same targets by the same
                              accumulating offset (fixed direction, constant rate, capped)
  CISP  coordinated ISP       every colluder applies the same per-frame random perturbation to its
                              detections of the same vehicle
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from gencpm.attacks import AttackContext, create_attack
from gencpm.cpm.tracking import SimpleTracker

TICK_S = 0.05
MATCH_M = 2.5      # a detection belongs to a vehicle if within this distance
SENSE_M = 50.0     # colluder reports a shared ghost only within its sensing range
BOX = ("location", "center", "extent", "angle")
SIMPLE = {"CF": "AddObjectAttack", "CS": "RemoveObjectAttack", "CSD": "DriftAttack", "ISP": "WhiteNoiseAttack",
          # advanced single-attacker attacks (attacks/advanced_attacks.py)
          "GT": "GhostTrackAttack", "SD": "StealthDriftAttack", "RP": "ReplayAttack", "EG": "EvasiveGhostAttack",
          "PF": "PoseFalsificationAttack"}
DEFAULT_PARAMS = {
    "CF": {"num_objects": 25, "radius_range": [2.0, 40.0], "z_offset": 0.0},
    "CS": {"max_remove_fraction": 0.95, "max_remove_count": 10000},
    "CSD": {"mean_step": 2.0, "std_step": 0.8, "yaw_step_mean": 10.0, "yaw_step_std": 5.0, "num_targets": 10000},
    "ISP": {"mean_drift": 3.0, "std_drift": 1.5, "z_drift_mean": 0.0, "z_drift_std": 0.5, "yaw_drift_mean": 15.0,
            "yaw_drift_std": 8.0, "pitch_drift_mean": 5.0, "pitch_drift_std": 3.0, "num_targets": 10000},
    "GT": {"num_ghosts": 2, "lane_offset": 3.5, "along_range": [-10.0, 10.0], "max_range": 40.0},
    "SD": {"rate_mps": 1.5, "max_offset_m": 8.0, "target_fraction": 1.0},
    "RP": {"delay_s": 2.0},
    "EG": {"num_ghosts": 2, "lane_offset": 3.5, "along_range": [-10.0, 10.0], "max_range": 40.0, "safe_m": 45.0},
    "PF": {"offset_m": 4.0, "yaw_deg": 5.0},
    "CG": {"num_ghosts": 2, "lane_offset": 3.5, "along": 10.0},
    "CSUP": {"num_targets": 2},
    "CCSD": {"num_targets": 4, "rate_mps": 1.5, "cap_m": 8.0},
    "CISP": {"mean": 3.0, "std": 1.5, "yaw_mean": 15.0, "yaw_std": 8.0},
}


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


# ----------------------------------------------------------------------------------------------
# episode data
# ----------------------------------------------------------------------------------------------

class Episode:
    """Per-episode view of the frames cache: detections, GT and poses per (cav, tick)."""

    def __init__(self, fm: pd.DataFrame, fa: dict, episode: str):
        rows = np.where(fm["episode"].to_numpy() == episode)[0]
        self.name = episode
        self.row = {(fm.at[r, "vehicle_id"], int(fm.at[r, "timestamp"])): r for r in rows}
        self.cavs = sorted({k[0] for k in self.row})
        self.ticks = sorted({k[1] for k in self.row})
        self.fa = fa
        # global vehicle truth per tick (union of all CAVs' GT lists)
        self.gt: Dict[int, Dict[int, np.ndarray]] = {}
        for (c, t), r in self.row.items():
            a, b = fa["gt_offsets"][r], fa["gt_offsets"][r + 1]
            d = self.gt.setdefault(t, {})
            for i, bx in zip(fa["gt_ids"][a:b], fa["gt_boxes"][a:b]):
                d.setdefault(int(i), bx)

    def pose(self, c, t):
        return self.fa["poses"][self.row[(c, t)]]

    def dets(self, c, t):
        r = self.row[(c, t)]
        a, b = self.fa["pred_offsets"][r], self.fa["pred_offsets"][r + 1]
        return self.fa["pred_ids"][a:b], self.fa["pred_boxes"][a:b], self.fa["pred_scores"][a:b]

    def gt_vel(self, vid, t):
        """Velocity (m/s) of a ground-truth vehicle from neighbouring ticks."""
        ts = [u for u in (t - 2, t, t + 2) if u in self.gt and vid in self.gt[u]]
        if len(ts) < 2:
            return np.zeros(2)
        p0, p1 = self.gt[ts[0]][vid][:2], self.gt[ts[-1]][vid][:2]
        return (p1 - p0) / ((ts[-1] - ts[0]) * TICK_S)

    def select_attackers(self, k: int, radius: float = 40.0) -> List[str]:
        """Greedy: start from the CAV pair most often within `radius`, add the CAV most often close to the group.
        At least one honest CAV is always left (else returns [])."""
        if k >= len(self.cavs):
            return []
        near = {c: {d: 0 for d in self.cavs} for c in self.cavs}
        for t in self.ticks:
            present = [c for c in self.cavs if (c, t) in self.row]
            for i, c in enumerate(present):
                for d in present[i + 1:]:
                    pc, pd_ = self.pose(c, t), self.pose(d, t)
                    if math.hypot(pc[0] - pd_[0], pc[1] - pd_[1]) <= radius:
                        near[c][d] += 1
                        near[d][c] += 1
        if k == 1:
            return [max(self.cavs, key=lambda c: (sum(near[c].values()), c))]
        pair = max(((c, d) for c in self.cavs for d in self.cavs if c < d), key=lambda p: (near[p[0]][p[1]], p))
        group = list(pair)
        while len(group) < k:
            rest = [c for c in self.cavs if c not in group]
            group.append(max(rest, key=lambda c: (sum(near[c][g] for g in group), c)))
        return sorted(group)

    def form_groups(self, k: int, radius: float = 40.0) -> List[List[str]]:
        """Partition all CAVs into colluding groups of k (the last group may be smaller). Greedy: start each group
        from the remaining pair most often within `radius`, add the remaining CAV most often close to the group."""
        near = {c: {d: 0 for d in self.cavs} for c in self.cavs}
        for t in self.ticks:
            present = [c for c in self.cavs if (c, t) in self.row]
            for i, c in enumerate(present):
                for d in present[i + 1:]:
                    pc, pd_ = self.pose(c, t), self.pose(d, t)
                    if math.hypot(pc[0] - pd_[0], pc[1] - pd_[1]) <= radius:
                        near[c][d] += 1
                        near[d][c] += 1
        rest, groups = list(self.cavs), []
        while rest:
            if len(rest) <= k:
                groups.append(sorted(rest))
                break
            pair = max(((c, d) for c in rest for d in rest if c < d), key=lambda p: (near[p[0]][p[1]], p))
            g = list(pair)
            while len(g) < k:
                g.append(max((c for c in rest if c not in g), key=lambda c: (sum(near[c][x] for x in g), c)))
            groups.append(sorted(g))
            rest = [c for c in rest if c not in g]
        return groups


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _vehicle_dicts(ids, boxes, track_ids):
    out = []
    for i, b, tid in zip(ids, boxes, track_ids):
        out.append({"id": str(int(i)), "track_id": int(tid),
                    "location": {"x": b[0], "y": b[1], "z": b[2]},
                    "center": {"x": b[3], "y": b[4], "z": b[5]},
                    "extent": {"x": b[6], "y": b[7], "z": b[8]},
                    "orientation": {"pitch": b[9], "yaw": b[10], "roll": b[11]}, "speed": 0.0})
    return out


def _from_dicts(vehicles):
    ids = np.array([int(v["id"]) for v in vehicles], dtype=np.int64)
    boxes = np.array([[v["location"]["x"], v["location"]["y"], v["location"]["z"],
                       v["center"]["x"], v["center"]["y"], v["center"]["z"],
                       v["extent"]["x"], v["extent"]["y"], v["extent"]["z"],
                       v["orientation"].get("pitch", 0.0), v["orientation"]["yaw"], v["orientation"].get("roll", 0.0)]
                      for v in vehicles], dtype=np.float64).reshape(-1, 12)
    return ids, boxes


def _match(boxes, pos, radius=MATCH_M):
    """Index of the detection closest to `pos` within radius, else -1."""
    if len(boxes) == 0:
        return -1
    d = np.hypot(boxes[:, 0] - pos[0], boxes[:, 1] - pos[1])
    j = int(d.argmin())
    return j if d[j] <= radius else -1


def _scores_for(ids, b_ids, b_scores, key):
    """Detector score per transmitted object; fabricated objects get U(0.8, 0.99) (seeded)."""
    lookup = dict(zip(b_ids.tolist(), b_scores.tolist()))
    sc = np.array([lookup.get(int(i), np.nan) for i in ids], dtype=np.float32)
    miss = np.isnan(sc)
    if miss.any():
        sc[miss] = np.random.default_rng(_seed(key)).uniform(0.8, 0.99, int(miss.sum())).astype(np.float32)
    return sc


# ----------------------------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------------------------

def generate(ep: Episode, attack: str, params: dict, attackers: List[str], seed: int,
             interval_s: float = 5.0, duty: float = 1.0, common: bool = False) -> Dict[Tuple[str, int], dict]:
    """Returns {(cav, tick): {"ids", "boxes", "scores", "obj_mal", "label", "attack"}} for the attackers' CPMs.
    obj_mal flags fabricated/displaced objects; label = 1 if the CPM is sent during an active interval."""
    p = dict(DEFAULT_PARAMS[attack]); p.update(params or {})
    iv_ticks = int(round(interval_s / TICK_S))
    t0_ep = ep.ticks[0]
    rng_iv = random.Random(_seed(ep.name, attack, seed, "duty"))
    active = {}  # interval index -> bool
    out = {}

    def interval_of(t):
        return (t - t0_ep) // iv_ticks

    def is_active(ii):
        if ii not in active:
            active[ii] = rng_iv.random() < duty
        return active[ii]

    if attack in SIMPLE:
        for c in attackers:
            random.seed(_seed(ep.name, attack, seed, c))
            tracker, history, obj, cur_iv = SimpleTracker(), [], None, None
            for t in ep.ticks:
                if (c, t) not in ep.row:
                    continue
                ids, boxes, scores = ep.dets(c, t)
                ti = tracker.update(boxes[:, :2] if len(boxes) else np.zeros((0, 2)), t)
                benign_vehicles = _vehicle_dicts(ids, boxes, ti)
                history.append((t, copy.deepcopy(benign_vehicles)))  # sender's own past detections (replay, ghosts)
                history = history[-200:]
                ii = interval_of(t)
                if not is_active(ii):
                    continue
                if ii != cur_iv:
                    obj, cur_iv = create_attack(SIMPLE[attack], p), ii
                pose = ep.pose(c, t)
                cpm = {"detected_vehicles": benign_vehicles,
                       "global_position": {"x": pose[0], "y": pose[1], "z": pose[2]}}
                others = {d: (float(ep.pose(d, t)[0]), float(ep.pose(d, t)[1]))
                          for d in ep.cavs if d != c and (d, t) in ep.row}
                obj.apply(cpm, AttackContext(vehicle_id=c, timestamp=t, history=list(history), others=others))
                nid, nbox = _from_dicts(cpm["detected_vehicles"])
                bpos = {int(i): b for i, b in zip(ids, boxes)}
                mal = np.array([int(i) not in bpos or np.abs(bpos[int(i)][[0, 1, 2, 10]] - nb[[0, 1, 2, 10]]).max() > 1e-6
                                for i, nb in zip(nid, nbox)], dtype=bool)
                out[(c, t)] = {"ids": nid, "boxes": nbox, "scores": _scores_for(nid, ids, scores, (ep.name, c, t, seed)),
                               "obj_mal": mal, "label": 1, "attack": attack}
        return out

    # ---------------- coordinated ----------------
    rng = np.random.default_rng(_seed(ep.name, attack, seed, "plan"))
    plan = {}  # interval -> shared plan

    def group_centroid(t):
        ps = [ep.pose(c, t)[:2] for c in attackers if (c, t) in ep.row]
        return np.mean(ps, axis=0) if ps else None

    def candidates(t):
        """Real vehicles detected by at least one colluder at tick t (ids from vehicle truth); with `common`,
        only the vehicles detected by every colluder present at t, so that all colluders can act on them."""
        sets = []
        for c in attackers:
            if (c, t) not in ep.row:
                continue
            _, boxes, _ = ep.dets(c, t)
            sets.append({vid for vid, gb in ep.gt.get(t, {}).items() if _match(boxes, gb[:2]) >= 0})
        if not sets:
            return []
        return sorted(set.intersection(*sets) if common else set.union(*sets))

    def near_all(xy, t):
        """True if every colluder present at t is within sensing range of position xy."""
        return all(math.hypot(ep.pose(c, t)[0] - xy[0], ep.pose(c, t)[1] - xy[1]) <= SENSE_M
                   for c in attackers if (c, t) in ep.row)

    for t in ep.ticks:
        ii = interval_of(t)
        if not is_active(ii):
            continue
        cen = group_centroid(t)
        if cen is None:
            continue
        if ii not in plan:
            cand = candidates(t)
            cand.sort(key=lambda v: (np.hypot(*(ep.gt[t][v][:2] - cen)), v))
            if attack == "CG":
                ghosts = []
                for a in cand:
                    if len(ghosts) >= int(p["num_ghosts"]):
                        break
                    gb = ep.gt[t][a].copy()
                    yaw = math.radians(gb[10])
                    h, nrm = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
                    off = rng.uniform(-p["along"], p["along"]) * h + rng.choice([-1, 1]) * p["lane_offset"] * nrm
                    gb[:2] = gb[:2] + off
                    if common and not near_all(gb[:2], t):  # every colluder must be able to report the ghost
                        continue
                    ghosts.append({"box": gb, "vel": ep.gt_vel(a, t), "t0": t,
                                   "id": int(rng.integers(0, 65536))})
                plan[ii] = {"ghosts": ghosts}
            elif attack == "CSUP":
                plan[ii] = {"targets": cand[: int(p["num_targets"])]}
            elif attack == "CCSD":
                tg = cand[: int(p["num_targets"])]
                plan[ii] = {"targets": tg, "t0": t,
                            "dir": {v: (lambda th: np.array([math.cos(th), math.sin(th)]))(rng.uniform(0, 2 * math.pi))
                                    for v in tg}}
            elif attack == "CISP":
                plan[ii] = {}
        pl = plan[ii]
        gt_t = ep.gt.get(t, {})
        for c in attackers:
            if (c, t) not in ep.row:
                continue
            ids, boxes, scores = ep.dets(c, t)
            nid, nbox = ids.copy(), boxes.copy()
            mal = np.zeros(len(nid), bool)
            keep = np.ones(len(nid), bool)
            if attack == "CSUP":
                for v in pl["targets"]:
                    if v in gt_t:
                        j = _match(nbox, gt_t[v][:2])
                        if j >= 0:
                            keep[j] = False
            elif attack == "CCSD":
                off = min(p["rate_mps"] * (t - pl["t0"]) * TICK_S, p["cap_m"])
                for v in pl["targets"]:
                    if v in gt_t:
                        j = _match(boxes, gt_t[v][:2])
                        if j >= 0 and off > 0:
                            nbox[j, :2] = boxes[j, :2] + off * pl["dir"][v]
                            mal[j] = True
            elif attack == "CISP":
                used = set()
                shared = set(candidates(t)) if common else None  # with `common`: vehicles every colluder detects
                for v, gb in gt_t.items():
                    if shared is not None and v not in shared:
                        continue
                    j = _match(boxes, gb[:2])
                    if j < 0 or j in used:
                        continue
                    used.add(j)
                    r = np.random.default_rng(_seed(ep.name, seed, "cisp", v, t))  # shared by all colluders
                    rad, th = max(0.0, r.normal(p["mean"], p["std"])), r.uniform(0, 2 * math.pi)
                    nbox[j, 0] += rad * math.cos(th)
                    nbox[j, 1] += rad * math.sin(th)
                    nbox[j, 10] += r.normal(p["yaw_mean"], p["yaw_std"])
                    mal[j] = True
            nid, nbox, mal = nid[keep], nbox[keep], mal[keep]
            if attack == "CG":
                pc = ep.pose(c, t)
                add_b, add_i = [], []
                for g in pl["ghosts"]:
                    gb = g["box"].copy()
                    gb[:2] = gb[:2] + g["vel"] * (t - g["t0"]) * TICK_S
                    if math.hypot(gb[0] - pc[0], gb[1] - pc[1]) > SENSE_M:
                        continue
                    jit = np.random.default_rng(_seed(ep.name, seed, "cg", c, g["id"], t)).normal(0, 0.1, 2)
                    gb[:2] += jit  # per-colluder measurement noise
                    add_b.append(gb); add_i.append(g["id"])
                if add_b:
                    nid = np.concatenate([nid, np.array(add_i, np.int64)])
                    nbox = np.concatenate([nbox, np.stack(add_b)])
                    mal = np.concatenate([mal, np.ones(len(add_b), bool)])
            out[(c, t)] = {"ids": nid, "boxes": nbox, "scores": _scores_for(nid, ids, scores, (ep.name, c, t, seed)),
                           "obj_mal": mal, "label": 1, "attack": attack}
    return out
