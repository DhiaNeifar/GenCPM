"""
Advanced (non-naive) CPM attacks. Each attack instance lives for one attack interval and keeps state
across the CPMs of that interval. They rely on extra context provided by the generator:

    ctx.history : list of (tick, benign detected_vehicles with track_id) of this sender, oldest first
    ctx.others  : {vehicle_id: (x, y)} positions of the other CAVs at this tick (known to an attacker
                  from the CPM headers it receives)

GhostTrackAttack     fabricated vehicles that move consistently (constant velocity along a real
                     vehicle's lane, realistic size/heading): passes kinematic plausibility checks.
StealthDriftAttack   targeted objects drift in a fixed direction at a small constant rate, so the
                     implied speed stays plausible and the constant-velocity residual stays small.
ReplayAttack         the sender re-transmits its own genuine detections from `delay_s` seconds ago.
EvasiveGhostAttack   ghost tracks spawned and shown only where no other CAV is within `safe_m`,
                     i.e. where no other vehicle could contradict them (defeats consistency checks).
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from .base import Attack, AttackContext
from .helpers import generate_unused_id, get_detected_vehicles

TICK_S = 0.05
GHOST_TRACK_BASE = 900000  # track ids used for fabricated tracks


def _xy(v: Dict[str, Any]) -> Tuple[float, float]:
    return float(v["location"]["x"]), float(v["location"]["y"])


def _velocity_from_history(ctx: AttackContext, track_id: int) -> Optional[Tuple[float, float]]:
    """Velocity (m/s) of a sender-tracked object from its last two benign observations."""
    obs = []
    for tick, vehicles in reversed(ctx.history or []):
        for v in vehicles:
            if v.get("track_id") == track_id:
                obs.append((tick, _xy(v)))
                break
        if len(obs) == 2:
            break
    if len(obs) < 2 or obs[0][0] == obs[1][0]:
        return None
    (t1, p1), (t0, p0) = obs
    dt = (t1 - t0) * TICK_S
    return (p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt


class _GhostBase(Attack):
    """Shared logic: spawn ghosts next to real vehicles (adjacent lane) and move them at constant velocity."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._ghosts: List[Dict[str, Any]] = []
        self._last_tick: Optional[int] = None

    def _spawn_ok(self, x: float, y: float, sender: Tuple[float, float], ctx: AttackContext) -> bool:
        return math.hypot(x - sender[0], y - sender[1]) <= float(self.params.get("max_range", 40.0))

    def _spawn(self, vehicles, ctx, sender, existing_ids) -> None:
        n = int(self.params.get("num_ghosts", 2))
        anchors = [v for v in vehicles if "track_id" in v]
        random.shuffle(anchors)
        for a in anchors:
            if len(self._ghosts) >= n:
                break
            ax, ay = _xy(a)
            yaw = float(a["orientation"]["yaw"])
            vel = _velocity_from_history(ctx, a["track_id"]) or (0.0, 0.0)
            hx, hy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            side = random.choice([-1.0, 1.0]) * float(self.params.get("lane_offset", 3.5))
            along = random.uniform(*self.params.get("along_range", (-10.0, 10.0)))
            gx, gy = ax + along * hx - side * hy, ay + along * hy + side * hx
            if not self._spawn_ok(gx, gy, sender, ctx):
                continue
            gid = generate_unused_id(existing_ids)
            existing_ids.add(gid)
            self._ghosts.append({
                "id": str(gid), "track_id": GHOST_TRACK_BASE + gid, "x": gx, "y": gy,
                "z": float(a["location"]["z"]), "vx": vel[0], "vy": vel[1], "yaw": yaw,
                "center": copy.deepcopy(a.get("center", {"x": 0.0, "y": 0.0, "z": 0.75})),
                "extent": copy.deepcopy(a.get("extent", {"x": 2.25, "y": 0.95, "z": 0.75})),
            })

    def _visible(self, g, ctx) -> bool:
        return True

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = get_detected_vehicles(cpm)
        gp = cpm.get("global_position", {})
        sender = (float(gp.get("x", 0.0)), float(gp.get("y", 0.0)))
        tick = int(ctx.timestamp)
        if self._last_tick is not None:  # constant-velocity motion
            dt = (tick - self._last_tick) * TICK_S
            for g in self._ghosts:
                g["x"] += g["vx"] * dt
                g["y"] += g["vy"] * dt
        self._last_tick = tick
        existing = set()
        for v in vehicles:
            try:
                existing.add(int(str(v.get("id"))))
            except (TypeError, ValueError):
                pass
        if len(self._ghosts) < int(self.params.get("num_ghosts", 2)):
            self._spawn(vehicles, ctx, sender, existing)
        added = {}
        for g in self._ghosts:
            if not self._visible(g, ctx):
                continue
            obj = {
                "id": g["id"], "track_id": g["track_id"],
                "location": {"x": g["x"], "y": g["y"], "z": g["z"]},
                "center": copy.deepcopy(g["center"]),
                "extent": copy.deepcopy(g["extent"]),
                "orientation": {"pitch": 0.0, "yaw": g["yaw"], "roll": 0.0},
                "speed": math.hypot(g["vx"], g["vy"]),
            }
            vehicles.append(obj)
            added[g["id"]] = obj
        cpm["detected_vehicles"] = vehicles
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": sorted(added),
            "altered_fields": ["detected_vehicles[*]"],
            "ground_truth_reference": {"added_objects": added},
        }


class GhostTrackAttack(_GhostBase):
    attack_type = "GhostTrackAttack"


class EvasiveGhostAttack(_GhostBase):
    """Ghosts exist only where no other CAV is within safe_m (nobody can contradict them)."""
    attack_type = "EvasiveGhostAttack"

    def _far_from_others(self, x, y, ctx) -> bool:
        safe = float(self.params.get("safe_m", 45.0))
        return all(math.hypot(x - ox, y - oy) > safe for ox, oy in (ctx.others or {}).values())

    def _spawn_ok(self, x, y, sender, ctx) -> bool:
        return super()._spawn_ok(x, y, sender, ctx) and self._far_from_others(x, y, ctx)

    def _visible(self, g, ctx) -> bool:
        return self._far_from_others(g["x"], g["y"], ctx)


class StealthDriftAttack(Attack):
    """Constant-direction, constant-rate drift of targeted objects (capped at max_offset_m)."""
    attack_type = "StealthDriftAttack"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._state: Dict[int, Dict[str, float]] = {}  # track_id -> direction, offset
        self._last_tick: Optional[int] = None

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = get_detected_vehicles(cpm)
        tick = int(ctx.timestamp)
        dt = 0.0 if self._last_tick is None else (tick - self._last_tick) * TICK_S
        self._last_tick = tick
        rate = float(self.params.get("rate_mps", 1.5))
        cap = float(self.params.get("max_offset_m", 8.0))
        frac = float(self.params.get("target_fraction", 1.0))
        for st in self._state.values():
            st["off"] = min(cap, st["off"] + rate * dt)
        targets, gt = [], {}
        for v in vehicles:
            tid = v.get("track_id")
            if tid is None:
                continue
            if tid not in self._state:
                if random.random() > frac:
                    self._state[tid] = {"dx": 0.0, "dy": 0.0, "off": 0.0, "skip": 1.0}
                else:
                    th = random.uniform(0.0, 2 * math.pi)
                    self._state[tid] = {"dx": math.cos(th), "dy": math.sin(th), "off": 0.0, "skip": 0.0}
            st = self._state[tid]
            if st["skip"] or st["off"] == 0.0:
                continue
            loc = v["location"]
            try:
                gt[int(v["id"])] = {"location": dict(loc), "orientation": dict(v.get("orientation", {}))}
            except (TypeError, ValueError):
                pass
            loc["x"] = float(loc["x"] + st["dx"] * st["off"])
            loc["y"] = float(loc["y"] + st["dy"] * st["off"])
            targets.append(v.get("id"))
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": targets,
            "altered_fields": ["location.x", "location.y"] if targets else [],
            "ground_truth_reference": gt,
        }


class ReplayAttack(Attack):
    """Re-transmit the sender's own genuine detections from delay_s seconds earlier."""
    attack_type = "ReplayAttack"

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        delay_ticks = int(round(float(self.params.get("delay_s", 2.0)) / TICK_S))
        target_tick = int(ctx.timestamp) - delay_ticks
        past = [(t, v) for t, v in (ctx.history or []) if t <= target_tick]
        if not past:  # not enough history yet: replay the oldest available frame
            past = (ctx.history or [])[:1]
        if not past:
            return {"attack_type": self.attack_type, "attack_parameters": self.params, "target_ids": [],
                    "altered_fields": [], "ground_truth_reference": None}
        replay_tick, replayed = past[-1]
        original = cpm.get("detected_vehicles", [])
        cpm["detected_vehicles"] = copy.deepcopy(replayed)
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": [v.get("id") for v in cpm["detected_vehicles"]],
            "altered_fields": ["detected_vehicles"],
            "ground_truth_reference": {"replayed_tick": replay_tick,
                                       "original_ids": [v.get("id") for v in original]},
        }


class PoseFalsificationAttack(Attack):
    """The sender misreports its own pose: every reported object undergoes the same rigid transform
    p' = p_s + R(d_psi) (p - p_s) + d, and the reported sender position becomes p_s + d. The offset d
    (|d| = offset_m, random direction) and rotation d_psi (+-yaw_deg) are fixed over the attack interval,
    so the message stays internally consistent and smooth in time (cf. position falsification in
    VeReMi and adversarial collaborator poses in V2XP-ASG)."""
    attack_type = "PoseFalsificationAttack"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._d: Optional[Tuple[float, float]] = None
        self._dpsi: float = 0.0

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        if self._d is None:
            th = random.uniform(0.0, 2 * math.pi)
            off = float(self.params.get("offset_m", 4.0))
            self._d = (off * math.cos(th), off * math.sin(th))
            self._dpsi = random.choice([-1.0, 1.0]) * float(self.params.get("yaw_deg", 5.0))
        gp = cpm.get("global_position", {})
        sx, sy = float(gp.get("x", 0.0)), float(gp.get("y", 0.0))
        c, s = math.cos(math.radians(self._dpsi)), math.sin(math.radians(self._dpsi))
        gt, targets = {}, []
        for v in get_detected_vehicles(cpm):
            loc = v["location"]
            try:
                gt[int(v["id"])] = {"location": dict(loc), "orientation": dict(v.get("orientation", {}))}
            except (TypeError, ValueError):
                pass
            rx, ry = float(loc["x"]) - sx, float(loc["y"]) - sy
            loc["x"] = sx + c * rx - s * ry + self._d[0]
            loc["y"] = sy + s * rx + c * ry + self._d[1]
            ori = v.setdefault("orientation", {})
            ori["yaw"] = float(ori.get("yaw", 0.0)) + self._dpsi
            targets.append(v.get("id"))
        if "global_position" in cpm:
            cpm["global_position"]["x"] = sx + self._d[0]
            cpm["global_position"]["y"] = sy + self._d[1]
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": targets,
            "altered_fields": ["global_position.x", "global_position.y", "location.x", "location.y", "orientation.yaw"],
            "ground_truth_reference": gt,
        }
