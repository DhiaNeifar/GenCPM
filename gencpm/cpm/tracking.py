"""
Persistent object identities for a transmitter's own detections.

PointPillars detection ids are new in every frame, but an ETSI CPM keeps an object id across
messages, and stateful attacks (CSD) need to follow the same physical object over time.
This module assigns a `track_id` to every detection of one vehicle's stream with a
constant-velocity, gated nearest-neighbour tracker (Hungarian assignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class _Track:
    track_id: int
    pos: np.ndarray            # last associated (x, y)
    vel: np.ndarray            # (vx, vy) per tick
    last_tick: int
    misses: int = 0


@dataclass
class SimpleTracker:
    """Constant-velocity tracker in world coordinates.

    gate_m:   max distance (m) between a predicted track position and a detection
    max_age:  number of consecutive frames a track may be unmatched before it is dropped
    """
    gate_m: float = 3.0
    max_age: int = 3
    _tracks: List[_Track] = field(default_factory=list)
    _next_id: int = 1

    def update(self, positions: np.ndarray, tick: int) -> List[int]:
        """positions: (N, 2) world x, y of this frame's detections. Returns one track id per detection."""
        n = len(positions)
        ids = [0] * n
        if self._tracks and n:
            pred = np.stack([t.pos + t.vel * (tick - t.last_tick) for t in self._tracks])
            cost = np.linalg.norm(pred[:, None, :] - positions[None, :, :], axis=2)
            big = 1e6
            rows, cols = linear_sum_assignment(np.where(cost <= self.gate_m, cost, big))
            matched_t, matched_d = set(), set()
            for r, c in zip(rows, cols):
                if cost[r, c] > self.gate_m:
                    continue
                t = self._tracks[r]
                dt = max(1, tick - t.last_tick)
                t.vel = (positions[c] - t.pos) / dt
                t.pos = positions[c].copy()
                t.last_tick = tick
                t.misses = 0
                ids[c] = t.track_id
                matched_t.add(r)
                matched_d.add(c)
            for r, t in enumerate(self._tracks):
                if r not in matched_t:
                    t.misses += 1
        else:
            for t in self._tracks:
                t.misses += 1
        for c in range(n):
            if ids[c] == 0:
                t = _Track(self._next_id, positions[c].copy(), np.zeros(2), tick)
                self._tracks.append(t)
                ids[c] = t.track_id
                self._next_id += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]
        return ids


def assign_track_ids(vehicles: List[Dict], tracker: SimpleTracker, tick: int) -> None:
    """Annotate CPM detected_vehicles entries in place with a persistent `track_id`."""
    if not vehicles:
        tracker.update(np.zeros((0, 2)), tick)
        return
    pos = np.array([[float(v["location"]["x"]), float(v["location"]["y"])] for v in vehicles])
    for v, tid in zip(vehicles, tracker.update(pos, tick)):
        v["track_id"] = int(tid)
