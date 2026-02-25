# attack.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Set


# ----------------------------------------------------------------------
# Context and base class
# ----------------------------------------------------------------------

@dataclass
class AttackContext:
    """Extra info about where/when the CPM is generated."""
    vehicle_id: str
    timestamp: float


class Attack:
    """
    Base class: modifies CPM in-place and returns metadata dict.

    Expected metadata keys:
      - attack_type
      - attack_parameters
      - target_ids
      - altered_fields
      - ground_truth_reference  (optional, e.g., pre-attack positions)
    """

    attack_type: str = "BaseAttack"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _get_detected_vehicles(cpm: Dict[str, Any]) -> List[Dict[str, Any]]:
    return cpm.get("detected_vehicles", [])


def _generate_unused_id(existing_ids: Set[int], id_min: int = 0, id_max: int = 65535) -> int:
    while True:
        new_id = random.randint(id_min, id_max)
        if new_id not in existing_ids:
            return new_id


# ----------------------------------------------------------------------
# DriftAttack
# ----------------------------------------------------------------------

class DriftAttack(Attack):
    """
    Gradual cumulative drift (random walk) in x, y, and optionally yaw.

    Params:
      - target_ids: Optional[List[int]]
      - num_targets: Optional[int]   (if no target_ids given)
      - mean_step: float (meters per frame)   [replaces mean_drift semantic]
      - std_step: float (meters per frame)    [replaces std_drift semantic]
      - yaw_step_mean: float (deg per frame, optional)
      - yaw_step_std: float (deg per frame, optional)
    """

    attack_type = "DriftAttack"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        # persistent offsets per object id
        # { id: {"dx": float, "dy": float, "dyaw": float} }
        self._offsets: Dict[int, Dict[str, float]] = {}

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = _get_detected_vehicles(cpm)
        if not vehicles:
            return self._empty_metadata()

        mean_step = float(self.params.get("mean_step", 0.05))  # m per frame
        std_step = float(self.params.get("std_step", 0.02))    # m per frame
        yaw_mean = float(self.params.get("yaw_step_mean", 0.0))   # deg per frame
        yaw_std = float(self.params.get("yaw_step_std", 0.0))     # deg per frame

        # Choose targets
        all_ids = [v.get("id") for v in vehicles]
        target_ids = self.params.get("target_ids")

        if target_ids is None:
            num_targets = int(self.params.get("num_targets", max(1, len(vehicles) // 3)))
            num_targets = max(1, min(num_targets, len(vehicles)))
            target_ids = random.sample(all_ids, num_targets)

        target_ids_set = set(target_ids)

        ground_truth: Dict[int, Dict[str, Any]] = {}
        altered_fields: List[str] = []

        for v in vehicles:
            vid = v.get("id")
            if vid not in target_ids_set:
                continue

            # ensure int key
            try:
                vid_int = int(vid)
            except (TypeError, ValueError):
                # if id is non-numeric, we skip persistent offsets
                vid_int = None

            loc = v.setdefault("location", {})
            ori = v.setdefault("orientation", {})

            # Save GT for this frame
            gt_loc = dict(loc)
            gt_ori = dict(ori)
            if vid_int is not None:
                ground_truth[vid_int] = {
                    "location": gt_loc,
                    "orientation": gt_ori,
                }

            # get previous cumulative offset
            if vid_int is not None:
                offset = self._offsets.setdefault(
                    vid_int, {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}
                )
            else:
                offset = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}

            # sample step in this frame
            step_r = max(0.0, random.gauss(mean_step, std_step))
            theta = random.uniform(0.0, 2 * math.pi)
            step_dx = step_r * math.cos(theta)
            step_dy = step_r * math.sin(theta)

            offset["dx"] += step_dx
            offset["dy"] += step_dy

            # apply cumulative offset on top of *current* ground-truth location
            loc["x"] = float(gt_loc.get("x", 0.0) + offset["dx"])
            loc["y"] = float(gt_loc.get("y", 0.0) + offset["dy"])

            if yaw_std != 0.0 or yaw_mean != 0.0:
                step_dyaw = random.gauss(yaw_mean, yaw_std)
                offset["dyaw"] += step_dyaw
                ori["yaw"] = float(gt_ori.get("yaw", 0.0) + offset["dyaw"])
                altered_fields.extend(["location.x", "location.y", "orientation.yaw"])
            else:
                altered_fields.extend(["location.x", "location.y"])

            # store back offsets if we had a numeric id
            if vid_int is not None:
                self._offsets[vid_int] = offset

        altered_fields = sorted(set(altered_fields))

        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": list(target_ids_set),
            "altered_fields": altered_fields,
            "ground_truth_reference": ground_truth,
        }

    def _empty_metadata(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": [],
            "altered_fields": [],
            "ground_truth_reference": None,
        }


# ----------------------------------------------------------------------
# WhiteNoiseAttack
# ----------------------------------------------------------------------

class WhiteNoiseAttack(Attack):
    """
    Adds a random spatial drift to selected objects.

    Params:
      - target_ids: Optional[List[int]]
      - num_targets: Optional[int]   (if no target_ids given)
      - mean_drift: float (meters)
      - std_drift: float (meters)
      - yaw_drift_mean: float (degrees, optional)
      - yaw_drift_std: float (degrees, optional)
    """

    attack_type = "WhiteNoiseAttack"

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = _get_detected_vehicles(cpm)
        if not vehicles:
            return self._empty_metadata()

        mean_drift = float(self.params.get("mean_drift", 0.5))
        std_drift  = float(self.params.get("std_drift", 0.2))
        z_mean = float(self.params.get("z_drift_mean", 0.0))
        z_std  = float(self.params.get("z_drift_std", 0.05))
        yaw_mean   = float(self.params.get("yaw_drift_mean", 0.0))
        yaw_std    = float(self.params.get("yaw_drift_std", 5.0))
        pitch_mean = float(self.params.get("pitch_drift_mean", 0.0))
        pitch_std  = float(self.params.get("pitch_drift_std", 2.0))

        # Choose targets
        all_ids = [v.get("id") for v in vehicles]
        target_ids = self.params.get("target_ids")

        if target_ids is None:
            num_targets = int(self.params.get("num_targets", max(1, len(vehicles) // 3)))
            num_targets = max(1, min(num_targets, len(vehicles)))
            target_ids = random.sample(all_ids, num_targets)

        target_ids_set = set(target_ids)

        ground_truth: Dict[int, Dict[str, Any]] = {}
        altered_fields: List[str] = []

        for v in vehicles:
            vid = v.get("id")
            if vid not in target_ids_set:
                continue

            loc = v.setdefault("location", {})
            ori = v.setdefault("orientation", {})

            # Save GT
            ground_truth[int(vid)] = {
                "location": dict(loc),
                "orientation": dict(ori),
            }

            # Sample drift
            r = max(0.0, random.gauss(mean_drift, std_drift))
            theta = random.uniform(0.0, 2 * math.pi)
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)

            loc["x"] = float(loc.get("x", 0.0) + dx)
            loc["y"] = float(loc.get("y", 0.0) + dy)

            if z_std != 0.0 or z_mean != 0.0:
                dz = random.gauss(z_mean, z_std)
                loc["z"] = float(loc.get("z", 0.0) + dz)
                altered_fields.extend(["location.x", "location.y", "location.z"])
            else:
                altered_fields.extend(["location.x", "location.y"])

            if yaw_std != 0.0 or yaw_mean != 0.0:
                dyaw = random.gauss(yaw_mean, yaw_std)
                ori["yaw"] = float(ori.get("yaw", 0.0) + dyaw)
                altered_fields.append("orientation.yaw")

            if pitch_std != 0.0 or pitch_mean != 0.0:
                dpitch = random.gauss(pitch_mean, pitch_std)
                ori["pitch"] = float(ori.get("pitch", 0.0) + dpitch)
                altered_fields.append("orientation.pitch")

        altered_fields = sorted(set(altered_fields))

        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": list(target_ids_set),
            "altered_fields": altered_fields,
            "ground_truth_reference": ground_truth,
        }

    def _empty_metadata(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": [],
            "altered_fields": [],
            "ground_truth_reference": None,
        }


# ----------------------------------------------------------------------
# AddObjectAttack
# ----------------------------------------------------------------------

class AddObjectAttack(Attack):
    """
    Injects fake objects into the CPM around the ego position.

    Params:
      - num_objects: int (default: 1)
      - radius_range: (r_min, r_max) in meters (default: (5, 20))
      - z_offset: float (vertical offset from ego, default: 0.0)
    """

    attack_type = "AddObjectAttack"

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = _get_detected_vehicles(cpm)
        ego = cpm.get("global_position", {"x": 0.0, "y": 0.0, "z": 0.0})

        num_objects = int(self.params.get("num_objects", 3))
        r_min, r_max = self.params.get("radius_range", (5.0, 20.0))
        z_offset = float(self.params.get("z_offset", 0.0))

        # Existing IDs
        existing_ids = {
            int(v.get("id")) for v in vehicles
            if isinstance(v.get("id"), (int, float))
        }

        # Compute average extent across existing vehicles
        extent_samples: List[Tuple[float, float, float]] = []
        avg_speed = 0
        for v in vehicles:
            ext = v.get("extent")
            avg_speed += v.get("speed", 0.0)
            if not isinstance(ext, dict):
                continue
            ex, ey, ez = ext.get("x"), ext.get("y"), ext.get("z")
            if isinstance(ex, (int, float)) and isinstance(ey, (int, float)) and isinstance(ez, (int, float)):
                extent_samples.append((float(ex), float(ey), float(ez)))
        if vehicles:
            avg_speed /= len(vehicles)
        if extent_samples:
            avg_ex = sum(e[0] for e in extent_samples) / len(extent_samples)
            avg_ey = sum(e[1] for e in extent_samples) / len(extent_samples)
            avg_ez = sum(e[2] for e in extent_samples) / len(extent_samples)
        else:
            # Fallback if extents are missing everywhere
            avg_ex, avg_ey, avg_ez = 2.0, 1.0, 1.0

        avg_extent = {"x": float(avg_ex), "y": float(avg_ey), "z": float(avg_ez)}

        new_ids: List[int] = []
        fake_objects: Dict[int, Dict[str, Any]] = {}

        for _ in range(num_objects):
            new_id = _generate_unused_id(existing_ids)
            existing_ids.add(new_id)
            new_ids.append(new_id)

            # Sample polar position around ego
            r = random.uniform(r_min, r_max)
            theta = random.uniform(0.0, 2 * math.pi)
            x = float(ego.get("x", 0.0) + r * math.cos(theta))
            y = float(ego.get("y", 0.0) + r * math.sin(theta))
            z = float(ego.get("z", 0.0) + z_offset)
            yaw = math.degrees(theta)

            obj = {
                "id": new_id,
                "location": {"x": x, "y": y, "z": z},
                "center": {"x": x, "y": y, "z": z},
                "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
                "extent": {"x": float(avg_extent["x"]), "y": float(avg_extent["y"]), "z": float(avg_extent["z"])},
                "speed": avg_speed,
            }

            vehicles.append(obj)
            fake_objects[new_id] = obj

        cpm["detected_vehicles"] = vehicles

        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": new_ids,
            "altered_fields": [
                "detected_vehicles[*].id",
                "detected_vehicles[*].location.*",
                "detected_vehicles[*].center.*",
                "detected_vehicles[*].orientation.*",
                "detected_vehicles[*].extent.*",
            ],
            "ground_truth_reference": {
                "added_objects": fake_objects
            },
        }



# ----------------------------------------------------------------------
# RemoveObjectAttack
# ----------------------------------------------------------------------

class RemoveObjectAttack(Attack):
    """
    Drops some real objects from the CPM.

    Params:
      - target_ids: Optional[List[int]]
      - max_remove_fraction: float in (0,1] (default: 0.5)
      - max_remove_count: Optional[int]
    """

    attack_type = "RemoveObjectAttack"

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = _get_detected_vehicles(cpm)
        if not vehicles:
            return self._empty_metadata()

        all_ids = [v.get("id") for v in vehicles]
        target_ids = self.params.get("target_ids")

        if target_ids is None:
            max_frac = float(self.params.get("max_remove_fraction", 0.5))
            max_frac = min(max(0.0, max_frac), 1.0)
            max_count = int(self.params.get("max_remove_count", len(vehicles)))
            candidate_count = min(max_count, max(1, int(len(vehicles) * max_frac)))
            candidate_count = min(candidate_count, len(vehicles))
            target_ids = random.sample(all_ids, candidate_count)

        target_ids_set = set(target_ids)

        removed_objects: Dict[int, Dict[str, Any]] = {}
        kept: List[Dict[str, Any]] = []
        for v in vehicles:
            vid = v.get("id")
            if vid in target_ids_set:
                removed_objects[int(vid)] = v
            else:
                kept.append(v)

        cpm["detected_vehicles"] = kept

        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": list(target_ids_set),
            "altered_fields": ["detected_vehicles"],
            "ground_truth_reference": {
                "removed_objects": removed_objects
            },
        }

    def _empty_metadata(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": [],
            "altered_fields": [],
            "ground_truth_reference": None,
        }


# ----------------------------------------------------------------------
# Registry / factory
# ----------------------------------------------------------------------

ATTACK_REGISTRY = {
    "DriftAttack": DriftAttack,
    "AddObjectAttack": AddObjectAttack,
    "RemoveObjectAttack": RemoveObjectAttack,
    "WhiteNoiseAttack": WhiteNoiseAttack,
    # "BurstAttack": BurstAttack,  # later
}


def create_attack(attack_type: str, params: Dict[str, Any] | None = None) -> Attack:
    cls = ATTACK_REGISTRY.get(attack_type)
    if cls is None:
        raise ValueError(f"Unknown attack_type: {attack_type}")
    return cls(params=params)
