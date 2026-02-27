from __future__ import annotations

import math
import random
from typing import Any, Dict, List

from .base import Attack, AttackContext
from .helpers import get_detected_vehicles


class DriftAttack(Attack):
    attack_type = "DriftAttack"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._offsets: Dict[int, Dict[str, float]] = {}

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = get_detected_vehicles(cpm)
        if not vehicles:
            return self._empty_metadata()

        mean_step = float(self.params.get("mean_step", 0.05))
        std_step = float(self.params.get("std_step", 0.02))
        yaw_mean = float(self.params.get("yaw_step_mean", 0.0))
        yaw_std = float(self.params.get("yaw_step_std", 0.0))

        all_ids = [v.get("id") for v in vehicles]
        target_ids = self.params.get("target_ids")

        if target_ids is None:
            num_targets = int(self.params.get("num_targets", max(1, len(vehicles) // 3)))
            num_targets = max(1, min(num_targets, len(vehicles)))
            target_ids = random.sample(all_ids, num_targets)

        target_ids_set = set(target_ids)

        ground_truth: Dict[int, Dict[str, Any]] = {}
        altered_fields: List[str] = []

        for vehicle in vehicles:
            vehicle_id = vehicle.get("id")
            if vehicle_id not in target_ids_set:
                continue

            try:
                vehicle_id_int = int(vehicle_id)
            except (TypeError, ValueError):
                vehicle_id_int = None

            location = vehicle.setdefault("location", {})
            orientation = vehicle.setdefault("orientation", {})

            gt_location = dict(location)
            gt_orientation = dict(orientation)
            if vehicle_id_int is not None:
                ground_truth[vehicle_id_int] = {
                    "location": gt_location,
                    "orientation": gt_orientation,
                }

            if vehicle_id_int is not None:
                offset = self._offsets.setdefault(
                    vehicle_id_int, {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}
                )
            else:
                offset = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}

            step_r = max(0.0, random.gauss(mean_step, std_step))
            theta = random.uniform(0.0, 2 * math.pi)
            step_dx = step_r * math.cos(theta)
            step_dy = step_r * math.sin(theta)

            offset["dx"] += step_dx
            offset["dy"] += step_dy

            location["x"] = float(gt_location.get("x", 0.0) + offset["dx"])
            location["y"] = float(gt_location.get("y", 0.0) + offset["dy"])

            if yaw_std != 0.0 or yaw_mean != 0.0:
                step_dyaw = random.gauss(yaw_mean, yaw_std)
                offset["dyaw"] += step_dyaw
                orientation["yaw"] = float(gt_orientation.get("yaw", 0.0) + offset["dyaw"])
                altered_fields.extend(["location.x", "location.y", "orientation.yaw"])
            else:
                altered_fields.extend(["location.x", "location.y"])

            if vehicle_id_int is not None:
                self._offsets[vehicle_id_int] = offset

        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": list(target_ids_set),
            "altered_fields": sorted(set(altered_fields)),
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
