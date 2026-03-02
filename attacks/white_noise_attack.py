from __future__ import annotations

import math
import random
from typing import Any, Dict, List

from .base import Attack, AttackContext
from .helpers import get_detected_vehicles


class WhiteNoiseAttack(Attack):
    attack_type = "WhiteNoiseAttack"

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = get_detected_vehicles(cpm)
        if not vehicles:
            return self._empty_metadata()

        mean_drift = float(self.params.get("mean_drift", 0.5))
        std_drift = float(self.params.get("std_drift", 0.2))
        z_mean = float(self.params.get("z_drift_mean", 0.0))
        z_std = float(self.params.get("z_drift_std", 0.05))
        yaw_mean = float(self.params.get("yaw_drift_mean", 0.0))
        yaw_std = float(self.params.get("yaw_drift_std", 5.0))
        pitch_mean = float(self.params.get("pitch_drift_mean", 0.0))
        pitch_std = float(self.params.get("pitch_drift_std", 2.0))

        all_ids = [v.get("id") for v in vehicles]
        target_ids = self.params.get("target_ids")

        if target_ids is None:
            num_targets = int(self.params.get("num_targets", max(1, len(vehicles) // 3)))
            num_targets = max(1, min(num_targets, len(vehicles)))
            target_ids = random.sample(all_ids, num_targets)

        target_ids_set = set(target_ids)

        ground_truth: Dict[Any, Dict[str, Any]] = {}
        altered_fields: List[str] = []

        for vehicle in vehicles:
            vehicle_id = vehicle.get("id")
            if vehicle_id not in target_ids_set:
                continue

            location = vehicle.setdefault("location", {})
            orientation = vehicle.setdefault("orientation", {})

            gt_key: Any
            try:
                gt_key = int(str(vehicle_id))
            except (TypeError, ValueError):
                gt_key = vehicle_id

            ground_truth[gt_key] = {
                "location": dict(location),
                "orientation": dict(orientation),
            }

            radius = max(0.0, random.gauss(mean_drift, std_drift))
            theta = random.uniform(0.0, 2 * math.pi)
            dx = radius * math.cos(theta)
            dy = radius * math.sin(theta)

            location["x"] = float(location.get("x", 0.0) + dx)
            location["y"] = float(location.get("y", 0.0) + dy)

            if z_std != 0.0 or z_mean != 0.0:
                dz = random.gauss(z_mean, z_std)
                location["z"] = float(location.get("z", 0.0) + dz)
                altered_fields.extend(["location.x", "location.y", "location.z"])
            else:
                altered_fields.extend(["location.x", "location.y"])

            if yaw_std != 0.0 or yaw_mean != 0.0:
                dyaw = random.gauss(yaw_mean, yaw_std)
                orientation["yaw"] = float(orientation.get("yaw", 0.0) + dyaw)
                altered_fields.append("orientation.yaw")

            if pitch_std != 0.0 or pitch_mean != 0.0:
                dpitch = random.gauss(pitch_mean, pitch_std)
                orientation["pitch"] = float(orientation.get("pitch", 0.0) + dpitch)
                altered_fields.append("orientation.pitch")

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
