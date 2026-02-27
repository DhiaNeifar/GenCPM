from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

from .base import Attack, AttackContext
from .helpers import generate_unused_id, get_detected_vehicles


class AddObjectAttack(Attack):
    attack_type = "AddObjectAttack"

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = get_detected_vehicles(cpm)
        ego = cpm.get("global_position", {"x": 0.0, "y": 0.0, "z": 0.0})

        num_objects = int(self.params.get("num_objects", 3))
        radius_min, radius_max = self.params.get("radius_range", (5.0, 20.0))
        z_offset = float(self.params.get("z_offset", 0.0))

        existing_ids = {
            int(v.get("id"))
            for v in vehicles
            if isinstance(v.get("id"), (int, float))
        }

        extent_samples: List[Tuple[float, float, float]] = []
        avg_speed = 0.0
        for vehicle in vehicles:
            extent = vehicle.get("extent")
            avg_speed += vehicle.get("speed", 0.0)
            if not isinstance(extent, dict):
                continue
            ex, ey, ez = extent.get("x"), extent.get("y"), extent.get("z")
            if isinstance(ex, (int, float)) and isinstance(ey, (int, float)) and isinstance(ez, (int, float)):
                extent_samples.append((float(ex), float(ey), float(ez)))

        if vehicles:
            avg_speed /= len(vehicles)

        if extent_samples:
            avg_ex = sum(e[0] for e in extent_samples) / len(extent_samples)
            avg_ey = sum(e[1] for e in extent_samples) / len(extent_samples)
            avg_ez = sum(e[2] for e in extent_samples) / len(extent_samples)
        else:
            avg_ex, avg_ey, avg_ez = 2.0, 1.0, 1.0

        avg_extent = {"x": avg_ex, "y": avg_ey, "z": avg_ez}

        new_ids: List[int] = []
        fake_objects: Dict[int, Dict[str, Any]] = {}

        for _ in range(num_objects):
            new_id = generate_unused_id(existing_ids)
            existing_ids.add(new_id)
            new_ids.append(new_id)

            radius = random.uniform(radius_min, radius_max)
            theta = random.uniform(0.0, 2 * math.pi)
            x = float(ego.get("x", 0.0) + radius * math.cos(theta))
            y = float(ego.get("y", 0.0) + radius * math.sin(theta))
            z = float(ego.get("z", 0.0) + z_offset)
            yaw = math.degrees(theta)

            obj = {
                "id": new_id,
                "location": {"x": x, "y": y, "z": z},
                "center": {"x": x, "y": y, "z": z},
                "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
                "extent": {
                    "x": float(avg_extent["x"]),
                    "y": float(avg_extent["y"]),
                    "z": float(avg_extent["z"]),
                },
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
            "ground_truth_reference": {"added_objects": fake_objects},
        }
