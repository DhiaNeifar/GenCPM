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

        existing_ids = set()
        for vehicle in vehicles:
            raw_id = vehicle.get("id")
            try:
                existing_ids.add(int(str(raw_id)))
            except (TypeError, ValueError):
                continue

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
            # Half-sizes of a typical passenger car (l/2, w/2, h/2), matching real detections.
            avg_ex, avg_ey, avg_ez = 2.25, 0.95, 0.75

        avg_extent = {"x": avg_ex, "y": avg_ey, "z": avg_ez}

        # Fake boxes must use the same convention as real detections: `location` is the box bottom
        # on the ground in world coordinates and `center` is the local offset [0, 0, h/2].
        # global_position is the LiDAR pose (~1.9 m above ground), so the ground height comes from
        # the real objects in this CPM, or the LiDAR height when there are none.
        real_z = sorted(float((v.get("location") or {}).get("z", 0.0)) for v in vehicles)
        if real_z:
            ground_z = real_z[len(real_z) // 2]
        else:
            ground_z = float(ego.get("z", 0.0)) - float(self.params.get("lidar_height", 1.9))

        new_ids: List[Any] = []
        fake_objects: Dict[Any, Dict[str, Any]] = {}

        for _ in range(num_objects):
            new_id = generate_unused_id(existing_ids)
            existing_ids.add(new_id)

            radius = random.uniform(radius_min, radius_max)
            theta = random.uniform(0.0, 2 * math.pi)
            x = float(ego.get("x", 0.0) + radius * math.cos(theta))
            y = float(ego.get("y", 0.0) + radius * math.sin(theta))
            z = float(ground_z + z_offset)
            yaw = math.degrees(theta)

            output_id: Any = str(new_id)
            obj = {
                "id": output_id,
                "location": {"x": x, "y": y, "z": z},
                "center": {"x": 0.0, "y": 0.0, "z": float(avg_extent["z"])},
                "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
                "extent": {
                    "x": float(avg_extent["x"]),
                    "y": float(avg_extent["y"]),
                    "z": float(avg_extent["z"]),
                },
                "speed": avg_speed,
            }

            vehicles.append(obj)
            new_ids.append(output_id)
            fake_objects[output_id] = obj

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
