from __future__ import annotations

import random
from typing import Any, Dict, List

from .base import Attack, AttackContext
from .helpers import get_detected_vehicles


class RemoveObjectAttack(Attack):
    attack_type = "RemoveObjectAttack"

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        vehicles = get_detected_vehicles(cpm)
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

        removed_objects: Dict[Any, Dict[str, Any]] = {}
        kept: List[Dict[str, Any]] = []
        for vehicle in vehicles:
            vehicle_id = vehicle.get("id")
            if vehicle_id in target_ids_set:
                removed_objects[vehicle_id] = vehicle
            else:
                kept.append(vehicle)

        cpm["detected_vehicles"] = kept

        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": list(target_ids_set),
            "altered_fields": ["detected_vehicles"],
            "ground_truth_reference": {"removed_objects": removed_objects},
        }

    def _empty_metadata(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "attack_parameters": self.params,
            "target_ids": [],
            "altered_fields": [],
            "ground_truth_reference": None,
        }
