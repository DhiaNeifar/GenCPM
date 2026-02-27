from __future__ import annotations

import random
from typing import Any, Dict, List, Set


def get_detected_vehicles(cpm: Dict[str, Any]) -> List[Dict[str, Any]]:
    return cpm.get("detected_vehicles", [])


def generate_unused_id(existing_ids: Set[int], id_min: int = 0, id_max: int = 65535) -> int:
    while True:
        new_id = random.randint(id_min, id_max)
        if new_id not in existing_ids:
            return new_id
