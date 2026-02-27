from __future__ import annotations

import random
from typing import Any, Dict, List


def generate_unique_random_ids(num_ids: int, id_min: int = 0, id_max: int = 65535) -> List[int]:
    population_size = id_max - id_min + 1
    if num_ids > population_size:
        raise ValueError(
            f"Cannot generate {num_ids} unique IDs from range [{id_min}, {id_max}] (size={population_size})."
        )
    return random.sample(range(id_min, id_max + 1), num_ids)


def assign_random_ids_to_objects(
    vehicles: List[Dict[str, Any]],
    pedestrians: List[Dict[str, Any]],
    id_min: int = 0,
    id_max: int = 65535,
) -> Dict[str, Any]:
    total_ids = 1 + len(vehicles) + len(pedestrians)
    ids = generate_unique_random_ids(total_ids, id_min=id_min, id_max=id_max)

    ego_id = ids[0]
    vehicle_ids = ids[1 : 1 + len(vehicles)]
    ped_ids = ids[1 + len(vehicles) :]

    for obj, new_id in zip(vehicles, vehicle_ids):
        obj["id"] = new_id

    for obj, new_id in zip(pedestrians, ped_ids):
        obj["id"] = new_id

    return {
        "ego_id": ego_id,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
    }
