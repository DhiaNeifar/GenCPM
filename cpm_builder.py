#!/usr/bin/env python3
"""
cpm_builder.py

Read a CARLA YAML frame and convert it into a Cooperative Perception Message (CPM)
YAML with a clean, explicit structure.

Usage:
    python cpm_builder.py input_frame.yaml output_cpm.yaml \
        --cpm-type benign --attack None

Notes:
    - All IDs (ego vehicle and detected objects) are remapped to random,
      distinct integers in [id_min, id_max] (defaults: [0, 65535]).
"""

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ----------------------------------------------------------------------
# Basic I/O helpers
# ----------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a Python dictionary.
    """
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """
    Save a Python dictionary as a nicely formatted YAML file.
    """
    with path.open("w") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,          # preserve key order for readability
            default_flow_style=False  # block-style YAML
        )


# ----------------------------------------------------------------------
# Small math / unit helpers
# ----------------------------------------------------------------------

def mps_to_kmh(speed_mps: float) -> float:
    """
    Convert speed from meters per second (m/s) to kilometers per hour (km/h).
    """
    return speed_mps * 3.6


# ----------------------------------------------------------------------
# Random ID helpers
# ----------------------------------------------------------------------

def generate_unique_random_ids(
    num_ids: int,
    id_min: int = 0,
    id_max: int = 65535
) -> List[int]:
    """
    Generate `num_ids` distinct random integers in [id_min, id_max].

    Uses random.sample over the integer range, so all IDs are unique within
    a single CPM.
    """
    population_size = id_max - id_min + 1
    if num_ids > population_size:
        raise ValueError(
            f"Cannot generate {num_ids} unique IDs from range "
            f"[{id_min}, {id_max}] (size={population_size})."
        )

    return random.sample(range(id_min, id_max + 1), num_ids)


def assign_random_ids_to_objects(
    vehicles: List[Dict[str, Any]],
    pedestrians: List[Dict[str, Any]],
    id_min: int = 0,
    id_max: int = 65535
) -> Dict[str, Any]:
    """
    Assign random unique IDs to:
      - ego vehicle (returns its ID)
      - each detected vehicle (overwrites 'id' field)
      - each detected pedestrian (overwrites 'id' field)

    All IDs are distinct in [id_min, id_max].

    Returns:
        {
            "ego_id": <int>,
            "vehicles": <updated vehicles list>,
            "pedestrians": <updated pedestrians list>
        }
    """
    total_ids = 1 + len(vehicles) + len(pedestrians)  # 1 for ego + others
    ids = generate_unique_random_ids(total_ids, id_min=id_min, id_max=id_max)

    ego_id = ids[0]
    vehicle_ids = ids[1:1 + len(vehicles)]
    ped_ids = ids[1 + len(vehicles):]

    # Overwrite vehicle IDs
    for obj, new_id in zip(vehicles, vehicle_ids):
        obj["id"] = new_id

    # Overwrite pedestrian IDs (future use; list is empty for now)
    for obj, new_id in zip(pedestrians, ped_ids):
        obj["id"] = new_id

    return {
        "ego_id": ego_id,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
    }


# ----------------------------------------------------------------------
# Extraction helpers
# ----------------------------------------------------------------------

def extract_ego_position(frame: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract ego global position from the CARLA frame.

    Priority:
    1. true_ego_pos (if available)
    2. lidar_pose   (fallback)
    3. (0, 0, 0)    (if none found)
    """
    pose = frame.get("true_ego_pos") or frame.get("lidar_pose")

    if pose and len(pose) >= 3:
        x, y, z = pose[0], pose[1], pose[2]
    else:
        x, y, z = 0.0, 0.0, 0.0

    return {"x": float(x), "y": float(y), "z": float(z)}


def extract_ego_speed(frame: Dict[str, Any]) -> float:
    """
    Extract ego speed in km/h.

    Assumes 'ego_speed' in the CARLA YAML is given in m/s.
    """
    speed_mps = float(frame.get("ego_speed", 0.0))
    return speed_mps


def parse_vehicle_entry(vehicle_id: str, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single CARLA 'vehicles' entry into a CPM 'detected_vehicles' entry.

    vehicle_data structure (from your example):
      angle:    [pitch, yaw, roll]
      center:   [cx, cy, cz]
      location: [x, y, z]
      extent, speed, ... (ignored here, but you can extend as needed)

    Note: The `id` here will later be overwritten with a random value.
    """
    # Keep original ID just to populate initially; it will be replaced
    try:
        vid = int(vehicle_id)
    except ValueError:
        vid = vehicle_id

    location = vehicle_data.get("location", [0.0, 0.0, 0.0])
    center = vehicle_data.get("center", [0.0, 0.0, 0.0])
    angle = vehicle_data.get("angle", [0.0, 0.0, 0.0])
    extent = vehicle_data.get("extent", [0.0, 0.0, 0.0])
    speed = vehicle_data.get("speed", [0.0])

    # Ensure we have at least 3 elements
    lx, ly, lz = (location + [0.0, 0.0, 0.0])[:3]
    cx, cy, cz = (center + [0.0, 0.0, 0.0])[:3]
    pitch, yaw, roll = (angle + [0.0, 0.0, 0.0])[:3]
    ex, ey, ez = (extent + [0.0, 0.0, 0.0])[:3]
    speed = speed + 0.0
    return {
        "id": vid,  # will be overwritten by random ID later
        "location": {"x": float(lx), "y": float(ly), "z": float(lz)},
        "center": {"x": float(cx), "y": float(cy), "z": float(cz)},
        "extent": {"x": float(ex), "y": float(ey), "z": float(ez)},
        "orientation": {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
        },
        "speed": speed
    }


def extract_detected_vehicles(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the list of detected vehicles from the CARLA frame.

    Uses the 'vehicles' block from your YAML example.
    """
    vehicles_block = frame.get("vehicles", {})
    detected_vehicles: List[Dict[str, Any]] = []

    for vid_str, vehicle_data in vehicles_block.items():
        detected_vehicles.append(parse_vehicle_entry(vid_str, vehicle_data))

    return detected_vehicles


def extract_detected_pedestrians(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detected pedestrians/walkers from the CARLA frame.

    Right now, your sample YAML doesn’t show pedestrians.
    This function is kept here for future extension.

    If your YAML has e.g. 'walkers' or 'pedestrians' blocks with a structure
    similar to 'vehicles', you can reuse parse_vehicle_entry or write a new
    parser for them.

    For now, it returns an empty list.
    """
    # Example placeholder (adapt when you have the actual structure):
    # walkers_block = frame.get("walkers", {})
    # pedestrians = []
    # for pid_str, ped_data in walkers_block.items():
    #     pedestrians.append(parse_ped_entry(pid_str, ped_data))
    # return pedestrians

    return []


# ----------------------------------------------------------------------
# CPM construction
# ----------------------------------------------------------------------

def build_cpm(
    frame: Dict[str, Any],
    cpm_type: str = "benign",
    attack: Optional[Any] = None,
    id_min: int = 0,
    id_max: int = 65535,
) -> Dict[str, Any]:
    """
    Build a CPM dictionary from a CARLA frame dictionary.

    All IDs (ego + detected objects) are replaced by random, unique IDs in
    [id_min, id_max].

    Output structure:

    {
      "vehicle_id": ...,
      "timestamp": ...,
      "vehicle_speed": ...,
      "global_position": { "x": ..., "y": ..., "z": ... },
      "detected_vehicles": [...],
      "detected_pedestrians": [...],
      "cpm_type": "benign" | "attack" | ...,
      "attack": null or attack description
    }
    """
    ego_position = extract_ego_position(frame)
    ego_speed_kmh = extract_ego_speed(frame)
    vehicles = extract_detected_vehicles(frame)
    pedestrians = extract_detected_pedestrians(frame)

    # Assign random unique IDs to ego + all detected objects
    id_assignments = assign_random_ids_to_objects(
        vehicles=vehicles,
        pedestrians=pedestrians,
        id_min=id_min,
        id_max=id_max,
    )
    ego_random_id = id_assignments["ego_id"]
    vehicles = id_assignments["vehicles"]
    pedestrians = id_assignments["pedestrians"]

    # If your input YAML contains a 'timestamp' key, we reuse it; otherwise use 0.
    timestamp = frame.get("timestamp", 0)

    cpm: Dict[str, Any] = {
        "vehicle_id": ego_random_id,
        "timestamp": timestamp,
        "vehicle_speed": ego_speed_kmh,
        "global_position": ego_position,
        "detected_vehicles": vehicles,
        "detected_pedestrians": pedestrians,
        "cpm_type": cpm_type,
        "attack": attack,
    }

    return cpm


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert a CARLA YAML frame into a CPM YAML file."
    )
    parser.add_argument("input", type=str, help="Path to input CARLA YAML frame.")
    parser.add_argument("output", type=str, help="Path to output CPM YAML file.")

    # Kept for compatibility, but not used directly anymore (ego ID is random)
    parser.add_argument(
        "--vehicle-id",
        type=int,
        default=0,
        help="(Ignored) Ego vehicle ID; CPM uses a random ID instead.",
    )
    parser.add_argument(
        "--cpm-type",
        type=str,
        default="benign",
        help='CPM type, e.g., "benign", "attack" (default: "benign").',
    )
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        help="Attack description/name (default: None).",
    )
    parser.add_argument(
        "--id-min",
        type=int,
        default=0,
        help="Minimum random ID value (inclusive, default: 0).",
    )
    parser.add_argument(
        "--id-max",
        type=int,
        default=65535,
        help="Maximum random ID value (inclusive, default: 65535).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible ID assignment (default: None).",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.

    1. Load CARLA YAML frame.
    2. Build CPM dictionary (with random IDs).
    3. Save CPM YAML.
    """
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    in_path = Path(args.input)
    out_path = Path(args.output)

    frame = load_yaml(in_path)

    cpm = build_cpm(
        frame=frame,
        cpm_type=args.cpm_type,
        attack=args.attack,
        id_min=args.id_min,
        id_max=args.id_max,
    )
    save_yaml(cpm, out_path)
    print(f"CPM saved to: {out_path}")


if __name__ == "__main__":
    # python gencpm.py C:\Users\dhian\Desktop\GenCPM\000069.yaml C:\Users\dhian\Desktop\GenCPM\000069_new.yaml
    main()
