#!/usr/bin/env python3
"""
cpm_builder.py

Read a CARLA YAML frame and convert it into a Cooperative Perception Message (CPM).
"""

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import assign_random_ids_to_objects, generate_unique_random_ids, load_yaml, save_yaml


# Re-export utility functions for backward compatibility.
__all__ = [
    "load_yaml",
    "save_yaml",
    "generate_unique_random_ids",
    "assign_random_ids_to_objects",
    "extract_ego_position",
    "extract_ego_speed",
    "parse_vehicle_entry",
    "extract_detected_vehicles",
    "extract_detected_pedestrians",
    "build_cpm",
]


def extract_ego_position(frame: Dict[str, Any]) -> Dict[str, float]:
    pose = frame.get("true_ego_pos") or frame.get("lidar_pose")

    if pose and len(pose) >= 3:
        x, y, z = pose[0], pose[1], pose[2]
    else:
        x, y, z = 0.0, 0.0, 0.0

    return {"x": float(x), "y": float(y), "z": float(z)}


def extract_ego_speed(frame: Dict[str, Any]) -> float:
    return float(frame.get("ego_speed", 0.0))


def parse_vehicle_entry(vehicle_id: str, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        vid = int(vehicle_id)
    except ValueError:
        vid = vehicle_id

    location = vehicle_data.get("location", [0.0, 0.0, 0.0])
    center = vehicle_data.get("center", [0.0, 0.0, 0.0])
    angle = vehicle_data.get("angle", [0.0, 0.0, 0.0])
    extent = vehicle_data.get("extent", [0.0, 0.0, 0.0])

    raw_speed = vehicle_data.get("speed", 0.0)
    if isinstance(raw_speed, list):
        speed = float(raw_speed[0]) if raw_speed else 0.0
    else:
        speed = float(raw_speed)

    lx, ly, lz = (location + [0.0, 0.0, 0.0])[:3]
    cx, cy, cz = (center + [0.0, 0.0, 0.0])[:3]
    pitch, yaw, roll = (angle + [0.0, 0.0, 0.0])[:3]
    ex, ey, ez = (extent + [0.0, 0.0, 0.0])[:3]

    return {
        "id": vid,
        "location": {"x": float(lx), "y": float(ly), "z": float(lz)},
        "center": {"x": float(cx), "y": float(cy), "z": float(cz)},
        "extent": {"x": float(ex), "y": float(ey), "z": float(ez)},
        "orientation": {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
        },
        "speed": speed,
    }


def extract_detected_vehicles(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    vehicles_block = frame.get("vehicles", {})
    detected_vehicles: List[Dict[str, Any]] = []

    for vid_str, vehicle_data in vehicles_block.items():
        detected_vehicles.append(parse_vehicle_entry(vid_str, vehicle_data))

    return detected_vehicles


def extract_detected_pedestrians(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    return []


def build_cpm(
    frame: Dict[str, Any],
    cpm_type: str = "benign",
    attack: Optional[Any] = None,
    id_min: int = 0,
    id_max: int = 65535,
) -> Dict[str, Any]:
    ego_position = extract_ego_position(frame)
    ego_speed = extract_ego_speed(frame)
    vehicles = extract_detected_vehicles(frame)
    pedestrians = extract_detected_pedestrians(frame)

    id_assignments = assign_random_ids_to_objects(
        vehicles=vehicles,
        pedestrians=pedestrians,
        id_min=id_min,
        id_max=id_max,
    )

    timestamp = frame.get("timestamp", 0)

    cpm: Dict[str, Any] = {
        "vehicle_id": id_assignments["ego_id"],
        "timestamp": timestamp,
        "vehicle_speed": ego_speed,
        "global_position": ego_position,
        "detected_vehicles": id_assignments["vehicles"],
        "detected_pedestrians": id_assignments["pedestrians"],
        "cpm_type": cpm_type,
        "attack": attack,
    }

    return cpm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CARLA YAML frame into a CPM YAML file."
    )
    parser.add_argument("input", type=str, help="Path to input CARLA YAML frame.")
    parser.add_argument("output", type=str, help="Path to output CPM YAML file.")

    parser.add_argument(
        "--vehicle-id",
        type=int,
        default=0,
        help="Ignored. CPM uses a random ID instead.",
    )
    parser.add_argument(
        "--cpm-type",
        type=str,
        default="benign",
        help='CPM type, e.g., "benign" or "attack".',
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
        help="Minimum random ID value (inclusive).",
    )
    parser.add_argument(
        "--id-max",
        type=int,
        default=65535,
        help="Maximum random ID value (inclusive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible ID assignment.",
    )

    return parser.parse_args()


def main() -> None:
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
    main()
