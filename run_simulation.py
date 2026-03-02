#!/usr/bin/env python3
"""
run_simulation.py

Skeleton:
- Walk simulation root <p>:
    <p>/
        <veh_id_1>/
            000000.yaml
            000500.yaml
            ...
        <veh_id_2>/
            000000.yaml
            ...

- Infer simulation duration from timestamps.
- For each vehicle:
    * Generate random attack intervals using a Poisson process
    * For each timestamp:
        - load raw frame
        - build benign CPM
        - query AttackScheduler for active attacks
        - apply attack modifications
        - save CPM as <vehicle_id>_<timestamp>.yaml (or repeat-indexed name in parallel mode)
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import load_yaml, save_yaml
from attacks import create_attack, AttackContext, Attack
from tqdm import tqdm


# ----------------------------------------------------------------------
# Attack config & scheduler
# ----------------------------------------------------------------------

@dataclass
class AttackInterval:
    attack_type: str
    start_ts: int      # ticks
    end_ts: int        # ticks
    params: Dict[str, Any] = field(default_factory=dict)
    attack_obj: Optional[Attack] = None   # persistent instance


class AttackScheduler:
    def __init__(self, intervals_per_vehicle: Dict[str, List[AttackInterval]]) -> None:
        self.intervals_per_vehicle = intervals_per_vehicle

    def get_attacks_for(self, vehicle_id: str, timestamp: int) -> List[AttackInterval]:
        """
        Return all AttackInterval objects active at this (vehicle_id, timestamp).
        timestamp is in ticks.
        """
        intervals = self.intervals_per_vehicle.get(vehicle_id, [])
        return [
            iv
            for iv in intervals
            if iv.start_ts <= timestamp <= iv.end_ts
        ]


def _pick_single_attack(active_attacks: List[AttackInterval]) -> Optional[AttackInterval]:
    if not active_attacks:
        return None
    # Deterministic tie-break: latest start_ts wins, then lexical attack type.
    return sorted(
        active_attacks,
        key=lambda iv: (iv.start_ts, iv.attack_type),
    )[-1]


# ----------------------------------------------------------------------
# Random attack schedule (Poisson-style)
# ----------------------------------------------------------------------

def generate_poisson_intervals_for_vehicle(
    sim_duration_ticks: int,
    dt: float,                    # time step in seconds (0.05)
    lambda_attacks: float,        # rate: attacks per second
    mean_duration: float,         # mean attack duration in seconds
    attack_type_probs: Dict[str, float],
) -> List[AttackInterval]:
    """
    Generate random attack intervals for a single vehicle.

    - Start times follow a Poisson process (exp inter-arrival with rate lambda_attacks).
    - Durations ~ Exp(mean_duration).
    - Attack types drawn from attack_type_probs.
    - All timestamps converted to integer ticks.
    """
    if lambda_attacks <= 0:
        return []

    types = list(attack_type_probs.keys())
    weights = list(attack_type_probs.values())

    intervals: List[AttackInterval] = []
    
    # Generate in seconds first (Poisson process)
    t_seconds = 0.0
    while True:
        gap = random.expovariate(lambda_attacks)
        t_seconds += gap
        t_tick = int(round(t_seconds / dt))
        if t_tick >= sim_duration_ticks:
            break
       
        duration_seconds = random.expovariate(1.0 / mean_duration)
        end_tick = min(t_tick + int(round(duration_seconds / dt)), sim_duration_ticks)

        attack_type = random.choices(types, weights=weights, k=1)[0]
    
        # After selecting attack_type, generate params:
        params: Dict[str, Any] = {}
        
        if attack_type == "DriftAttack":
            params = {
                "mean_step": 2.0,
                "std_step": 0.8,
                "yaw_step_mean": 10.0,
                "yaw_step_std": 5.0,
                "num_targets": 10000,
            }
            
        elif attack_type == "AddObjectAttack":
            params = {
                "num_objects": 25,
                "radius_range": (2.0, 40.0),
                "z_offset": 0.0,
            }
            
        elif attack_type == "RemoveObjectAttack":
            params = {
                "max_remove_fraction": 0.95,
                "max_remove_count": 10000,
            }
            
        elif attack_type == "WhiteNoiseAttack":
            params = {
                "mean_drift": 3.0,
                "std_drift": 1.5,
                "z_drift_mean": 0.0,
                "z_drift_std": 0.5,
                "yaw_drift_mean": 15.0,
                "yaw_drift_std": 8.0,
                "pitch_drift_mean": 5.0,
                "pitch_drift_std": 3.0,
                "num_targets": 10000,
            }

        intervals.append(
            AttackInterval(
                attack_type=attack_type,
                start_ts=t_tick,
                end_ts=end_tick,
                params=params,
            )
        )

    return intervals


def build_random_schedule_for_all_vehicles(
    root_dir: Path,
    sim_duration_ticks: int,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
) -> Dict[str, List[AttackInterval]]:
    """
    For each vehicle folder under root_dir, generate its own random attack intervals.
    """
    intervals_per_vehicle: Dict[str, List[AttackInterval]] = {}

    for vehicle_dir in sorted(root_dir.iterdir()):
        if not vehicle_dir.is_dir():
            continue
        veh_id = vehicle_dir.name
        intervals_per_vehicle[veh_id] = generate_poisson_intervals_for_vehicle(
            sim_duration_ticks=sim_duration_ticks,
            dt=dt,
            lambda_attacks=lambda_attacks,
            mean_duration=mean_duration,
            attack_type_probs=attack_type_probs,
        )

    return intervals_per_vehicle


def parse_timestamp_from_preds_filename(path: Path) -> int:
    name = path.name
    suffix = "_preds.yaml"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected preds filename format: {name}")
    return int(name[: -len(suffix)])


def list_preds_paths(vehicle_dir: Path) -> List[Path]:
    return sorted(
        [p for p in vehicle_dir.glob("*_preds.yaml") if p.is_file()],
        key=parse_timestamp_from_preds_filename,
    )


def _to_float3(value: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(value, dict):
        return (
            float(value.get("x", default[0])),
            float(value.get("y", default[1])),
            float(value.get("z", default[2])),
        )
    if isinstance(value, list):
        padded = (value + [default[0], default[1], default[2]])[:3]
        return float(padded[0]), float(padded[1]), float(padded[2])
    return default


def extract_detected_vehicles_from_preds(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    detected = frame.get("detected_objects", {})
    if not isinstance(detected, dict):
        return []

    vehicles: List[Dict[str, Any]] = []
    for key, obj in detected.items():
        if not isinstance(obj, dict):
            continue
        # Normalize IDs to string so generated CPM labels are consistent.
        vid = str(obj.get("id", obj.get("object_id", key)))

        lx, ly, lz = _to_float3(obj.get("location"), (0.0, 0.0, 0.0))
        cx, cy, cz = _to_float3(obj.get("center"), (lx, ly, lz))
        ex, ey, ez = _to_float3(obj.get("extent"), (2.0, 1.0, 1.0))
        pitch, yaw, roll = _to_float3(obj.get("angle"), (0.0, 0.0, 0.0))
        if "orientation" in obj:
            pitch, yaw, roll = _to_float3(obj.get("orientation"), (pitch, yaw, roll))

        speed_raw = obj.get("speed", 0.0)
        if isinstance(speed_raw, list):
            speed = float(speed_raw[0]) if speed_raw else 0.0
        else:
            speed = float(speed_raw)

        vehicles.append(
            {
                "id": vid,
                "location": {"x": lx, "y": ly, "z": lz},
                "center": {"x": cx, "y": cy, "z": cz},
                "extent": {"x": ex, "y": ey, "z": ez},
                "orientation": {"pitch": pitch, "yaw": yaw, "roll": roll},
                "speed": speed,
            }
        )

    return vehicles


def build_cpm_from_preds_frame(frame: Dict[str, Any], vehicle_id: str, timestamp: int, cpm_type: str) -> Dict[str, Any]:
    lidar_pose = frame.get("lidar_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ego_speed = float(frame.get("ego_speed", 0.0))
    detected_vehicles = extract_detected_vehicles_from_preds(frame)

    return {
        "vehicle_id": vehicle_id,
        "timestamp": timestamp,
        "vehicle_speed": ego_speed,
        "global_position": {
            "x": float(lidar_pose[0]) if len(lidar_pose) > 0 else 0.0,
            "y": float(lidar_pose[1]) if len(lidar_pose) > 1 else 0.0,
            "z": float(lidar_pose[2]) if len(lidar_pose) > 2 else 0.0,
        },
        "detected_vehicles": detected_vehicles,
        "detected_objects": {},
        "detected_pedestrians": [],
        "cpm_type": cpm_type,
        "attacks": [],
        "source_file_type": "_preds.yaml",
        "source_field": "detected_objects",
    }


def vehicles_to_detected_objects(vehicles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    detected_objects: Dict[str, Dict[str, Any]] = {}
    for vehicle in vehicles:
        vid = vehicle.get("id")
        vid_str = str(vid)
        loc = vehicle.get("location", {})
        ori = vehicle.get("orientation", {})
        center = vehicle.get("center", loc)
        extent = vehicle.get("extent", {"x": 2.0, "y": 1.0, "z": 1.0})
        if isinstance(center, dict):
            center_out = [
                float(center.get("x", 0.0)),
                float(center.get("y", 0.0)),
                float(center.get("z", 0.0)),
            ]
        elif isinstance(center, list):
            center_pad = (center + [0.0, 0.0, 0.0])[:3]
            center_out = [float(center_pad[0]), float(center_pad[1]), float(center_pad[2])]
        else:
            center_out = [0.0, 0.0, 0.0]

        if isinstance(extent, dict):
            extent_out = [
                float(extent.get("x", 2.0)),
                float(extent.get("y", 1.0)),
                float(extent.get("z", 1.0)),
            ]
        elif isinstance(extent, list):
            extent_pad = (extent + [2.0, 1.0, 1.0])[:3]
            extent_out = [float(extent_pad[0]), float(extent_pad[1]), float(extent_pad[2])]
        else:
            extent_out = [2.0, 1.0, 1.0]

        detected_objects[vid_str] = {
            "angle": [
                float(ori.get("pitch", 0.0)),
                float(ori.get("yaw", 0.0)),
                float(ori.get("roll", 0.0)),
            ],
            "center": center_out,
            "extent": extent_out,
            "location": [
                float(loc.get("x", 0.0)),
                float(loc.get("y", 0.0)),
                float(loc.get("z", 0.0)),
            ],
            "speed": float(vehicle.get("speed", 0.0)),
        }

    return detected_objects


# ----------------------------------------------------------------------
# Attack application
# ----------------------------------------------------------------------

def apply_attacks_to_cpm(
    cpm: Dict[str, Any],
    vehicle_id: str,
    timestamp: int,
    active_attacks: List[AttackInterval],
) -> Dict[str, Any]:
    """
    Given a benign CPM and active intervals, apply at most one attack.

    - Reuses the same attack_obj instance across frames within an interval.
    - Enforces single-attack-per-CPM even if intervals overlap.
    - Forces AddObjectAttack chaos mode by replacing existing detections.
    """
    selected_attack = _pick_single_attack(active_attacks)
    if selected_attack is None:
        cpm["cpm_type"] = "benign"
        cpm["attacks"] = []
        return cpm

    cpm["cpm_type"] = "malicious"
    ctx = AttackContext(vehicle_id=str(vehicle_id), timestamp=timestamp)
    if selected_attack.attack_type == "AddObjectAttack":
        # Force AddObjectAttack chaos mode: replace current detections.
        cpm["detected_vehicles"] = []

    if selected_attack.attack_obj is None:
        selected_attack.attack_obj = create_attack(selected_attack.attack_type, selected_attack.params)

    meta = selected_attack.attack_obj.apply(cpm, ctx)
    meta["vehicle_id"] = vehicle_id
    meta["timestamp"] = timestamp
    cpm["attacks"] = [meta]
    return cpm


# ----------------------------------------------------------------------
# Simulation traversal
# ----------------------------------------------------------------------

def infer_sim_duration(root_dir: Path) -> int:
    """
    Look at all *_preds.yaml timestamp filenames and infer max tick index.
    Returns the maximum tick index found.
    """
    max_idx = 0
    for vehicle_dir in root_dir.iterdir():
        if not vehicle_dir.is_dir():
            continue
        for preds_path in vehicle_dir.glob("*_preds.yaml"):
            try:
                idx = parse_timestamp_from_preds_filename(preds_path)
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue

    return max_idx


def process_vehicle_folder(
    vehicle_dir: Path,
    output_dir: Path,
    attack_scheduler: AttackScheduler,
    repeat_idx: Optional[int] = None,
) -> None:
    veh_id = vehicle_dir.name

    preds_files = list_preds_paths(vehicle_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for preds_path in tqdm(preds_files, desc=f"{veh_id}", leave=False, unit="frame"):
        sim_tick = parse_timestamp_from_preds_filename(preds_path)

        frame = load_yaml(preds_path)

        # Build CPM and then label it benign/malicious based on active attacks.
        cpm = build_cpm_from_preds_frame(
            frame=frame,
            vehicle_id=veh_id,
            timestamp=sim_tick,
            cpm_type="benign",
        )

        # Query random schedule
        active_attacks = attack_scheduler.get_attacks_for(veh_id, sim_tick)

        # Apply one selected attack (if any).
        cpm = apply_attacks_to_cpm(
            cpm=cpm,
            vehicle_id=veh_id,
            timestamp=sim_tick,
            active_attacks=active_attacks,
        )

        cpm["detected_objects"] = vehicles_to_detected_objects(cpm.get("detected_vehicles", []))

        if repeat_idx is None:
            out_name = f"{veh_id}_{sim_tick:06d}.yaml"
        else:
            out_name = f"r{repeat_idx:03d}_{veh_id}_{sim_tick:06d}.yaml"
        save_yaml(cpm, output_dir / out_name)


def run_simulation(
    root_dir: Path,
    output_dir: Path,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
    seed: Optional[int] = None,
    repeat_idx: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) infer total duration in ticks
    sim_duration_ticks = infer_sim_duration(root_dir)

    # 2) build random schedule per vehicle
    intervals_per_vehicle = build_random_schedule_for_all_vehicles(
        root_dir=root_dir,
        sim_duration_ticks=sim_duration_ticks,
        dt=dt,
        lambda_attacks=lambda_attacks,
        mean_duration=mean_duration,
        attack_type_probs=attack_type_probs,
    )

    # 2bis) save simulation config / metadata
    sim_config: Dict[str, Any] = {
        "simulation_name": root_dir.name,
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "dt": dt,
        "lambda_attacks": lambda_attacks,
        "mean_duration": mean_duration,
        "attack_type_probs": attack_type_probs,
        "source_file_pattern": "*_preds.yaml",
        "source_object_field": "detected_objects",
        "one_attack_per_cpm": True,
        "outputs": {
            "single_output_dir": ".",
            "label_field": "cpm_type",
            "label_values": ["benign", "malicious"],
        },
        "seed": seed,
        "repeat_idx": repeat_idx,
        "sim_duration_ticks": sim_duration_ticks,
        "vehicles": {},
    }

    for veh_id, intervals in intervals_per_vehicle.items():
        sim_config["vehicles"][veh_id] = [
            {
                "attack_type": iv.attack_type,
                "start_ts": iv.start_ts,
                "end_ts": iv.end_ts,
                "params": iv.params,
            }
            for iv in intervals
        ]

    if repeat_idx is None:
        config_path = output_dir / "simulation_config.yaml"
    else:
        config_path = output_dir / f"simulation_config_r{repeat_idx:03d}.yaml"
    save_yaml(sim_config, config_path)

    # 3) create scheduler and generate CPMs
    attack_scheduler = AttackScheduler(intervals_per_vehicle=intervals_per_vehicle)

    # 4) generate CPMs
    for vehicle_dir in sorted(root_dir.iterdir()):
        if not vehicle_dir.is_dir():
            continue
        process_vehicle_folder(
            vehicle_dir=vehicle_dir,
            output_dir=output_dir,
            attack_scheduler=attack_scheduler,
            repeat_idx=repeat_idx,
        )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random attack simulation over CPMs using a Poisson process."
    )
    parser.add_argument("root", type=str, help="Path <p> to simulation root.")
    parser.add_argument("output", type=str, help="Output folder for CPMs.")

    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step between frames in seconds (default: 0.05).")
    parser.add_argument("--lambda-attacks", type=float, default=1/10,
                        help="Attack rate (attacks per second) per vehicle (default: 1/60 ≈ 1 per minute).")
    parser.add_argument("--mean-duration", type=float, default=20.0,
                        help="Mean attack duration in seconds (default: 10s).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root)
    output_dir = Path(args.output)

    attack_type_probs = {
        "DriftAttack": 0.0,
        "AddObjectAttack": 1.0,
        "RemoveObjectAttack": 0.0,
        "WhiteNoiseAttack": 0.0,
    }

    run_simulation(
        root_dir=root_dir,
        output_dir=output_dir,
        dt=args.dt,
        lambda_attacks=args.lambda_attacks,
        mean_duration=args.mean_duration,
        attack_type_probs=attack_type_probs,
        seed=args.seed,
    )


if __name__ == "__main__":
    # python run_simulation.py "/Users/dhianeifar/Desktop/Projects/CPM/Final Figure/2021_08_16_22_26_54" "/Users/dhianeifar/Desktop/Projects/CPM/Final Figure/CF"
    # python run_simulation.py "C:\Users\dhian\PycharmProjects\AdverCPM\experiments\raw simulations\2021_08_16_22_26_54" "C:\Users\dhian\Desktop\GenCPM\CPM collected"
    main()
