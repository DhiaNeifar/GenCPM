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
        - save CPM as <vehicle_id>_<timestamp>.yaml
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cpm_builder import build_cpm, load_yaml, save_yaml
from attack import create_attack, AttackContext, Attack
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
            mean_step = random.uniform(0.03, 0.07)  # m per frame
            std_step = random.uniform(0.01, 0.03)   # m per frame
            yaw_mean = random.uniform(-0.5, 0.5)    # deg per frame
            yaw_std = random.uniform(0.0, 0.3)      # deg per frame

            params = {
                "mean_step": mean_step,
                "std_step": std_step,
                "yaw_step_mean": yaw_mean,
                "yaw_step_std": yaw_std
            }
            
        elif attack_type == "AddObjectAttack":
            num_objects = random.randint(7, 10)
            r_min = random.uniform(5.0, 10.0)
            r_max = random.uniform(15.0, 25.0)
            z_offset = random.uniform(-0.5, 0.5)
            
            params = {
                "num_objects": num_objects,
                "radius_range": (r_min, r_max),
                "z_offset": z_offset
            }
            
        elif attack_type == "RemoveObjectAttack":
            # target_ids will be determined dynamically when applying the attack
            params = {
                "target_ids": None  # Will be populated during attack application
            }
            
        elif attack_type == "WhiteNoiseAttack":
            mean_drift = random.uniform(0.3, 0.7)
            std_drift = random.uniform(0.1, 0.3)
            z_mean = random.uniform(-0.05, 0.05)
            z_std = random.uniform(0.02, 0.08)
            yaw_mean = random.uniform(-2.0, 2.0)
            yaw_std = random.uniform(3.0, 7.0)
            pitch_mean = random.uniform(-1.0, 1.0)
            pitch_std = random.uniform(1.0, 3.0)
            
            params = {
                "mean_drift": mean_drift,
                "std_drift": std_drift,
                "z_drift_mean": z_mean,
                "z_drift_std": z_std,
                "yaw_drift_mean": yaw_mean,
                "yaw_drift_std": yaw_std,
                "pitch_drift_mean": pitch_mean,
                "pitch_drift_std": pitch_std
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
    Given a benign CPM and the list of active attacks, return a modified CPM.

    - Reuses the same attack_obj instance across frames within an interval,
      so stateful attacks (e.g., DriftAttack) can accumulate effects.
    - For RemoveObjectAttack, dynamically select target_ids from detected_vehicles.
    """
    if not active_attacks:
        cpm["cpm_type"] = "benign"
        cpm["attacks"] = []
        return cpm

    cpm["cpm_type"] = "malicious"
    cpm_attacks: List[Dict[str, Any]] = []

    ctx = AttackContext(vehicle_id=str(vehicle_id), timestamp=timestamp)

    for interval in active_attacks:
        # Handle RemoveObjectAttack target_ids dynamically
        if interval.attack_type == "RemoveObjectAttack" and interval.params.get("target_ids") is None:
            detected_vehicles = cpm.get("detected_vehicles", [])
            if detected_vehicles:
                # Randomly select 1 to min(3, total) vehicles to remove
                num_to_remove = random.randint(1, min(3, len(detected_vehicles)))
                target_ids = random.sample(range(len(detected_vehicles)), num_to_remove)
                interval.params["target_ids"] = target_ids
            else:
                interval.params["target_ids"] = []
        
        # create once, reuse later
        if interval.attack_obj is None:
            interval.attack_obj = create_attack(interval.attack_type, interval.params)

        attack = interval.attack_obj
        meta = attack.apply(cpm, ctx)
        meta["vehicle_id"] = vehicle_id
        meta["timestamp"] = timestamp
        cpm_attacks.append(meta)

    cpm["attacks"] = cpm_attacks
    return cpm


# ----------------------------------------------------------------------
# Simulation traversal
# ----------------------------------------------------------------------

def parse_timestamp_from_filename(stem: str) -> int:
    """
    Convert filename stem (e.g. '000069') to a numeric tick index.
    """
    return int(stem)


def infer_sim_duration(root_dir: Path) -> int:
    """
    Look at all timestamp filenames and infer the total simulation duration in ticks.
    Returns the maximum tick index found.
    """
    max_idx = 0
    for vehicle_dir in root_dir.iterdir():
        if not vehicle_dir.is_dir():
            continue
        for yaml_path in vehicle_dir.glob("*.yaml"):
            try:
                idx = parse_timestamp_from_filename(yaml_path.stem)
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue

    return max_idx


def process_vehicle_folder(
    vehicle_dir: Path,
    output_dir: Path,
    attack_scheduler: AttackScheduler,
    dt: float,
) -> None:
    veh_id = vehicle_dir.name

    yaml_files = sorted(
        [p for p in vehicle_dir.glob("*.yaml") if p.is_file()],
        key=lambda p: p.stem,
    )

    for yaml_path in tqdm(yaml_files, desc=f"{veh_id}", leave=False, unit="frame"):
        sim_tick = parse_timestamp_from_filename(yaml_path.stem)

        frame = load_yaml(yaml_path)

        # Build benign CPM
        cpm = build_cpm(
            frame=frame,
            cpm_type="benign",
            attack=None,
        )

        cpm["timestamp"] = sim_tick
        
        # Query random schedule
        active_attacks = attack_scheduler.get_attacks_for(veh_id, sim_tick)

        # Apply attacks
        cpm = apply_attacks_to_cpm(
            cpm=cpm,
            vehicle_id=veh_id,
            timestamp=sim_tick,
            active_attacks=active_attacks,
        )

        # Save as <vehicle_id>_<timestamp>.yaml
        out_name = f"{veh_id}_{yaml_path.stem}.yaml"
        out_path = output_dir / out_name
        save_yaml(cpm, out_path)


def run_simulation(
    root_dir: Path,
    output_dir: Path,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
    seed: Optional[int] = None,
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
        "seed": seed,
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

    config_path = output_dir / "simulation_config.yaml"
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
            dt=dt,
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