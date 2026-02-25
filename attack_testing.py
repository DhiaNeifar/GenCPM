#!/usr/bin/env python3
"""
test_attack.py

Given a single CARLA frame YAML:
  - build a benign CPM
  - apply ONE attack with chosen parameters
  - save benign + malicious CPMs

Usage:
    python test_attack.py frame.yaml benign_cpm.yaml malicious_cpm.yaml
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

from cpm_builder import build_cpm, load_yaml, save_yaml
from attack import create_attack, AttackContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("frame", type=str, help="Input CARLA frame YAML (<timestamp>.yaml)")
    p.add_argument("benign_out", type=str, help="Output benign CPM YAML")
    p.add_argument("malicious_out", type=str, help="Output malicious CPM YAML")
    return p.parse_args()


def choose_attack() -> tuple[str, Dict[str, Any]]:
    """
    Manually choose ONE attack + params here.
    Comment/uncomment to test different attacks.
    """

    # -------- WhiteNoiseAttack (per-frame noise, 3D + attitude) --------
    # attack_type = "WhiteNoiseAttack"
    # params = {
    #     "mean_drift": 0.5,
    #     "std_drift": 0.2,
    #     "z_drift_mean": 0.0,
    #     "z_drift_std": 0.05,
    #     "yaw_drift_mean": 0.0,
    #     "yaw_drift_std": 5.0,
    #     "pitch_drift_mean": 0.0,
    #     "pitch_drift_std": 2.0,
    #     # "target_ids": [3, 5],  # optional
    #     # "num_targets": 3,
    # }

    # -------- DriftAttack (cumulative random walk) --------
    # attack_type = "DriftAttack"
    # params = {
    #     "mean_step": 0.05,        # meters per frame
    #     "std_step": 0.02,         # meters per frame
    #     "yaw_step_mean": 0.0,     # deg per frame
    #     "yaw_step_std": 1.0,      # deg per frame
    #     # "target_ids": [3, 5],   # optional
    #     # "num_targets": 3,
    # }

    # -------- AddObjectAttack --------
    # attack_type = "AddObjectAttack"
    # params = {
    #     "num_objects": 2,
    #     "radius_range": (5.0, 15.0),  # meters around ego
    #     "z_offset": 0.0,
    # }

    # -------- RemoveObjectAttack --------
    # attack_type = "RemoveObjectAttack"
    # params = {
    #     "max_remove_fraction": 0.3,  # up to 30% of vehicles
    #     "max_remove_count": 5,       # but no more than 5
    #     # "target_ids": [1, 2],      # if set, prefer removing these
    # }

    return attack_type, params



def main() -> None:
    args = parse_args()

    frame_path = Path(args.frame)

    # 1) load raw CARLA frame
    frame = load_yaml(frame_path)

    # 2) build benign CPM
    benign_cpm = build_cpm(
        frame=frame,
        cpm_type="benign",
        attack=None,
    )

    # save benign for inspection
    save_yaml(benign_cpm, Path(args.benign_out))

    # 3) clone CPM and apply one attack
    malicious_cpm = copy.deepcopy(benign_cpm)

    attack_type, params = choose_attack()
    attack = create_attack(attack_type, params=params)

    # Context is mostly for bookkeeping; timestamp is arbitrary here
    ctx = AttackContext(
        vehicle_id=str(malicious_cpm.get("vehicle_id", "test")),
        timestamp=float(malicious_cpm.get("timestamp", 0.0)),
    )

    meta = attack.apply(malicious_cpm, ctx)
    malicious_cpm["cpm_type"] = "malicious"
    malicious_cpm["attacks"] = [meta]

    # 4) save malicious CPM
    save_yaml(malicious_cpm, Path(args.malicious_out))

    print(f"Benign CPM saved to   {args.benign_out}")
    print(f"Malicious CPM saved to {args.malicious_out}")
    print("Attack metadata:", meta)


if __name__ == "__main__":
    main()
