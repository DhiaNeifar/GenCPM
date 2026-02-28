#!/usr/bin/env python3
"""
Apply attacks directly to OPV2V scenario prediction YAMLs.

- Copies source scenario into a timestamped output folder.
- Excludes .png files from the copied scenario.
- Applies attacks only to the attacker vehicle folder.
- Modifies detected_objects in t_preds.yaml files.
"""

import argparse
import random
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from attacks import AttackContext, create_attack
from run_simulation import AttackScheduler, AttackInterval
from utils import load_yaml, save_yaml


def parse_timestamp_from_preds_filename(path: Path) -> int:
    name = path.name
    suffix = "_preds.yaml"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected preds filename format: {name}")
    ts_str = name[: -len(suffix)]
    return int(ts_str)


def list_preds_paths(vehicle_dir: Path) -> List[Path]:
    return sorted(
        [p for p in vehicle_dir.glob("*_preds.yaml") if p.is_file()],
        key=lambda p: parse_timestamp_from_preds_filename(p),
    )


def build_attack_schedule(frame_paths: List[Path], attack_type: str) -> List[AttackInterval]:
    if not frame_paths:
        return []

    min_tick = min(parse_timestamp_from_preds_filename(p) for p in frame_paths)
    max_tick = max(parse_timestamp_from_preds_filename(p) for p in frame_paths)

    # Full timeline coverage; frame_attack_rate controls how many timestamps are attacked.
    return [
        AttackInterval(
            attack_type=attack_type,
            start_ts=min_tick,
            end_ts=max_tick,
            params={},
        )
    ]


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


def _extract_detected_objects(preds_yaml: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    detected = preds_yaml.get("detected_objects", {})
    normalized: List[Dict[str, Any]] = []
    mapping: Dict[str, Dict[str, Any]] = {}

    if isinstance(detected, dict):
        iterable = detected.items()
    elif isinstance(detected, list):
        iterable = list(enumerate(detected))
    else:
        return [], {}

    for key, obj in iterable:
        if not isinstance(obj, dict):
            continue

        vid = obj.get("id")
        if vid is None:
            vid = obj.get("object_id")
        if vid is None:
            vid = key

        vid_str = str(vid)

        lx, ly, lz = _to_float3(obj.get("location"), (0.0, 0.0, 0.0))
        pitch, yaw, roll = _to_float3(obj.get("angle"), (0.0, 0.0, 0.0))
        if "orientation" in obj:
            pitch, yaw, roll = _to_float3(obj.get("orientation"), (pitch, yaw, roll))

        normalized.append(
            {
                "id": int(vid) if vid_str.isdigit() else vid,
                "location": {"x": lx, "y": ly, "z": lz},
                "orientation": {"pitch": pitch, "yaw": yaw, "roll": roll},
            }
        )

        mapping[vid_str] = {"obj": obj, "container_key": key}

    return normalized, mapping


def _write_back_detected_objects(
    preds_yaml: Dict[str, Any],
    attacked_objects: List[Dict[str, Any]],
    mapping: Dict[str, Dict[str, Any]],
) -> None:
    for attacked in attacked_objects:
        vid = attacked.get("id")
        vid_str = str(vid)

        if vid_str not in mapping:
            continue

        obj = mapping[vid_str]["obj"]
        new_loc = attacked.get("location", {})
        new_ori = attacked.get("orientation", {})

        # Preserve existing shape for location field.
        if isinstance(obj.get("location"), dict):
            obj_loc = obj.setdefault("location", {})
            obj_loc["x"] = float(new_loc.get("x", obj_loc.get("x", 0.0)))
            obj_loc["y"] = float(new_loc.get("y", obj_loc.get("y", 0.0)))
            obj_loc["z"] = float(new_loc.get("z", obj_loc.get("z", 0.0)))
        else:
            obj["location"] = [
                float(new_loc.get("x", 0.0)),
                float(new_loc.get("y", 0.0)),
                float(new_loc.get("z", 0.0)),
            ]

        # Preserve existing shape for orientation/angle.
        if isinstance(obj.get("orientation"), dict):
            obj_ori = obj.setdefault("orientation", {})
            obj_ori["pitch"] = float(new_ori.get("pitch", obj_ori.get("pitch", 0.0)))
            obj_ori["yaw"] = float(new_ori.get("yaw", obj_ori.get("yaw", 0.0)))
            obj_ori["roll"] = float(new_ori.get("roll", obj_ori.get("roll", 0.0)))
        else:
            obj["angle"] = [
                float(new_ori.get("pitch", 0.0)),
                float(new_ori.get("yaw", 0.0)),
                float(new_ori.get("roll", 0.0)),
            ]


def apply_attacks_to_attacker_preds(
    attacker_dir: Path,
    attacker_id: str,
    attack_schedule: List[AttackInterval],
    frame_attack_rate: float,
    rng: random.Random,
) -> None:
    scheduler = AttackScheduler({attacker_id: attack_schedule})
    preds_paths = list_preds_paths(attacker_dir)

    for preds_path in preds_paths:
        if rng.random() > frame_attack_rate:
            continue

        preds_yaml = load_yaml(preds_path)
        timestamp = parse_timestamp_from_preds_filename(preds_path)
        active_attacks = scheduler.get_attacks_for(attacker_id, timestamp)
        if not active_attacks:
            continue

        detected_objects, mapping = _extract_detected_objects(preds_yaml)
        if not detected_objects:
            continue

        cpm_like = {"detected_vehicles": detected_objects}
        ctx = AttackContext(vehicle_id=attacker_id, timestamp=timestamp)

        for interval in active_attacks:
            if interval.attack_obj is None:
                interval.attack_obj = create_attack(interval.attack_type, interval.params)
            interval.attack_obj.apply(cpm_like, ctx)

        _write_back_detected_objects(preds_yaml, cpm_like.get("detected_vehicles", []), mapping)
        save_yaml(preds_yaml, preds_path)


def build_metadata(
    schedule: List[AttackInterval],
    attacker_id: str,
    attack_type: str,
    frame_attack_rate: float,
    seed: int | None,
    source_scenario: Path,
) -> Dict[str, Any]:
    return {
        "attacker_id": attacker_id,
        "source_scenario": str(source_scenario),
        "attack_type": attack_type,
        "frame_attack_rate": frame_attack_rate,
        "seed": seed,
        "target_file_pattern": "*_preds.yaml",
        "target_field": "detected_objects",
        "schedule": [{**asdict(interval), "attack_obj": None} for interval in schedule],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply attacks directly to OPV2V t_preds.yaml files.")
    parser.add_argument("--source-root", type=str, default="dataset/opv2v_scenario")
    parser.add_argument("--scenario-name", type=str, default="2021_08_16_22_26_54")
    parser.add_argument("--output-root", type=str, default="dataset/opv2v_scenario_attacked")
    parser.add_argument("--attacker-id", type=str, default="659")
    parser.add_argument(
        "--attack-type",
        type=str,
        default="DriftAttack",
        choices=["DriftAttack", "AddObjectAttack", "RemoveObjectAttack", "WhiteNoiseAttack"],
    )
    parser.add_argument("--frame-attack-rate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)

    # Drift defaults (aggressive, tunable).
    parser.add_argument("--drift-mean-step", type=float, default=2.0)
    parser.add_argument("--drift-std-step", type=float, default=0.8)
    parser.add_argument("--drift-yaw-mean", type=float, default=10.0)
    parser.add_argument("--drift-yaw-std", type=float, default=5.0)

    # AddObject defaults.
    parser.add_argument("--add-num-objects", type=int, default=25)
    parser.add_argument("--add-radius-min", type=float, default=2.0)
    parser.add_argument("--add-radius-max", type=float, default=40.0)
    parser.add_argument("--add-z-offset", type=float, default=0.0)

    # RemoveObject defaults (aggressive).
    parser.add_argument("--remove-max-fraction", type=float, default=0.95)
    parser.add_argument("--remove-max-count", type=int, default=10000)

    # WhiteNoise defaults (aggressive).
    parser.add_argument("--wn-mean-drift", type=float, default=3.0)
    parser.add_argument("--wn-std-drift", type=float, default=1.5)
    parser.add_argument("--wn-z-mean", type=float, default=0.0)
    parser.add_argument("--wn-z-std", type=float, default=0.5)
    parser.add_argument("--wn-yaw-mean", type=float, default=15.0)
    parser.add_argument("--wn-yaw-std", type=float, default=8.0)
    parser.add_argument("--wn-pitch-mean", type=float, default=5.0)
    parser.add_argument("--wn-pitch-std", type=float, default=3.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root)
    scenario_dir = source_root / args.scenario_name
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_dir}")

    source_attacker_dir = scenario_dir / args.attacker_id
    if not source_attacker_dir.exists():
        raise FileNotFoundError(f"Attacker folder not found in source scenario: {source_attacker_dir}")

    source_preds_paths = list_preds_paths(source_attacker_dir)
    if not source_preds_paths:
        raise FileNotFoundError(
            f"No *_preds.yaml files found under source attacker folder: {source_attacker_dir}"
        )

    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / timestamp_label
    output_scenario = output_root / args.scenario_name
    if output_scenario.exists():
        raise FileExistsError(f"Output scenario already exists: {output_scenario}")

    shutil.copytree(scenario_dir, output_scenario, ignore=shutil.ignore_patterns("*.png"))

    attacker_dir = output_scenario / args.attacker_id
    if not attacker_dir.exists():
        raise FileNotFoundError(f"Attacker folder not found: {attacker_dir}")

    schedule = build_attack_schedule(source_preds_paths, args.attack_type)

    for interval in schedule:
        if interval.attack_type == "DriftAttack":
            interval.params = {
                "mean_step": args.drift_mean_step,
                "std_step": args.drift_std_step,
                "yaw_step_mean": args.drift_yaw_mean,
                "yaw_step_std": args.drift_yaw_std,
                "num_targets": 10000,
            }
        elif interval.attack_type == "AddObjectAttack":
            interval.params = {
                "num_objects": args.add_num_objects,
                "radius_range": (args.add_radius_min, args.add_radius_max),
                "z_offset": args.add_z_offset,
            }
        elif interval.attack_type == "RemoveObjectAttack":
            interval.params = {
                "max_remove_fraction": args.remove_max_fraction,
                "max_remove_count": args.remove_max_count,
            }
        elif interval.attack_type == "WhiteNoiseAttack":
            interval.params = {
                "mean_drift": args.wn_mean_drift,
                "std_drift": args.wn_std_drift,
                "z_drift_mean": args.wn_z_mean,
                "z_drift_std": args.wn_z_std,
                "yaw_drift_mean": args.wn_yaw_mean,
                "yaw_drift_std": args.wn_yaw_std,
                "pitch_drift_mean": args.wn_pitch_mean,
                "pitch_drift_std": args.wn_pitch_std,
                "num_targets": 10000,
            }
        else:
            interval.params = {}

    rng = random.Random(args.seed)
    apply_attacks_to_attacker_preds(
        attacker_dir=attacker_dir,
        attacker_id=args.attacker_id,
        attack_schedule=schedule,
        frame_attack_rate=args.frame_attack_rate,
        rng=rng,
    )

    metadata = build_metadata(
        schedule=schedule,
        attacker_id=args.attacker_id,
        attack_type=args.attack_type,
        frame_attack_rate=args.frame_attack_rate,
        seed=args.seed,
        source_scenario=scenario_dir,
    )
    save_yaml(metadata, output_root / "attack_metadata.yaml")

    print(f"Attacked scenario written to: {output_scenario}")
    print(f"Metadata written to: {output_root / 'attack_metadata.yaml'}")


if __name__ == "__main__":
    main()
