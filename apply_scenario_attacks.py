#!/usr/bin/env python3
"""
Apply attacks directly to OPV2V scenario prediction YAMLs.

- Copies source scenario into a timestamped output folder.
- Excludes .png files from the copied scenario.
- Applies attacks only to the attacker vehicle folder.
- Modifies detected_objects in *_preds.yaml files.
- Keeps *_score.npy aligned to detected_objects sorted by object id.
"""

import argparse
import random
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

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


def _id_sort_key(object_id: Any) -> Tuple[int, Any]:
    s = str(object_id)
    return (0, int(s)) if s.isdigit() else (1, s)


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


def build_attack_schedule(frame_paths: List[Path], attack_type: str) -> List[AttackInterval]:
    if not frame_paths:
        return []

    min_tick = min(parse_timestamp_from_preds_filename(p) for p in frame_paths)
    max_tick = max(parse_timestamp_from_preds_filename(p) for p in frame_paths)

    return [
        AttackInterval(
            attack_type=attack_type,
            start_ts=min_tick,
            end_ts=max_tick,
            params={},
        )
    ]


def _extract_detected_objects(preds_yaml: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    detected = preds_yaml.get("detected_objects", {})
    normalized: List[Dict[str, Any]] = []
    mapping: Dict[str, Dict[str, Any]] = {}

    if isinstance(detected, dict):
        items = list(detected.items())
    elif isinstance(detected, list):
        items = [(str(i), obj) for i, obj in enumerate(detected)]
    else:
        return [], {}, []

    # Deterministic order by object id.
    sortable = []
    for key, obj in items:
        if not isinstance(obj, dict):
            continue
        vid = obj.get("id", obj.get("object_id", key))
        sortable.append((vid, key, obj))
    sortable.sort(key=lambda x: _id_sort_key(x[0]))

    sorted_ids: List[str] = []
    for vid, key, obj in sortable:
        vid_str = str(vid)
        sorted_ids.append(vid_str)

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

    return normalized, mapping, sorted_ids


def _rebuild_detected_objects(
    attacked_objects: List[Dict[str, Any]],
    mapping: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    rebuilt: Dict[str, Dict[str, Any]] = {}

    for attacked in attacked_objects:
        vid = attacked.get("id")
        vid_str = str(vid)

        if vid_str in mapping:
            obj = mapping[vid_str]["obj"]
            new_loc = attacked.get("location", {})
            new_ori = attacked.get("orientation", {})

            obj["location"] = [
                float(new_loc.get("x", 0.0)),
                float(new_loc.get("y", 0.0)),
                float(new_loc.get("z", 0.0)),
            ]
            obj["angle"] = [
                float(new_ori.get("pitch", 0.0)),
                float(new_ori.get("yaw", 0.0)),
                float(new_ori.get("roll", 0.0)),
            ]
            rebuilt[vid_str] = obj
            continue

        # Added fake object
        new_loc = attacked.get("location", {})
        new_ori = attacked.get("orientation", {})
        extent = attacked.get("extent", {"x": 2.0, "y": 1.0, "z": 1.0})
        center = attacked.get("center", new_loc)

        if isinstance(center, dict):
            cx, cy, cz = float(center.get("x", 0.0)), float(center.get("y", 0.0)), float(center.get("z", 0.0))
        else:
            center_l = (center + [0.0, 0.0, 0.0])[:3] if isinstance(center, list) else [0.0, 0.0, 0.0]
            cx, cy, cz = float(center_l[0]), float(center_l[1]), float(center_l[2])

        if isinstance(extent, dict):
            ex, ey, ez = float(extent.get("x", 2.0)), float(extent.get("y", 1.0)), float(extent.get("z", 1.0))
        else:
            ext_l = (extent + [2.0, 1.0, 1.0])[:3] if isinstance(extent, list) else [2.0, 1.0, 1.0]
            ex, ey, ez = float(ext_l[0]), float(ext_l[1]), float(ext_l[2])

        rebuilt[vid_str] = {
            "angle": [
                float(new_ori.get("pitch", 0.0)),
                float(new_ori.get("yaw", 0.0)),
                float(new_ori.get("roll", 0.0)),
            ],
            "center": [cx, cy, cz],
            "extent": [ex, ey, ez],
            "location": [
                float(new_loc.get("x", 0.0)),
                float(new_loc.get("y", 0.0)),
                float(new_loc.get("z", 0.0)),
            ],
            "speed": float(attacked.get("speed", 0.0)),
        }

    # Deterministic id order in output mapping.
    ordered_items = sorted(rebuilt.items(), key=lambda kv: _id_sort_key(kv[0]))
    return {k: v for k, v in ordered_items}


def _in_eval_range_relative_to_lidar(obj: Dict[str, Any], lidar_pose: List[float]) -> bool:
    loc = obj.get("location", [0.0, 0.0, 0.0])
    if not isinstance(loc, list) or len(loc) < 2:
        return False
    dx = float(loc[0]) - float(lidar_pose[0])
    dy = float(loc[1]) - float(lidar_pose[1])
    # Approximate eval range in attacker-local frame. For close-radius fake objects,
    # this is equivalent in practice.
    return (-140.0 <= dx <= 140.0) and (-40.0 <= dy <= 40.0)


def _score_path_from_preds(preds_path: Path) -> Path:
    return preds_path.with_name(preds_path.name.replace("_preds.yaml", "_score.npy"))


def copy_scenario_without_png(source_scenario: Path, output_scenario: Path) -> None:
    files_to_copy = [
        p for p in source_scenario.rglob("*")
        if p.is_file() and p.suffix.lower() != ".png"
    ]
    output_scenario.mkdir(parents=True, exist_ok=False)
    for src in tqdm(files_to_copy, desc="Copying scenario (excluding .png)", unit="file"):
        rel = src.relative_to(source_scenario)
        dst = output_scenario / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def apply_attacks_to_attacker_preds(
    attacker_dir: Path,
    attacker_id: str,
    attack_schedule: List[AttackInterval],
    frame_attack_rate: float,
    rng: random.Random,
    add_fake_score_min: float,
    add_fake_score_max: float,
    add_chaos_mode: bool,
    debug_attack: bool,
    debug_attack_limit: int,
) -> None:
    scheduler = AttackScheduler({attacker_id: attack_schedule})
    preds_paths = list_preds_paths(attacker_dir)

    debug_count = 0

    for preds_path in tqdm(preds_paths, desc="Applying attacks", unit="frame"):
        if rng.random() > frame_attack_rate:
            continue

        preds_yaml = load_yaml(preds_path)
        lidar_pose = preds_yaml.get("lidar_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        timestamp = parse_timestamp_from_preds_filename(preds_path)
        active_attacks = scheduler.get_attacks_for(attacker_id, timestamp)
        if not active_attacks:
            continue

        score_path = _score_path_from_preds(preds_path)
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score file for {preds_path}: {score_path}")

        score_before = np.load(score_path).reshape(-1)

        detected_objects, mapping, sorted_ids_before = _extract_detected_objects(preds_yaml)
        if not detected_objects:
            continue

        if len(sorted_ids_before) != len(score_before):
            raise ValueError(
                f"Pre-attack score/object mismatch in {preds_path}: "
                f"detected_objects={len(sorted_ids_before)} score_len={len(score_before)}"
            )

        score_by_id = {
            sorted_ids_before[i]: float(score_before[i]) for i in range(len(sorted_ids_before))
        }

        cpm_like = {
            "detected_vehicles": list(detected_objects),
            "global_position": {
                "x": float(preds_yaml.get("lidar_pose", [0.0, 0.0, 0.0])[0]),
                "y": float(preds_yaml.get("lidar_pose", [0.0, 0.0, 0.0])[1]),
                "z": float(preds_yaml.get("lidar_pose", [0.0, 0.0, 0.0])[2]),
            },
        }

        replacement_used = False
        if add_chaos_mode and any(iv.attack_type == "AddObjectAttack" for iv in active_attacks):
            cpm_like["detected_vehicles"] = []
            replacement_used = True

        ctx = AttackContext(vehicle_id=attacker_id, timestamp=timestamp)
        for interval in active_attacks:
            if interval.attack_obj is None:
                interval.attack_obj = create_attack(interval.attack_type, interval.params)
            interval.attack_obj.apply(cpm_like, ctx)

        attacked_list = cpm_like.get("detected_vehicles", [])
        rebuilt = _rebuild_detected_objects(attacked_list, mapping)
        preds_yaml["detected_objects"] = rebuilt

        sorted_ids_after = sorted(rebuilt.keys(), key=_id_sort_key)
        scores_after = np.zeros((len(sorted_ids_after),), dtype=np.float32)

        in_range_fake_count = 0
        for i, vid in enumerate(sorted_ids_after):
            if vid in score_by_id:
                scores_after[i] = float(score_by_id[vid])
            else:
                scores_after[i] = float(rng.uniform(add_fake_score_min, add_fake_score_max))
                if _in_eval_range_relative_to_lidar(rebuilt[vid], lidar_pose):
                    in_range_fake_count += 1

        np.save(score_path, scores_after)
        save_yaml(preds_yaml, preds_path)

        if debug_attack and debug_count < debug_attack_limit:
            print(
                f"[attack-debug] file={preds_path.name} "
                f"num_detected_objects_before={len(sorted_ids_before)} "
                f"num_detected_objects_after={len(sorted_ids_after)} "
                f"num_scores_before={len(score_before)} "
                f"num_scores_after={len(scores_after)} "
                f"in_range_fake_count={in_range_fake_count} "
                f"replacement_mode_used={replacement_used}"
            )
            debug_count += 1


def build_metadata(
    schedule: List[AttackInterval],
    attacker_id: str,
    attack_type: str,
    frame_attack_rate: float,
    seed: int | None,
    source_scenario: Path,
    add_chaos_mode: bool,
) -> Dict[str, Any]:
    return {
        "attacker_id": attacker_id,
        "source_scenario": str(source_scenario),
        "attack_type": attack_type,
        "frame_attack_rate": frame_attack_rate,
        "seed": seed,
        "target_file_pattern": "*_preds.yaml",
        "target_field": "detected_objects",
        "score_sync": "strict_sorted_id",
        "add_chaos_mode": add_chaos_mode,
        "schedule": [{**asdict(interval), "attack_obj": None} for interval in schedule],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply attacks directly to OPV2V *_preds.yaml files.")
    parser.add_argument("--source-root", type=str, default="dataset/opv2v_scenario_inference")
    parser.add_argument("--scenario-name", type=str, default="2021_08_16_22_26_54")
    parser.add_argument("--output-root", type=str, default="dataset/opv2v_scenario_attacked")
    parser.add_argument("--attacker-id", type=str, default="659")
    parser.add_argument(
        "--attack-type",
        type=str,
        default="DriftAttack",
        choices=["DriftAttack", "AddObjectAttack", "RemoveObjectAttack", "WhiteNoiseAttack"],
    )
    parser.add_argument("--frame-attack-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--drift-mean-step", type=float, default=2.0)
    parser.add_argument("--drift-std-step", type=float, default=0.8)
    parser.add_argument("--drift-yaw-mean", type=float, default=10.0)
    parser.add_argument("--drift-yaw-std", type=float, default=5.0)

    parser.add_argument("--add-num-objects", type=int, default=25)
    parser.add_argument("--add-radius-min", type=float, default=2.0)
    parser.add_argument("--add-radius-max", type=float, default=40.0)
    parser.add_argument("--add-z-offset", type=float, default=0.0)
    parser.add_argument("--add-fake-score-min", type=float, default=0.8)
    parser.add_argument("--add-fake-score-max", type=float, default=0.99)
    parser.add_argument("--add-chaos-mode", action="store_true",
                        help="Replace existing detections with fake objects. Always enabled for AddObjectAttack.")

    parser.add_argument("--remove-max-fraction", type=float, default=0.95)
    parser.add_argument("--remove-max-count", type=int, default=10000)

    parser.add_argument("--wn-mean-drift", type=float, default=3.0)
    parser.add_argument("--wn-std-drift", type=float, default=1.5)
    parser.add_argument("--wn-z-mean", type=float, default=0.0)
    parser.add_argument("--wn-z-std", type=float, default=0.5)
    parser.add_argument("--wn-yaw-mean", type=float, default=15.0)
    parser.add_argument("--wn-yaw-std", type=float, default=8.0)
    parser.add_argument("--wn-pitch-mean", type=float, default=5.0)
    parser.add_argument("--wn-pitch-std", type=float, default=3.0)

    parser.add_argument("--debug-attack", action="store_true",
                        help="Print per-file attack diagnostics.")
    parser.add_argument("--debug-attack-limit", type=int, default=30)

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

    copy_scenario_without_png(scenario_dir, output_scenario)

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
                "num_targets": 10000,
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

    add_chaos_mode = args.add_chaos_mode or args.attack_type == "AddObjectAttack"

    rng = random.Random(args.seed)
    apply_attacks_to_attacker_preds(
        attacker_dir=attacker_dir,
        attacker_id=args.attacker_id,
        attack_schedule=schedule,
        frame_attack_rate=args.frame_attack_rate,
        rng=rng,
        add_fake_score_min=args.add_fake_score_min,
        add_fake_score_max=args.add_fake_score_max,
        add_chaos_mode=add_chaos_mode,
        debug_attack=args.debug_attack,
        debug_attack_limit=args.debug_attack_limit,
    )

    metadata = build_metadata(
        schedule=schedule,
        attacker_id=args.attacker_id,
        attack_type=args.attack_type,
        frame_attack_rate=args.frame_attack_rate,
        seed=args.seed,
        source_scenario=scenario_dir,
        add_chaos_mode=add_chaos_mode,
    )
    save_yaml(metadata, output_root / "attack_metadata.yaml")

    print(f"Attacked scenario written to: {output_scenario}")
    print(f"Metadata written to: {output_root / 'attack_metadata.yaml'}")


if __name__ == "__main__":
    main()
