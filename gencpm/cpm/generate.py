from typing import Any, Dict, Iterable, Optional, Tuple
import argparse
import multiprocessing as mp
import subprocess
import time
import traceback
import zlib
from pathlib import Path

from tqdm import tqdm

from gencpm.cpm.simulation import run_simulation


def is_scenario_root(candidate: Path) -> bool:
    """
    A scenario root has vehicle subdirectories that contain *_preds.yaml files.
    """
    if not candidate.is_dir():
        return False
    for vehicle_dir in candidate.iterdir():
        if not vehicle_dir.is_dir():
            continue
        if any(vehicle_dir.glob("*_preds.yaml")):
            return True
    return False


def discover_scenario_roots(parent_root: Path) -> Iterable[Path]:
    """
    Recursively find scenario roots under parent_root.
    """
    stack = [parent_root]
    while stack:
        current = stack.pop()
        if is_scenario_root(current):
            yield current
            continue
        for child in current.iterdir():
            if child.is_dir():
                stack.append(child)


GENERATOR_VERSION = "v2"
GENERATOR_CHANGES = [
    "CF appends fabricated objects to the real list (v1 replaced it)",
    "CF objects use local center [0,0,h/2] and ground-level z (v1 used world coords in center)",
    "CSD accumulates per persistent track_id (v1 keyed by per-frame detection id, never accumulated)",
    "detected_vehicles carry a persistent track_id (constant-velocity tracker, utils/tracking.py)",
    "per-run seed = base_seed + crc32(relative scenario path) + repeat_idx (v1 used Python hash(), not reproducible)",
    "advanced attacks available: GhostTrackAttack, StealthDriftAttack, ReplayAttack, EvasiveGhostAttack",
]

# Shared frame counter for the live progress bar (set in each worker by the pool initializer).
_PROGRESS = None


def _init_worker(counter) -> None:
    global _PROGRESS
    _PROGRESS = counter


def stable_run_seed(base_seed: int, rel_path: Path, repeat_idx: int) -> int:
    """Deterministic across processes and machines (unlike built-in hash())."""
    return base_seed + (zlib.crc32(rel_path.as_posix().encode()) % 10000) + repeat_idx


def git_state(repo: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def process_single_simulation(
    task: Tuple[
        Path,
        Path,
        Path,
        int,
        float,
        float,
        float,
        Dict[str, float],
        Optional[int],
        Dict[str, Any],
    ]
) -> Tuple[str, bool, str]:
    """
    Top-level worker function so it is pickleable on Windows multiprocessing.
    """
    (
        sim_dir,
        parent_root,
        parent_output,
        repeat_idx,
        dt,
        lambda_attacks,
        mean_duration,
        attack_type_probs,
        seed,
        generator_info,
    ) = task

    rel_path = sim_dir.relative_to(parent_root)
    out_dir = parent_output / rel_path
    run_label = f"{rel_path} [repeat={repeat_idx:03d}]"
    config_marker = out_dir / f"simulation_config_r{repeat_idx:03d}.yaml"

    try:
        if config_marker.exists():
            return (run_label, True, "skipped (exists)")

        sim_seed = None
        if seed is not None:
            sim_seed = stable_run_seed(seed, rel_path, repeat_idx)

        run_simulation(
            root_dir=sim_dir,
            output_dir=out_dir,
            dt=dt,
            lambda_attacks=lambda_attacks,
            mean_duration=mean_duration,
            attack_type_probs=attack_type_probs,
            seed=sim_seed,
            repeat_idx=repeat_idx,
            progress=_PROGRESS,
            generator_info=generator_info,
        )
        return (run_label, True, "")
    except Exception:
        return (run_label, False, traceback.format_exc())


def run_all_simulations_parallel(
    parent_root: Path,
    parent_output: Path,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
    repeats: int = 1,
    seed: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> None:
    scenario_dirs = sorted(discover_scenario_roots(parent_root))
    if not scenario_dirs:
        print("No scenario roots found. Check your dataset path.")
        return

    repeats = max(1, int(repeats))
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    generator_info = {"version": GENERATOR_VERSION, "changes": GENERATOR_CHANGES, "base_seed": seed,
                      "git": git_state(Path(__file__).resolve().parent)}

    tasks = []
    total_frames = 0
    for scenario_dir in scenario_dirs:
        n_frames = sum(1 for _ in scenario_dir.glob("*/*_preds.yaml"))
        for repeat_idx in range(repeats):
            tasks.append(
                (
                    scenario_dir,
                    parent_root,
                    parent_output,
                    repeat_idx,
                    dt,
                    lambda_attacks,
                    mean_duration,
                    attack_type_probs,
                    seed,
                    generator_info,
                )
            )
            total_frames += n_frames

    print(f"Generator {GENERATOR_VERSION} | git {generator_info['git']}")
    print(f"Processing {len(tasks)} simulation runs ({len(scenario_dirs)} scenarios x {repeats} repeats, "
          f"{total_frames:,} CPMs) with {num_workers} workers, base seed {seed}")
    print(f"Input : {parent_root}\nOutput: {parent_output}\n")

    counter = mp.Value("q", 0)
    results = []
    with mp.get_context("fork").Pool(processes=num_workers, initializer=_init_worker, initargs=(counter,)) as pool:
        pending = [pool.apply_async(process_single_simulation, (t,)) for t in tasks]
        with tqdm(total=total_frames, desc="CPMs", unit="cpm", position=0, dynamic_ncols=True) as bar_cpm, \
                tqdm(total=len(tasks), desc="Runs", unit="run", position=1, dynamic_ncols=True) as bar_run:
            done = set()
            while len(done) < len(pending):
                for i, r in enumerate(pending):
                    if i not in done and r.ready():
                        done.add(i)
                        res = r.get()
                        results.append(res)
                        bar_run.update(1)
                        bar_run.set_postfix_str(res[0])
                        if not res[1]:
                            tqdm.write(f"FAILED {res[0]}\n{res[2]}")
                bar_cpm.n = counter.value
                bar_cpm.refresh()
                time.sleep(0.5)
            bar_cpm.n = counter.value
            bar_cpm.refresh()

    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful

    print(f"\nCompleted: {successful}/{len(results)} runs, {counter.value:,} CPMs written")
    if failed > 0:
        failed_runs = [name for name, success, _ in results if not success]
        print(f"Failed: {failed} runs")
        print(f"  Failed runs: {', '.join(failed_runs)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel CPM generation from *_preds.yaml with one attack max per frame."
    )
    parser.add_argument("root", type=str, help="Path to parent directory containing towns/scenarios.")
    parser.add_argument("output", type=str, help="Parent output folder for CPMs.")

    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step between frames in seconds (default: 0.05).")
    parser.add_argument("--lambda-attacks", type=float, default=1/5,
                        help="Attack start rate (attacks per second) per vehicle (default: 0.2).")
    parser.add_argument("--mean-duration", type=float, default=5.0,
                        help="Mean attack duration in seconds (default: 5.0).")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Number of runs per scenario (default: 1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: None).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count - 1).")

    parser.add_argument("--prob-drift", type=float, default=0.25,
                        help="Attack type probability weight for DriftAttack.")
    parser.add_argument("--prob-add", type=float, default=0.25,
                        help="Attack type probability weight for AddObjectAttack.")
    parser.add_argument("--prob-remove", type=float, default=0.25,
                        help="Attack type probability weight for RemoveObjectAttack.")
    parser.add_argument("--prob-whitenoise", type=float, default=0.25,
                        help="Attack type probability weight for WhiteNoiseAttack.")

    parser.add_argument("--attack-probs", type=str, default=None,
                        help='JSON dict of attack_type -> weight; overrides the four --prob-* options, e.g. '
                             '\'{"GhostTrackAttack": 1, "StealthDriftAttack": 1, "ReplayAttack": 1, "EvasiveGhostAttack": 1}\'')

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_root = Path(args.root)
    parent_output = Path(args.output)

    attack_type_probs = {
        "DriftAttack": max(0.0, float(args.prob_drift)),
        "AddObjectAttack": max(0.0, float(args.prob_add)),
        "RemoveObjectAttack": max(0.0, float(args.prob_remove)),
        "WhiteNoiseAttack": max(0.0, float(args.prob_whitenoise)),
    }
    if args.attack_probs:
        import json
        attack_type_probs = {k: max(0.0, float(v)) for k, v in json.loads(args.attack_probs).items()}
    if sum(attack_type_probs.values()) <= 0:
        raise ValueError("At least one attack type probability must be > 0.")

    run_all_simulations_parallel(
        parent_root=parent_root,
        parent_output=parent_output,
        dt=args.dt,
        lambda_attacks=args.lambda_attacks,
        mean_duration=args.mean_duration,
        attack_type_probs=attack_type_probs,
        repeats=args.repeats,
        seed=args.seed,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
