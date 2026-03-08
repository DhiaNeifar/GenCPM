from typing import Dict, Iterable, Optional, Tuple
import argparse
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from run_simulation import run_simulation


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
    ]
) -> Tuple[str, bool]:
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
    ) = task

    rel_path = sim_dir.relative_to(parent_root)
    out_dir = parent_output / rel_path
    run_label = f"{rel_path} [repeat={repeat_idx:03d}]"
    config_marker = out_dir / f"simulation_config_r{repeat_idx:03d}.yaml"

    try:
        if config_marker.exists():
            return (run_label, True)

        sim_seed = None
        if seed is not None:
            sim_seed = seed + (hash(str(rel_path)) % 10000) + repeat_idx

        run_simulation(
            root_dir=sim_dir,
            output_dir=out_dir,
            dt=dt,
            lambda_attacks=lambda_attacks,
            mean_duration=mean_duration,
            attack_type_probs=attack_type_probs,
            seed=sim_seed,
            repeat_idx=repeat_idx,
        )
        return (run_label, True)
    except Exception:
        return (run_label, False)


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

    tasks = []
    for scenario_dir in scenario_dirs:
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
                )
            )

    print(f"Processing {len(tasks)} simulation runs from {len(scenario_dirs)} scenarios using {num_workers} workers...")

    if num_workers == 1:
        results = []
        for task in tqdm(tasks, total=len(tasks), desc="Simulations", unit="run"):
            results.append(process_single_simulation(task))
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_simulation, tasks),
                    total=len(tasks),
                    desc="Simulations",
                    unit="run",
                )
            )

    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    print(f"\nCompleted: {successful}/{len(results)} runs")
    if failed > 0:
        failed_runs = [name for name, success in results if not success]
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
