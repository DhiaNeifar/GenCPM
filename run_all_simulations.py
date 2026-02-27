from typing import Dict, Iterable, Optional, Tuple
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

from run_simulation import run_simulation


def is_sim_root(candidate: Path) -> bool:
    """
    A simulation root has immediate subdirs (vehicle IDs) that contain YAML frames.
    """
    if not candidate.is_dir():
        return False
    for child in candidate.iterdir():
        if not child.is_dir():
            continue
        if any(child.glob("*.yaml")):
            return True
    return False


def discover_sim_roots(parent_root: Path) -> Iterable[Path]:
    """
    Recursively find simulation roots under parent_root.
    """
    stack = [parent_root]
    while stack:
        current = stack.pop()
        if is_sim_root(current):
            yield current
            continue
        for child in current.iterdir():
            if child.is_dir():
                stack.append(child)


def process_single_simulation(task: Tuple[Path, Path, Path, float, float, float, Dict[str, float], Optional[int]]) -> Tuple[str, bool]:
    """
    Top-level worker function so it is pickleable on Windows multiprocessing.
    """
    sim_dir, parent_root, parent_output, dt, lambda_attacks, mean_duration, attack_type_probs, seed = task
    rel_path = sim_dir.relative_to(parent_root)
    out_dir = parent_output / rel_path

    try:
        if out_dir.exists():
            return (str(rel_path), True)

        sim_seed = None
        if seed is not None:
            sim_seed = seed + hash(str(rel_path)) % 10000

        run_simulation(
            root_dir=sim_dir,
            output_dir=out_dir,
            dt=dt,
            lambda_attacks=lambda_attacks,
            mean_duration=mean_duration,
            attack_type_probs=attack_type_probs,
            seed=sim_seed,
        )
        return (str(rel_path), True)
    except Exception:
        return (str(rel_path), False)


def run_all_simulations(
    parent_root: Path,
    parent_output: Path,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
    seed: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> None:
    sim_dirs = sorted(discover_sim_roots(parent_root))
    if not sim_dirs:
        print("No simulation roots found. Check your dataset path.")
        return

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"Processing {len(sim_dirs)} simulations using {num_workers} workers...")
    tasks = [
        (sim_dir, parent_root, parent_output, dt, lambda_attacks, mean_duration, attack_type_probs, seed)
        for sim_dir in sim_dirs
    ]

    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_simulation, tasks),
                total=len(sim_dirs),
                desc="Simulations",
                unit="sim",
            )
        )

    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    print(f"\nCompleted: {successful}/{len(results)} simulations")
    if failed > 0:
        failed_sims = [name for name, success in results if not success]
        print(f"Failed: {failed} simulations")
        print(f"  Failed simulations: {', '.join(failed_sims)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random attack simulation over CPMs using a Poisson process."
    )
    parser.add_argument("root", type=str, help="Path to dataset root (e.g., carla_dataset).")
    parser.add_argument("output", type=str, help="Output folder for CPMs.")

    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step between frames in seconds (default: 0.05).")
    parser.add_argument("--lambda-attacks", type=float, default=1/5,
                        help="Attack rate (attacks per second) per vehicle.")
    parser.add_argument("--mean-duration", type=float, default=5.0,
                        help="Mean attack duration in seconds.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count - 1).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_root = Path(args.root)
    parent_output = Path(args.output)

    attack_type_probs = {
        "DriftAttack": 0.25,
        "AddObjectAttack": 0.25,
        "RemoveObjectAttack": 0.25,
        "WhiteNoiseAttack": 0.25,
    }

    run_all_simulations(
        parent_root=parent_root,
        parent_output=parent_output,
        dt=args.dt,
        lambda_attacks=args.lambda_attacks,
        mean_duration=args.mean_duration,
        attack_type_probs=attack_type_probs,
        seed=args.seed,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
