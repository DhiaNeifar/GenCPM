from typing import Dict, Optional, Tuple
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from run_simulation import run_simulation


def process_single_simulation(
    sim_dir: Path,
    parent_output: Path,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
    seed: Optional[int],
) -> Tuple[str, bool]:
    """
    Process a single simulation directory.
    Returns (sim_name, success_flag).
    """
    try:
        sim_name = sim_dir.name
        out_dir = parent_output / sim_name
        
        # If seed is provided, make it unique per simulation
        sim_seed = None
        if seed is not None:
            sim_seed = seed + hash(sim_name) % 10000
        
        run_simulation(
            root_dir=sim_dir,
            output_dir=out_dir,
            dt=dt,
            lambda_attacks=lambda_attacks,
            mean_duration=mean_duration,
            attack_type_probs=attack_type_probs,
            seed=sim_seed,
        )
        return (sim_name, True)
    except Exception as e:
        return (sim_name, False)


def run_all_simulations_parallel(
    parent_root: Path,
    parent_output: Path,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
    seed: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> None:
    """
    Run simulations in parallel using multiprocessing.
    
    Args:
        num_workers: Number of parallel processes (default: cpu_count - 1)
    """
    # Filter simulation directories
    sim_dirs = [
        d for d in sorted(parent_root.iterdir())
        if d.is_dir() and not (parent_output / d.name).exists()
    ]
    
    if not sim_dirs:
        print("No simulations to process!")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Processing {len(sim_dirs)} simulations using {num_workers} workers...")
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_single_simulation,
        parent_output=parent_output,
        dt=dt,
        lambda_attacks=lambda_attacks,
        mean_duration=mean_duration,
        attack_type_probs=attack_type_probs,
        seed=seed,
    )
    
    # Process in parallel with progress bar
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, sim_dirs),
            total=len(sim_dirs),
            desc="Simulations",
            unit="sim"
        ))
    
    # Report results
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"\n✓ Completed: {successful}/{len(results)} simulations")
    if failed > 0:
        print(f"✗ Failed: {failed} simulations")
        failed_sims = [name for name, success in results if not success]
        print(f"  Failed simulations: {', '.join(failed_sims)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel random attack simulation over CPMs using a Poisson process."
    )
    parser.add_argument("root", type=str, help="Path to parent directory containing simulation folders.")
    parser.add_argument("output", type=str, help="Parent output folder for CPMs.")

    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step between frames in seconds (default: 0.05).")
    parser.add_argument("--lambda-attacks", type=float, default=1/5,
                        help="Attack rate (attacks per second) per vehicle (default: 1/5 = 1 per 5 seconds).")
    parser.add_argument("--mean-duration", type=float, default=5.0,
                        help="Mean attack duration in seconds (default: 5s).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (default: None). Each simulation gets a unique derived seed.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count - 1).")

    return parser.parse_args()


def main():
    args = parse_args()
    parent_root = Path(args.root)
    parent_output = Path(args.output)

    attack_type_probs = {
        "DriftAttack": 0.25,
        "AddObjectAttack": 0.25,
        "RemoveObjectAttack": 0.25,
        "WhiteNoiseAttack": 0.25,
    }

    run_all_simulations_parallel(
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
    # Important for macOS: this guard is required for multiprocessing
    # python run_simulations_parallel.py "C:\Users\dhian\PycharmProjects\GenCPM\dataset\carla_dataset"  "C:\Users\dhian\PycharmProjects\GenCPM\dataset\CPM" --workers 10
    main()
