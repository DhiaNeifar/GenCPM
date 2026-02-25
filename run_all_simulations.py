from typing import Dict, Optional
import argparse
from pathlib import Path
from tqdm import tqdm

from run_simulation import run_simulation  # if in separate file

def run_all_simulations(
    parent_root: Path,
    parent_output: Path,
    dt: float,
    lambda_attacks: float,
    mean_duration: float,
    attack_type_probs: Dict[str, float],
    seed: Optional[int] = None,
) -> None:
    sim_dirs = [
    d for d in sorted(parent_root.iterdir())
    if d.is_dir() and not (parent_output / d.name).exists()
]
    # sim_dirs = [d for d in sorted(parent_root.iterdir()) if d.is_dir()]
    for sim_dir in tqdm(sim_dirs, desc="Simulations", unit="sim"):
        sim_name = sim_dir.name
        out_dir = parent_output / sim_name
        run_simulation(
            root_dir=sim_dir,
            output_dir=out_dir,
            dt=dt,
            lambda_attacks=lambda_attacks,
            mean_duration=mean_duration,
            attack_type_probs=attack_type_probs,
            seed=seed,
        )
        
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random attack simulation over CPMs using a Poisson process."
    )
    parser.add_argument("root", type=str, help="Path <p> to simulation root.")
    parser.add_argument("output", type=str, help="Output folder for CPMs.")

    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step between frames in seconds (default: 0.05).")
    parser.add_argument("--lambda-attacks", type=float, default=1/5,
                        help="Attack rate (attacks per second) per vehicle (default: 1/60 ≈ 1 per minute).")
    parser.add_argument("--mean-duration", type=float, default=5.0,
                        help="Mean attack duration in seconds (default: 10s).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None).")

    return parser.parse_args()

def main():
    args = parse_args_multi()
    parent_root = Path(args.parent_root)
    parent_output = Path(args.parent_output)

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
    )

if __name__ == "__main__":
    # python run_all_simulations.py "D:\opv2v\train" "C:\Users\dhian\Desktop\GenCPM\CPM collected"
    main()
