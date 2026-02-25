from pathlib import Path
from run_simulation import build_random_schedule_for_all_vehicles

root = Path(r"C:\Users\dhian\PycharmProjects\AdverCPM\experiments\raw simulations\2021_08_16_22_26_54")
sim_duration = 600.0  # e.g. 10 minutes
lambda_attacks = 1/60
mean_duration = 10.0
attack_type_probs = {"DriftAttack": 0.5, "AddObjectAttack": 0.5}

intervals_per_vehicle = build_random_schedule_for_all_vehicles(
    root_dir=root,
    sim_duration=sim_duration,
    lambda_attacks=lambda_attacks,
    mean_duration=mean_duration,
    attack_type_probs=attack_type_probs,
)

for vid, intervals in intervals_per_vehicle.items():
    print(f"Vehicle {vid}: {len(intervals)} intervals")
    for iv in intervals:
        print(f"  {iv.attack_type}: [{iv.start_ts:.1f}, {iv.end_ts:.1f}]")
