# GenCPM

Generate CPM (Cooperative Perception Message) datasets from CARLA frames and apply attack simulations.

## Project Layout

- `run_simulation.py`: Run one simulation root (one timestamp folder with vehicle subfolders).
- `run_simulations_parallel.py`: Run multiple simulation roots in parallel.
- `run_all_simulations.py`: Discover all simulation roots under a dataset and run them in parallel.
- `cpm_builder.py`: Build one CPM from one CARLA frame.
- `attacks/`: Attack framework split by attack type.
- `utils/`: Shared helpers (`yaml_utils`, `id_utils`).
- `tests/unit/`: Unit tests for attacks and CPM builder.
- `legacy/`: Older helper/testing scripts kept for reference.

## Attack Modules

- `attacks/drift_attack.py`
- `attacks/white_noise_attack.py`
- `attacks/add_object_attack.py`
- `attacks/remove_object_attack.py`
- `attacks/registry.py` (`create_attack`, `ATTACK_REGISTRY`)

Backward compatibility is kept via `attack.py`.

## Setup

```powershell
pip install pyyaml tqdm
```

## Typical Commands

Build a single CPM:

```powershell
python cpm_builder.py "path\to\frame.yaml" "path\to\out.yaml"
```

Run one simulation root:

```powershell
python run_simulation.py "path\to\Town01\2025_12_22_00_29_15" "path\to\output"
```

Run all simulations under a town in parallel:

```powershell
python run_simulations_parallel.py "path\to\Town01" "path\to\CPM\Town01" --workers 10
```

Run full dataset in parallel:

```powershell
python run_all_simulations.py "C:\Users\dhian\PycharmProjects\GenCPM\dataset\carla_dataset" "C:\Users\dhian\PycharmProjects\GenCPM\dataset\CPM" --workers 20
```

## Tests

Run unit tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Notes

- `run_all_simulations.py` and `run_simulations_parallel.py` skip simulations whose output folder already exists.
- On Windows multiprocessing, worker functions must be top-level (already handled in the current code).
