#!/usr/bin/env python3
"""
Compute per-town scenario statistics for LaTeX tables.

Supported roots:
1) Raw dataset root (TownXX/<scenario>/*.txt with Parameters line)
2) CPM root (TownXX/<scenario>/simulation_config_r*.yaml)
"""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


PARAMS_RE = re.compile(
    r"Parameters:\s*CAVs=(?P<cavs>\d+),\s*Vehicles=(?P<vehicles>\d+),\s*Pedestrians=(?P<pedestrians>\d+),\s*Ticks=(?P<ticks>\d+)"
)


@dataclass(frozen=True)
class ScenarioParams:
    town: str
    scenario: str
    cavs: int
    vehicles: int
    pedestrians: int
    ticks: int
    source_file: Path


def parse_parameters_from_text_file(path: Path) -> Tuple[int, int, int, int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Fast path: parameters line is usually line 3.
    if len(lines) >= 3:
        match = PARAMS_RE.search(lines[2])
        if match:
            return (
                int(match.group("cavs")),
                int(match.group("vehicles")),
                int(match.group("pedestrians")),
                int(match.group("ticks")),
            )

    # Fallback: search all lines.
    for line in lines:
        match = PARAMS_RE.search(line)
        if match:
            return (
                int(match.group("cavs")),
                int(match.group("vehicles")),
                int(match.group("pedestrians")),
                int(match.group("ticks")),
            )

    raise ValueError(f"Could not parse Parameters line in: {path}")


def town_sort_key(town_name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", town_name)
    if match:
        return (int(match.group(1)), town_name)
    return (10**9, town_name)


def mean_std(values: List[int]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), 0.0)
    return (statistics.mean(values), statistics.stdev(values))


def iter_txt_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.txt"):
        if path.is_file():
            yield path


def iter_simulation_configs(root: Path) -> Iterable[Path]:
    for path in root.rglob("simulation_config*.yaml"):
        if path.is_file():
            yield path


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data
    return {}


def load_all_scenario_params_from_txt(root: Path) -> List[ScenarioParams]:
    scenarios: List[ScenarioParams] = []

    for txt_path in sorted(iter_txt_files(root)):
        rel = txt_path.relative_to(root)
        if len(rel.parts) < 2:
            continue

        town = rel.parts[0]
        scenario = rel.parts[1]

        cavs, vehicles, pedestrians, ticks = parse_parameters_from_text_file(txt_path)
        scenarios.append(
            ScenarioParams(
                town=town,
                scenario=scenario,
                cavs=cavs,
                vehicles=vehicles,
                pedestrians=pedestrians,
                ticks=ticks,
                source_file=txt_path,
            )
        )

    return scenarios


def find_scenario_text_from_config(
    cpm_root: Path,
    config_path: Path,
    config: Dict,
    town: str,
    scenario: str,
) -> Optional[Path]:
    root_dir_str = config.get("root_dir")
    if isinstance(root_dir_str, str) and root_dir_str.strip():
        root_dir = Path(root_dir_str)
        direct = root_dir / f"{scenario}.txt"
        if direct.exists():
            return direct

        txts = sorted([p for p in root_dir.glob("*.txt") if p.is_file()])
        if txts:
            return txts[0]

    # Fallback: look next to config under CPM root.
    scenario_dir = cpm_root / town / scenario
    local_txts = sorted([p for p in scenario_dir.glob("*.txt") if p.is_file()])
    if local_txts:
        return local_txts[0]

    return None


def load_all_scenario_params_from_cpm(root: Path) -> List[ScenarioParams]:
    scenarios: List[ScenarioParams] = []
    seen_scenarios: set[Tuple[str, str]] = set()

    for cfg_path in sorted(iter_simulation_configs(root)):
        rel = cfg_path.relative_to(root)
        if len(rel.parts) < 3:
            continue

        town = rel.parts[0]
        scenario = rel.parts[1]
        scenario_key = (town, scenario)

        # Deduplicate repeats (r000/r001) per scenario.
        if scenario_key in seen_scenarios:
            continue

        cfg = load_yaml(cfg_path)
        txt_path = find_scenario_text_from_config(
            cpm_root=root,
            config_path=cfg_path,
            config=cfg,
            town=town,
            scenario=scenario,
        )
        if txt_path is None:
            raise FileNotFoundError(
                f"Could not find scenario .txt for {town}/{scenario} from config: {cfg_path}"
            )

        cavs, vehicles, pedestrians, ticks = parse_parameters_from_text_file(txt_path)
        scenarios.append(
            ScenarioParams(
                town=town,
                scenario=scenario,
                cavs=cavs,
                vehicles=vehicles,
                pedestrians=pedestrians,
                ticks=ticks,
                source_file=txt_path,
            )
        )
        seen_scenarios.add(scenario_key)

    return scenarios


def detect_layout(root: Path) -> str:
    has_sim_cfg = any(iter_simulation_configs(root))
    has_txt = any(iter_txt_files(root))
    if has_sim_cfg:
        return "cpm"
    if has_txt:
        return "txt"
    raise RuntimeError(
        f"No simulation_config*.yaml or .txt files found under: {root}"
    )


def load_all_scenario_params(root: Path) -> Tuple[str, List[ScenarioParams]]:
    layout = detect_layout(root)
    if layout == "cpm":
        return layout, load_all_scenario_params_from_cpm(root)
    return layout, load_all_scenario_params_from_txt(root)


def build_stats_by_town(
    scenarios: List[ScenarioParams],
    non_connected_mode: str,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[ScenarioParams]] = {}
    for item in scenarios:
        grouped.setdefault(item.town, []).append(item)

    stats: Dict[str, Dict[str, float]] = {}
    for town, rows in grouped.items():
        cavs_vals = [r.cavs for r in rows]

        if non_connected_mode == "vehicles_minus_cavs":
            non_connected_vals = [r.vehicles - r.cavs for r in rows]
        else:
            # Treat Vehicles field as non-connected vehicles.
            non_connected_vals = [r.vehicles for r in rows]

        cavs_mean, cavs_std = mean_std(cavs_vals)
        non_conn_mean, non_conn_std = mean_std(non_connected_vals)

        stats[town] = {
            "n_runs": len(rows),
            "cavs_mean": cavs_mean,
            "cavs_std": cavs_std,
            "non_conn_mean": non_conn_mean,
            "non_conn_std": non_conn_std,
        }

    return stats


def render_latex_table(stats: Dict[str, Dict[str, float]]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-map scenario statistics in \mydataset{}. Values are mean $\pm$ standard deviation over runs.}"
    )
    lines.append(r"\label{tab:scenario-config}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Map} & \textbf{CAVs per run} & \textbf{Non-connected vehicles per run} \\")
    lines.append(r"\midrule")

    for town in sorted(stats.keys(), key=town_sort_key):
        s = stats[town]
        lines.append(
            f"{town} & "
            f"{s['cavs_mean']:.2f} $\\pm$ {s['cavs_std']:.2f} & "
            f"{s['non_conn_mean']:.2f} $\\pm$ {s['non_conn_std']:.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-town scenario stats from raw dataset or CPM output."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path("dataset"),
        help="Root path (raw dataset or CPM root).",
    )
    parser.add_argument(
        "--non-connected-mode",
        choices=["vehicles", "vehicles_minus_cavs"],
        default="vehicles",
        help=(
            "How to compute non-connected vehicles: "
            "'vehicles' uses Vehicles directly (default), "
            "'vehicles_minus_cavs' uses Vehicles - CAVs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    layout, scenarios = load_all_scenario_params(root)
    if not scenarios:
        raise RuntimeError(f"No scenarios parsed from: {root}")

    stats = build_stats_by_town(
        scenarios=scenarios,
        non_connected_mode=args.non_connected_mode,
    )

    print(f"Root path: {root}")
    print(f"Detected layout: {layout}")
    print(f"Parsed scenarios: {len(scenarios)}")
    print(f"Non-connected mode: {args.non_connected_mode}")
    print()

    print("Per-town summary")
    for town in sorted(stats.keys(), key=town_sort_key):
        s = stats[town]
        print(
            f"{town}: n={int(s['n_runs'])}, "
            f"CAVs={s['cavs_mean']:.2f} +/- {s['cavs_std']:.2f}, "
            f"Non-connected={s['non_conn_mean']:.2f} +/- {s['non_conn_std']:.2f}"
        )

    print("\nLaTeX table:\n")
    print(render_latex_table(stats))


if __name__ == "__main__":
    main()

