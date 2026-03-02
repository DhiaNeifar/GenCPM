#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scenario folders look like: 2026_02_15_12_34_56
SCENARIO_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")


# ----------------------------
# YAML parsing helpers
# ----------------------------
def categorize_road_type(doc: dict) -> str:
    rt = doc.get("road_type", {}) or {}
    junc = rt.get("junction", {}) or {}

    road_class = rt.get("road_class")
    is_ramp = bool(rt.get("is_ramp"))
    is_junction = bool(rt.get("is_junction"))
    junction_type = junc.get("type")

    if road_class == "highway":
        return "ramp" if is_ramp else "highway"
    elif road_class == "street" and is_junction:
        if junction_type == "cross_intersection":
            return "cross_intersection"
        elif junction_type == "t_intersection":
            return "t_intersection"
        else:
            return "street"
    else:
        return "street"


def find_scenario_name(ypath: Path) -> str:
    for p in ypath.parents:
        if SCENARIO_RE.match(p.name):
            return p.name
    return "unknown_scenario"


def find_ego_vehicle_id(ypath: Path) -> str:
    # Your structure seems to be .../<scenario>/<cav_id>/<timestamp>.yaml
    for p in ypath.parents:
        if p.name.isdigit():
            return p.name
    return ypath.parent.name


def process_yaml_file(ypath: Path) -> dict | None:
    try:
        with ypath.open("r", encoding="utf-8", errors="replace") as f:
            # Keep BaseLoader (everything as strings) like your notebook did
            doc = yaml.load(f, Loader=yaml.BaseLoader) or {}

        scenario = find_scenario_name(ypath)
        ego_vehicle_id = find_ego_vehicle_id(ypath)

        stem = ypath.stem
        try:
            timestamp = int(stem)
        except Exception:
            timestamp = stem

        ego_speed = doc.get("ego_speed", None)
        road_type = categorize_road_type(doc)

        vehicles = doc.get("vehicles", {}) or {}
        speeds = []
        for v in vehicles.values():
            if isinstance(v, dict) and "speed" in v:
                speeds.append(v["speed"])

        return {
            "scenario": scenario,
            "timestamp": timestamp,
            "ego_vehicle_id": int(ego_vehicle_id) if str(ego_vehicle_id).isdigit() else str(ego_vehicle_id),
            "ego_vehicle_velocity": ego_speed,
            "road_type": road_type,
            "number_vehicles": len(speeds),
            "vehicles_velocity": speeds,
        }
    except Exception:
        return None


def collect_yaml_files(root: Path) -> list[Path]:
    """
    Expected structure:
      root/
        <map_dir>/
          <scenario_dir YYYY_MM_DD_HH_MM_SS>/
            <cav_id digit>/
              *.yaml
    """
    yaml_files: list[Path] = []

    map_dirs = [p for p in root.iterdir() if p.is_dir()]
    for map_dir in map_dirs:
        scenario_dirs = [p for p in map_dir.iterdir() if p.is_dir() and SCENARIO_RE.match(p.name)]
        for scenario_dir in scenario_dirs:
            cav_dirs = [p for p in scenario_dir.iterdir() if p.is_dir() and p.name.isdigit()]
            for cav_dir in cav_dirs:
                yaml_files.extend(cav_dir.glob("*.yaml"))

    return yaml_files


def load_cpm_yaml_dataset(path: str | Path, n_workers: int | None = None) -> pd.DataFrame:
    root = Path(path)
    if n_workers is None:
        n_workers = min(32, cpu_count() * 4)

    # 1) Collect YAML files
    yaml_files = collect_yaml_files(root)

    if not yaml_files:
        return pd.DataFrame(
            columns=[
                "scenario",
                "timestamp",
                "ego_vehicle_id",
                "ego_vehicle_velocity",
                "road_type",
                "number_vehicles",
                "vehicles_velocity",
            ]
        )

    # 2) Parse YAMLs in parallel + tqdm in terminal
    rows = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(process_yaml_file, ypath) for ypath in yaml_files]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading YAMLs",
            unit="file",
            file=sys.stderr,  # tqdm default; shows nicely in terminal
            dynamic_ncols=True,
        ):
            r = fut.result()
            if r is not None:
                rows.append(r)

    df = pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "timestamp",
            "ego_vehicle_id",
            "ego_vehicle_velocity",
            "road_type",
            "number_vehicles",
            "vehicles_velocity",
        ],
    )

    if not df.empty:
        df = df.sort_values(["scenario", "ego_vehicle_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    return df


# ----------------------------
# Scenario meta (.txt) loader
# ----------------------------
def load_scenario_meta_df(root_path: str | Path) -> pd.DataFrame:
    root = Path(root_path)
    rows = []

    for map_dir in root.iterdir():
        if not map_dir.is_dir():
            continue

        for scenario_dir in map_dir.iterdir():
            if not (scenario_dir.is_dir() and SCENARIO_RE.match(scenario_dir.name)):
                continue

            txt_files = list(scenario_dir.glob("*.txt"))
            if not txt_files:
                continue

            txt_path = txt_files[0]  # assume one
            lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()

            map_name = None
            cavs = None
            non_cavs = None

            if len(lines) >= 2 and lines[1].startswith("Map:"):
                map_name = lines[1].split(":", 1)[1].strip()

            if len(lines) >= 3 and lines[2].startswith("Parameters:"):
                params = lines[2].split(":", 1)[1].strip()
                kv = {}
                for part in params.split(","):
                    if "=" in part:
                        k, v = part.strip().split("=", 1)
                        kv[k.strip()] = v.strip()
                cavs = int(kv.get("CAVs")) if kv.get("CAVs", "").isdigit() else None
                non_cavs = int(kv.get("Vehicles")) if kv.get("Vehicles", "").isdigit() else None

            rows.append(
                {
                    "scenario": scenario_dir.name,
                    "map": map_name,
                    "CAVs": cavs,
                    "Non CAVs": non_cavs,
                }
            )

    return pd.DataFrame(rows, columns=["scenario", "map", "CAVs", "Non CAVs"])


# ----------------------------
# Stats table builder
# ----------------------------
def build_stats(df: pd.DataFrame, meta_df: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.copy()

    # % coverage by road_type
    pct = d["road_type"].value_counts(normalize=True) * 100

    # density = detected vehicles per frame
    d["density_detected"] = pd.to_numeric(d["number_vehicles"], errors="coerce").fillna(0.0)

    # speed per frame = mean(ego + detected vehicles)
    def frame_mean_speed(r):
        speeds = []
        ego = r.get("ego_vehicle_velocity", None)
        if pd.notna(ego):
            try:
                speeds.append(float(ego))
            except Exception:
                pass

        vv = r.get("vehicles_velocity", [])
        if isinstance(vv, (list, tuple)):
            for x in vv:
                if x is None:
                    continue
                try:
                    speeds.append(float(x))
                except Exception:
                    pass

        return float(np.mean(speeds)) if speeds else np.nan

    d["speed_frame_mean"] = d.apply(frame_mean_speed, axis=1)

    # attach scenario-level CAVs if provided
    if meta_df is not None and {"scenario", "CAVs"}.issubset(meta_df.columns):
        d = d.merge(meta_df[["scenario", "CAVs"]], on="scenario", how="left")
    else:
        d["CAVs"] = np.nan

    g = d.groupby("road_type", dropna=False)

    dens_m, dens_s = g["density_detected"].mean(), g["density_detected"].std(ddof=1)
    spd_m, spd_s = g["speed_frame_mean"].mean(), g["speed_frame_mean"].std(ddof=1)
    cav_m, cav_s = g["CAVs"].mean(), g["CAVs"].std(ddof=1)

    order = ["highway", "street", "t_intersection", "cross_intersection"]
    pretty = {
        "highway": "Highway",
        "street": "Street",
        "t_intersection": "T-intersection",
        "cross_intersection": "Cross-intersection",
    }

    def fmt(m, s):
        return f"{m:.2f} ± {s:.2f}" if pd.notna(m) else "NA"

    stats = pd.DataFrame(
        {
            "Road Type": [pretty[k] for k in order],
            "Percentage (%)": pct.reindex(order).fillna(0).round(2).to_numpy(),
            "Traffic Density (m±s)": [fmt(dens_m.get(k, np.nan), dens_s.get(k, np.nan)) for k in order],
            "CAV Density (m±s)": [fmt(cav_m.get(k, np.nan), cav_s.get(k, np.nan)) for k in order],
            "Traffic Speed (m±s)": [fmt(spd_m.get(k, np.nan), spd_s.get(k, np.nan)) for k in order],
        }
    ).set_index("Road Type")

    return stats


# ----------------------------
# CLI / main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Load CPM YAML dataset and build road-type stats table.")
    ap.add_argument("--path", required=True, help="Root dataset path (the folder that contains map folders).")
    ap.add_argument("--workers", type=int, default=None, help="Thread workers for YAML loading (default: auto).")
    ap.add_argument("--save-csv", default=None, help="Optional output CSV path for the stats table.")
    args = ap.parse_args()

    path = args.path

    df = load_cpm_yaml_dataset(path, n_workers=args.workers)
    if df.empty:
        print("No YAML files found. Stats table is empty.", file=sys.stderr)
        print(pd.DataFrame())
        return 0

    # normalize ramp -> highway (like your notebook)
    df["road_type"] = df["road_type"].replace("ramp", "highway")

    meta_df = load_scenario_meta_df(path)
    stats = build_stats(df, meta_df)

    # Print results
    print("\nRoad type counts:")
    print(df["road_type"].value_counts(dropna=False))
    print("\nStats table:")
    print(stats)

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(out)
        print(f"\nSaved stats CSV to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

