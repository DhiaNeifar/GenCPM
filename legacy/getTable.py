from pathlib import Path
import re
import pandas as pd


# -----------------------------
# 1) Parse per-run metadata
# -----------------------------
def parse_run_metadata(root_dir: str) -> pd.DataFrame:
    """
    Expects a structure like:
      root_dir/
        Town01/
          2025_12_01_12_00_00/
            meta.txt   (or any .txt containing Map:/Parameters:)
          2025_12_01_12_10_00/
            ...
        Town02/
          ...

    The .txt file should contain lines like:
      Map: Town02
      Parameters: CAVs=2, Vehicles=55, Pedestrians=0, ...
    """
    root = Path(root_dir)
    rows = []

    for town_dir in root.iterdir():
        if not town_dir.is_dir():
            continue

        # each subdir = one run (timestamp folder)
        for run_dir in town_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # find a candidate txt (adjust pattern if needed)
            txt_files = list(run_dir.glob("*.txt"))
            if not txt_files:
                continue

            # pick the first txt that contains "Map:" or "Parameters:"
            meta_path = None
            for p in txt_files:
                try:
                    s = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if ("Map:" in s) or ("Parameters:" in s):
                    meta_path = p
                    break
            if meta_path is None:
                continue

            text = meta_path.read_text(encoding="utf-8", errors="replace")

            # Map line
            m_map = re.search(r"^\s*Map:\s*([A-Za-z0-9_]+)\s*$", text, flags=re.MULTILINE)
            map_name = m_map.group(1) if m_map else town_dir.name  # fallback: folder name

            # Parameters line (extract key=value pairs)
            m_params = re.search(r"^\s*Parameters:\s*(.+?)\s*$", text, flags=re.MULTILINE)
            params_str = m_params.group(1) if m_params else ""

            pairs = re.findall(r"([A-Za-z_]+)\s*=\s*([0-9]+)", params_str)
            params = {k: int(v) for k, v in pairs}

            # Required (at least these two)
            if "CAVs" not in params or "Vehicles" not in params:
                # skip runs that don't have enough info
                continue

            rows.append({
                "map": map_name,
                "run_id": run_dir.name,
                "cavs": params["CAVs"],
                "vehicles_total": params["Vehicles"],
                # optional if present in your meta:
                "ncv_logged": params.get("NCV", None),
            })

    return pd.DataFrame(rows)


# -----------------------------
# 2) Compute mean ± std per map
# -----------------------------
def compute_scenario_stats(meta_df: pd.DataFrame, vehicles_total_includes_cavs: bool = True) -> pd.DataFrame:
    """
    Returns a per-map stats table with mean/std for:
      - cavs
      - ncv (non-connected vehicles)

    If vehicles_total_includes_cavs=True:
        ncv = vehicles_total - cavs
    Else:
        ncv = vehicles_total
    If you log NCV explicitly (NCV=...), we use that when available.
    """
    df = meta_df.copy()

    # Build NCV series
    if "ncv_logged" in df.columns and df["ncv_logged"].notna().any():
        df["ncv"] = df["ncv_logged"]
    else:
        df["ncv"] = (df["vehicles_total"] - df["cavs"]) if vehicles_total_includes_cavs else df["vehicles_total"]

    # group stats
    g = df.groupby("map", as_index=False).agg(
        cavs_mean=("cavs", "mean"),
        cavs_std=("cavs", "std"),
        ncv_mean=("ncv", "mean"),
        ncv_std=("ncv", "std"),
        runs=("run_id", "count"),
    )

    # if a map has 1 run, std becomes NaN; replace by 0.0 for LaTeX niceness
    for c in ["cavs_std", "ncv_std"]:
        g[c] = g[c].fillna(0.0)

    return g


# -----------------------------
# 3) Emit LaTeX rows
# -----------------------------
def format_pm(mu: float, sigma: float, decimals: int = 2) -> str:
    return f"{mu:.{decimals}f} $\\pm$ {sigma:.{decimals}f}"

def emit_latex_rows(stats_df: pd.DataFrame, order=None, decimals: int = 2) -> str:
    """
    Produces lines like:
      Town01 & 2.00 ± 0.00 & 53.20 ± 1.30 \\
    """
    df = stats_df.copy()

    if order is not None:
        # order: ["Town01","Town02",...]
        df["__ord"] = df["map"].apply(lambda x: order.index(x) if x in order else 10**9)
        df = df.sort_values("__ord").drop(columns="__ord")
    else:
        df = df.sort_values("map")

    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"{r['map']} & {format_pm(r['cavs_mean'], r['cavs_std'], decimals)}"
            f" & {format_pm(r['ncv_mean'], r['ncv_std'], decimals)} \\\\"
        )
    return "\n".join(lines)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    ROOT = "/Users/dhianeifar/Desktop/Projects/CPM/train"  # <-- change this

    meta_df = parse_run_metadata(ROOT)

    # If your "Vehicles" counts ALL vehicles including CAVs, keep True:
    stats = compute_scenario_stats(meta_df, vehicles_total_includes_cavs=True)

    latex = emit_latex_rows(stats, order=["Town01", "Town02", "Town03", "Town04", "Town06"], decimals=2)
    print(latex)
