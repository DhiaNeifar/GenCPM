"""Dataset and output locations, read from configs/paths.yaml (copied from configs/paths.example.yaml).

No path of any machine is stored in the repository: every location comes from this file, from the environment
variable GENCPM_PATHS (another YAML file), or from command-line options.
"""

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
KEYS = ("carla_frames", "carla_frames_second", "cooperscene_frames", "work")


def load(path=None):
    path = Path(path or os.environ.get("GENCPM_PATHS") or REPO / "configs" / "paths.yaml")
    if not path.exists():
        raise SystemExit(f"{path} not found: copy configs/paths.example.yaml to configs/paths.yaml and set your paths")
    cfg = yaml.safe_load(path.read_text()) or {}
    return {k: (str(Path(v).expanduser()) if v else "") for k, v in cfg.items() if k in KEYS}


def main(argv=None):
    """`python -m gencpm paths --shell` prints the paths as shell variables for the reproduce/ scripts."""
    import argparse
    ap = argparse.ArgumentParser(prog="gencpm paths", description=main.__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shell", action="store_true")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    for k in KEYS:
        v = cfg.get(k, "")
        print(f"{k.upper()}={v!r}" if a.shell else f"{k:22s} {v or '(not set)'}")
