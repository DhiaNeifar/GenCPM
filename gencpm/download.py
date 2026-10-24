"""Download the AdverCPM release from Hugging Face and unpack it into the layout the code reads.

    python -m gencpm download --to /path/to/AdverCPM                      # detections + models (~2 GB)
    python -m gencpm download --to /path/to/AdverCPM --parts raw               # raw sensors, ~1.1 TB

Each episode is stored as one archive (frames/<detector>/Town*/<episode>.tar.gz) and unpacked to
frames/<detector>/Town*/<episode>/<vehicle>/<frame>_preds.yaml and _score.npy. Then set in configs/paths.yaml:
carla_frames: <to>/frames/pointpillars, carla_frames_second: <to>/frames/second.
"""

import argparse
import tarfile
from pathlib import Path

REPO_ID = "DhiaNeifar/AdverCPM"
PARTS = {"frames/pointpillars": "PointPillars detections (Tables 6, 8, Figs. 5-8)",
         "frames/second": "SECOND detections (Table 9)",
         "models": "trained temporal models and the PointPillars detector trained on CooperScene",
         "raw": "raw CARLA sensor data: RGB cameras, LiDAR, poses, ground truth (~1.1 TB; not needed to reproduce)"}


def unpack(root: Path, keep: bool):
    # frames/<detector>/Town*/<episode>.tar.gz -> <episode>/...; raw/Town*/<episode>/<vehicle>.tar -> <episode>/<vehicle>/...
    archives = [(a, a.parent, a.parent / a.name[:-len(".tar.gz")]) for a in sorted(root.glob("frames/*/Town*/*.tar.gz"))]
    archives += [(a, a.parent.parent, a.parent / a.stem) for a in sorted(root.glob("raw/Town*/*/*.tar"))]
    for i, (a, into, dst) in enumerate(archives, 1):
        if not dst.is_dir():
            print(f"  [{i}/{len(archives)}] unpack {a.relative_to(root)}", flush=True)
            with tarfile.open(a) as t:
                try:
                    t.extractall(into, filter="data")
                except TypeError:  # Python without extraction filters
                    t.extractall(into)
        if not keep:
            a.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gencpm download", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", required=True, help="local folder of the release")
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=["frames/pointpillars", "frames/second", "models"])
    ap.add_argument("--keep-archives", action="store_true", help="keep the .tar.gz files after unpacking")
    a = ap.parse_args(argv)
    from huggingface_hub import snapshot_download
    root = Path(a.to).expanduser()
    for p in a.parts:
        print(f"download {p}: {PARTS[p]}", flush=True)
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=root, allow_patterns=[f"{p}/**" for p in a.parts] + ["README.md"])
    unpack(root, a.keep_archives)
    print(f"done. In configs/paths.yaml set:\n  carla_frames: {root / 'frames/pointpillars'}\n"
          f"  carla_frames_second: {root / 'frames/second'}")
