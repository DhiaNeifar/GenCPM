"""
Arrange CooperScene single-agent predictions (output of `python -m gencpm infer`, <src>/<split>/<scene>/<agent>/)
in the layout the GenCPM pipeline reads: <dst>/TownCooperScene/<split>_<scene>_s<k>/<agent>/<ts>_preds.yaml ...

  * Timestamps: CooperScene frames are numbered 1 apart at 0.1 s; CARLA frames 2 apart at 0.05 s per tick. Frames are
    renumbered ts' = 2 * (ts - first ts of the segment), so every time constant of GenCPM, the attacks and the defenses
    (0.05 s per tick) applies unchanged, and the CARLA-trained defenses see speeds in the units they were trained on.
  * Segments: each scene is cut into consecutive segments of --segment frames, so the 4 evaluation scenes give enough
    episodes for the 5-fold protocol (used for threshold calibration only; every transferred model is CARLA-trained).
  * --mode move renames the files (same file system), copy duplicates them (tests). mapping.csv records every frame.

Usage: python -m gencpm cooperscene-layout --src DETECTIONS --dst FRAMES_ROOT
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

SUFFIXES = (".yaml", ".pcd", "_preds.yaml", "_pred.npy", "_score.npy", "_gt.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", nargs="+", default=["validate", "test"])
    ap.add_argument("--town", default="TownCooperScene")
    ap.add_argument("--segment", type=int, default=250, help="target frames per episode (250 = 25 s); scenes are cut into equal segments")
    ap.add_argument("--mode", choices=["move", "copy"], default="copy")
    ap.add_argument("--max-segments", type=int, default=None, help="tests: only the first N segments")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst) / a.town
    rows, n_seg = [], 0
    for split in a.splits:
        for scene in sorted(os.listdir(src / split), key=lambda s: int(s) if s.isdigit() else s):
            sdir = src / split / scene
            agents = sorted(x for x in os.listdir(sdir) if (sdir / x).is_dir())
            stamps = None  # frames with predictions for every agent
            for ag in agents:
                ts = {int(m.group(1)) for f in os.listdir(sdir / ag) if (m := re.fullmatch(r"(\d+)_preds\.yaml", f))}
                stamps = ts if stamps is None else stamps & ts
            stamps = sorted(stamps)
            n_parts = max(1, round(len(stamps) / a.segment))  # equal segments of about --segment frames
            for k in range(n_parts):
                seg = stamps[k * len(stamps) // n_parts:(k + 1) * len(stamps) // n_parts]
                if a.max_segments is not None and n_seg >= a.max_segments:
                    break
                ep = f"{split}_{scene}_s{k}"
                t0 = seg[0]
                for ag in agents:
                    out = dst / ep / ag
                    out.mkdir(parents=True, exist_ok=True)
                    for ts in seg:
                        new = 2 * (ts - t0)
                        for suf in SUFFIXES:
                            f = sdir / ag / f"{ts}{suf}"
                            if not os.path.lexists(f):
                                continue
                            g = out / f"{new:06d}{suf}"
                            if a.mode == "move":
                                os.rename(f, g)
                            else:
                                shutil.copy2(f, g, follow_symlinks=False)
                        rows.append((f"{a.town}/{ep}", ag, f"{new:06d}", split, scene, ts))
                n_seg += 1
                print(f"{a.town}/{ep}: {len(seg)} frames x {len(agents)} agents (source {split}/{scene} {seg[0]}..{seg[-1]})")
    with open(Path(a.dst) / f"mapping_{a.town}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "agent", "timestamp", "split", "scene", "source_frame"])
        w.writerows(rows)
    print(f"{n_seg} episodes, {len(rows):,} agent-frames -> {dst} (mapping_{a.town}.csv)")


if __name__ == "__main__":
    main()
