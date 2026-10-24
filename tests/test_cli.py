import subprocess
import sys


def test_help_lists_every_stage():
    out = subprocess.run([sys.executable, "-m", "gencpm", "--help"], capture_output=True, text=True, check=True).stdout
    for cmd in ("frames", "generate", "cache", "coordinated", "features", "fit", "train", "apply", "baselines",
                "transfer", "summarize", "fusion", "sweep", "table-main", "figures", "paths", "download"):
        assert cmd in out
