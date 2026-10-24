from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# C-accelerated loader/dumper when libyaml is available (same semantics, much faster).
_LOADER = getattr(yaml, "CUnsafeLoader", yaml.UnsafeLoader)
_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as file_obj:
        return yaml.load(file_obj, Loader=_LOADER)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        yaml.dump(
            data,
            file_obj,
            Dumper=_DUMPER,
            sort_keys=False,
            default_flow_style=False,
        )
