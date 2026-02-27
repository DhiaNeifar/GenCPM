from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as file_obj:
        return yaml.load(file_obj, Loader=yaml.UnsafeLoader)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        yaml.safe_dump(
            data,
            file_obj,
            sort_keys=False,
            default_flow_style=False,
        )
