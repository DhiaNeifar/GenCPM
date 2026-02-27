from __future__ import annotations

from typing import Any, Dict, Type

from .add_object_attack import AddObjectAttack
from .base import Attack
from .drift_attack import DriftAttack
from .remove_object_attack import RemoveObjectAttack
from .white_noise_attack import WhiteNoiseAttack


ATTACK_REGISTRY: Dict[str, Type[Attack]] = {
    "DriftAttack": DriftAttack,
    "AddObjectAttack": AddObjectAttack,
    "RemoveObjectAttack": RemoveObjectAttack,
    "WhiteNoiseAttack": WhiteNoiseAttack,
}


def create_attack(attack_type: str, params: Dict[str, Any] | None = None) -> Attack:
    attack_cls = ATTACK_REGISTRY.get(attack_type)
    if attack_cls is None:
        raise ValueError(f"Unknown attack_type: {attack_type}")
    return attack_cls(params=params)
