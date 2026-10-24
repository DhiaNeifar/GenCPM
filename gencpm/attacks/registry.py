from __future__ import annotations

from typing import Any, Dict, Type

from .content_fabrication import AddObjectAttack
from .base import Attack
from .cumulative_drift import DriftAttack
from .content_suppression import RemoveObjectAttack
from .spatial_perturbation import WhiteNoiseAttack
from .track_attacks import (EvasiveGhostAttack, GhostTrackAttack, PoseFalsificationAttack, ReplayAttack,
                               StealthDriftAttack)


ATTACK_REGISTRY: Dict[str, Type[Attack]] = {
    "DriftAttack": DriftAttack,
    "AddObjectAttack": AddObjectAttack,
    "RemoveObjectAttack": RemoveObjectAttack,
    "WhiteNoiseAttack": WhiteNoiseAttack,
    "GhostTrackAttack": GhostTrackAttack,
    "StealthDriftAttack": StealthDriftAttack,
    "ReplayAttack": ReplayAttack,
    "EvasiveGhostAttack": EvasiveGhostAttack,
    "PoseFalsificationAttack": PoseFalsificationAttack,
}


def create_attack(attack_type: str, params: Dict[str, Any] | None = None) -> Attack:
    attack_cls = ATTACK_REGISTRY.get(attack_type)
    if attack_cls is None:
        raise ValueError(f"Unknown attack_type: {attack_type}")
    return attack_cls(params=params)
