from .content_fabrication import AddObjectAttack
from .base import Attack, AttackContext
from .cumulative_drift import DriftAttack
from .registry import ATTACK_REGISTRY, create_attack
from .content_suppression import RemoveObjectAttack
from .spatial_perturbation import WhiteNoiseAttack

__all__ = [
    "Attack",
    "AttackContext",
    "DriftAttack",
    "AddObjectAttack",
    "RemoveObjectAttack",
    "WhiteNoiseAttack",
    "ATTACK_REGISTRY",
    "create_attack",
]
