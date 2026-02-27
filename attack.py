"""Compatibility shim.

Prefer importing from the attacks package:
    from attacks import create_attack, AttackContext
"""

from attacks import (
    ATTACK_REGISTRY,
    AddObjectAttack,
    Attack,
    AttackContext,
    DriftAttack,
    RemoveObjectAttack,
    WhiteNoiseAttack,
    create_attack,
)

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
