from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AttackContext:
    """Extra info about where/when the CPM is generated."""

    vehicle_id: str
    timestamp: float
    # Optional context for stateful/advanced attacks (see advanced_attacks.py).
    history: Optional[List[Tuple[int, List[Dict[str, Any]]]]] = None
    others: Optional[Dict[str, Tuple[float, float]]] = None


class Attack:
    """
    Base class: modifies CPM in-place and returns metadata dict.

    Expected metadata keys:
      - attack_type
      - attack_parameters
      - target_ids
      - altered_fields
      - ground_truth_reference
    """

    attack_type: str = "BaseAttack"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def apply(self, cpm: Dict[str, Any], ctx: AttackContext) -> Dict[str, Any]:
        raise NotImplementedError
