import copy

import pytest

from gencpm.attacks import AttackContext, create_attack
from gencpm.attacks.registry import ATTACK_REGISTRY

SINGLE = ["AddObjectAttack", "RemoveObjectAttack", "DriftAttack", "WhiteNoiseAttack", "GhostTrackAttack",
          "StealthDriftAttack", "ReplayAttack", "PoseFalsificationAttack"]  # CF CS CSD ISP GT SD RP PF


def make_cpm() -> dict:
    vehicles = [{"id": i, "track_id": i, "location": {"x": 5.0 * i, "y": 2.0 * i, "z": 0.0},
                 "center": {"x": 0.0, "y": 0.0, "z": 0.8}, "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 10.0 * i},
                 "extent": {"x": 2.2, "y": 0.9, "z": 0.8}, "speed": 0.0} for i in range(1, 6)]
    return {"global_position": {"x": 0.0, "y": 0.0, "z": 0.0}, "detected_vehicles": vehicles}


def test_all_eight_single_attacker_attacks_are_registered():
    assert set(SINGLE) <= set(ATTACK_REGISTRY)


def test_unknown_attack_raises():
    with pytest.raises(ValueError):
        create_attack("UnknownAttack")


def test_content_fabrication_adds_objects():
    cpm = make_cpm()
    meta = create_attack("AddObjectAttack", {"num_objects": 2, "radius_range": (1.0, 2.0)}).apply(
        cpm, AttackContext(vehicle_id="ego", timestamp=0.0))
    assert meta["attack_type"] == "AddObjectAttack" and len(meta["target_ids"]) == 2
    assert len(cpm["detected_vehicles"]) == 7


def test_content_suppression_removes_targets():
    cpm = make_cpm()
    meta = create_attack("RemoveObjectAttack", {"target_ids": [1]}).apply(cpm, AttackContext(vehicle_id="ego", timestamp=0.0))
    ids = {v["id"] for v in cpm["detected_vehicles"]}
    assert meta["attack_type"] == "RemoveObjectAttack" and 1 not in ids and 2 in ids


@pytest.mark.parametrize("name", ["DriftAttack", "WhiteNoiseAttack", "PoseFalsificationAttack"])
def test_perturbation_attacks_change_the_content(name):
    attack = create_attack(name, {})
    changed = False
    for t in range(0, 40, 2):  # attacks may start slowly; check that the content changes within the interval
        cpm = make_cpm(); before = copy.deepcopy(cpm)
        attack.apply(cpm, AttackContext(vehicle_id="ego", timestamp=float(t)))
        changed |= cpm != before
    assert changed
