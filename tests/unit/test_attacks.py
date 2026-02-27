import unittest

from attacks import AttackContext, create_attack


def make_cpm() -> dict:
    return {
        "global_position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "detected_vehicles": [
            {
                "id": 1,
                "location": {"x": 1.0, "y": 1.0, "z": 0.0},
                "center": {"x": 1.0, "y": 1.0, "z": 0.0},
                "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "extent": {"x": 2.0, "y": 1.0, "z": 1.0},
                "speed": 4.0,
            },
            {
                "id": 2,
                "location": {"x": 2.0, "y": 2.0, "z": 0.0},
                "center": {"x": 2.0, "y": 2.0, "z": 0.0},
                "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "extent": {"x": 2.5, "y": 1.2, "z": 1.1},
                "speed": 6.0,
            },
        ],
    }


class TestAttacks(unittest.TestCase):
    def test_factory_unknown_attack_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_attack("UnknownAttack")

    def test_add_object_attack_adds_objects(self) -> None:
        cpm = make_cpm()
        attack = create_attack("AddObjectAttack", {"num_objects": 2, "radius_range": (1.0, 2.0)})
        meta = attack.apply(cpm, AttackContext(vehicle_id="ego", timestamp=0.0))

        self.assertEqual(meta["attack_type"], "AddObjectAttack")
        self.assertEqual(len(meta["target_ids"]), 2)
        self.assertEqual(len(cpm["detected_vehicles"]), 4)

    def test_remove_object_attack_removes_targets(self) -> None:
        cpm = make_cpm()
        attack = create_attack("RemoveObjectAttack", {"target_ids": [1]})
        meta = attack.apply(cpm, AttackContext(vehicle_id="ego", timestamp=0.0))

        remaining_ids = {v["id"] for v in cpm["detected_vehicles"]}
        self.assertEqual(meta["attack_type"], "RemoveObjectAttack")
        self.assertNotIn(1, remaining_ids)
        self.assertIn(2, remaining_ids)


if __name__ == "__main__":
    unittest.main()
