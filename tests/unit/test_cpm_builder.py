import unittest

from cpm_builder import build_cpm


class TestCpmBuilder(unittest.TestCase):
    def test_build_cpm_assigns_unique_ids(self) -> None:
        frame = {
            "timestamp": 42,
            "ego_speed": 3.5,
            "true_ego_pos": [10.0, 20.0, 1.0],
            "vehicles": {
                "100": {
                    "location": [1.0, 2.0, 3.0],
                    "center": [1.0, 2.0, 3.0],
                    "angle": [0.0, 90.0, 0.0],
                    "extent": [2.0, 1.0, 1.0],
                    "speed": 8.0,
                },
                "200": {
                    "location": [4.0, 5.0, 6.0],
                    "center": [4.0, 5.0, 6.0],
                    "angle": [0.0, 180.0, 0.0],
                    "extent": [2.2, 1.1, 1.0],
                    "speed": 9.0,
                },
            },
        }

        cpm = build_cpm(frame)

        ids = [cpm["vehicle_id"]] + [v["id"] for v in cpm["detected_vehicles"]]
        self.assertEqual(cpm["timestamp"], 42)
        self.assertEqual(cpm["vehicle_speed"], 3.5)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(cpm["detected_vehicles"]), 2)


if __name__ == "__main__":
    unittest.main()
