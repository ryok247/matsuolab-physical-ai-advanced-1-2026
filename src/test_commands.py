"""Meaningful, dependency-free tests for command boundaries and task scoring."""
import math
import unittest
from commands import CONFIG, cases, command_at, score_trial


def ideal_rows(stage="full", mirror=1):
    result = []
    for step in range(1500):
        segment, c = command_at(step, stage, mirror)
        result.append({"step": step, "time": (step+1)*0.02, "segment": segment,
                       "vx": c[0], "vy": 0.0, "wz": c[2], "done": False,
                       "roll": 0.0, "pitch": 0.0})
    return result


class CommandsTest(unittest.TestCase):
    def test_switch_uses_next_observation_boundary(self):
        self.assertEqual(command_at(149)[1], [0, 0, 0])
        self.assertEqual(command_at(150)[1], [0.2, 0, 0])
        self.assertEqual(command_at(750)[1], [0, 0, 0.5])
        self.assertEqual(command_at(750, mirror=-1)[1], [0, 0, -0.5])
        with self.assertRaises(ValueError):
            command_at(1500)

    def test_all_conditions_are_required(self):
        self.assertTrue(score_trial(ideal_rows())["success"])
        self.assertFalse(score_trial(ideal_rows()[:-1])["success"])
        rows = ideal_rows()
        rows[300]["done"] = True
        self.assertFalse(score_trial(rows)["success"])

    def test_standing_cannot_pass_motion_or_turns(self):
        rows = ideal_rows()
        for r in rows:
            r.update(vx=0, vy=0, wz=0)
        metrics = score_trial(rows)
        self.assertFalse(metrics["success"])
        self.assertTrue(metrics["segments"][0]["success"])
        self.assertFalse(metrics["segments"][4]["success"])

    def test_stop_requires_half_second_continuous_dwell(self):
        rows = ideal_rows()
        for r in rows:
            if r["segment"] == 3:
                r["vx"] = 0 if r["step"] % 24 < 23 else 0.1
        self.assertIsNone(score_trial(rows)["segments"][3]["settled_at_s"])

    def test_slow_stop_is_failure_even_if_late_mae_is_good(self):
        rows = ideal_rows()
        for r in rows:
            if r["segment"] == 3 and r["time"] < 13.1:
                r["vx"] = 0.06
        seg = score_trial(rows)["segments"][3]
        self.assertLess(seg["vx_mae"], 0.1)
        self.assertFalse(seg["success"])

    def test_formal_cases_include_both_turns_and_yaws(self):
        c = cases(formal=True)
        self.assertEqual(len(c), 30)
        self.assertEqual({r["yaw_degrees"] for r in c}, {-15, 0, 15})
        self.assertEqual({r["mirror"] for r in c}, {-1, 1})
        self.assertTrue(set(CONFIG["formal_seeds"]).isdisjoint({r["seed"] for r in cases()}))


if __name__ == "__main__":
    unittest.main()
