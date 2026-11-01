import math
import unittest
from pushes import lateral_force, pulse_active, cases, score


class PushTests(unittest.TestCase):
    def test_pulse(self):
        self.assertEqual(sum(pulse_active(i) for i in range(600)), 10)
        self.assertFalse(pulse_active(210))
    def test_frame(self):
        f=lateral_force(math.pi/2, 1, .2, 10)
        self.assertAlmostEqual(f[0], -19.62)
        self.assertAlmostEqual(f[1], 0)
        self.assertEqual(f[2], 0)
    def test_cases(self):
        self.assertEqual(len(cases(True)),60)
        self.assertFalse({c['seed'] for c in cases(True)} & {c['seed'] for c in cases()})
    def test_recovery(self):
        rows=[dict(time=(i+1)*.02,vx=.5,vy=0,wz=0,done=False) for i in range(600)]
        self.assertTrue(score(rows)['success'])
        rows[599]['done']=True
        self.assertFalse(score(rows)['success'])
        self.assertFalse(score(rows[:400])['success'])
    def test_dwell(self):
        rows=[dict(time=(i+1)*.02,vx=.5,vy=.2 if i%20==0 else 0,wz=0,done=False) for i in range(600)]
        self.assertIsNone(score(rows)['recovery_s'])


if __name__ == '__main__': unittest.main()
