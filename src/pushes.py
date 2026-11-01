"""Pure functions for timed physical pushes and recovery scoring."""
import math

DT = .02
LEVELS = (.10, .20, .30)


def lateral_force(yaw, sign, alpha, mass):
    if sign not in (-1, 1) or not (mass > 0 and math.isfinite(mass)) or not (0 <= alpha <= .30):
        raise ValueError('Invalid push parameters')
    magnitude = sign * alpha * mass * 9.81
    return [-math.sin(yaw)*magnitude, math.cos(yaw)*magnitude, 0.0]


def pulse_active(step, start=200, duration=10):
    return start <= step < start + duration


def cases(formal=False):
    seeds = range(2000, 2010) if formal else range(90, 95)
    return [dict(alpha=a, sign=s, seed=seed) for a in LEVELS for s in (-1, 1) for seed in seeds]


def score(rows, force_end=4.2):
    def good(r):
        return not r['done'] and abs(r['vx']-.5) <= .1 and abs(r['vy']) <= .1 and abs(r['wz']) <= .15
    before = [r for r in rows if 3.5 < r['time'] <= 4.0]
    streak, recovery = 0, None
    for r in rows:
        if r['time'] <= force_end:
            continue
        streak = streak+1 if good(r) else 0
        if streak >= 25 and recovery is None:
            recovery = r['time']-force_end
    complete = len(rows) == 600 and not any(r['done'] for r in rows)
    return dict(success=bool(complete and len(before)==25 and all(map(good,before)) and recovery is not None and recovery <= 2),
                recovery_s=recovery, completed=complete)
