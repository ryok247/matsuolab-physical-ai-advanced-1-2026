"""Command schedules and trajectory scoring; usable without Torch/Genesis."""
import json
import math
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name("command-config.json").read_text())
MODES = ("stop", "straight", "turn", "arc")


def schedule(stage="full", mirror=1):
    if mirror not in (-1, 1):
        raise ValueError("mirror must be +1 or -1")
    end = [3, 7, 11, 15, 19, 23, 27, 30]
    commands = [(0, 0, 0), (0.2, 0, 0), (0.7, 0, 0), (0, 0, 0),
                (0, 0, 0.5 * mirror), (0, 0, -0.5 * mirror),
                (0.4, 0, 0.4 * mirror), (0, 0, 0)]
    if stage == "speed":
        commands = [(0, 0, 0), (0.2, 0, 0), (0.5, 0, 0), (0, 0, 0),
                    (0.3, 0, 0), (0, 0, 0), (0.5, 0, 0), (0, 0, 0)]
    return [{"start": 0 if i == 0 else end[i-1], "end": end[i], "command": list(c)}
            for i, c in enumerate(commands)]


def command_at(step, stage="full", mirror=1):
    # Integer boundaries avoid drift from repeatedly adding floating point dt.
    for index, segment in enumerate(schedule(stage, mirror)):
        if step < round(segment["end"] / CONFIG["dt"]):
            return index, segment["command"]
    raise ValueError("step outside schedule")


def cases(formal=False, stage="full", recording=False):
    if recording:
        return [{"seed": 90, "yaw_degrees": 0, "mirror": 1}]
    seeds = CONFIG["formal_seeds"] if formal else [90]
    yaws = CONFIG["formal_yaw_degrees"] if formal else [0]
    mirrors = CONFIG["mirrors"] if stage == "full" else [1]
    return [{"seed": seed, "yaw_degrees": yaw, "mirror": mirror}
            for yaw in yaws for mirror in mirrors for seed in seeds]


def score_trial(rows, stage="full", mirror=1, thresholds=None):
    limits = thresholds or CONFIG["thresholds"]
    dt = CONFIG["dt"]
    segments = []
    failure = any(r["done"] for r in rows)
    for index, seg in enumerate(schedule(stage, mirror)):
        group = [r for r in rows if r["segment"] == index]
        cmd = seg["command"]
        stop = cmd == [0, 0, 0]
        turn = cmd[0] == 0 and cmd[2] != 0
        stable = [r for r in group if r["time"] - seg["start"] > limits["exclude_s"] and not r["done"]]
        def mean(fn):
            return sum(fn(r) for r in stable) / len(stable) if stable else None
        vx = mean(lambda r: abs(r["vx"] - cmd[0]))
        vy = mean(lambda r: abs(r["vy"]))
        yaw = mean(lambda r: abs(r["wz"] - cmd[2]))
        xy = mean(lambda r: math.hypot(r["vx"], r["vy"]))
        consecutive, settled = 0, None
        for row in group:
            if stop:
                good = math.hypot(row["vx"], row["vy"]) < limits["stop_xy_speed"] and abs(row["wz"]) < limits["stop_yaw_speed"]
            else:
                good = abs(row["vx"]-cmd[0]) <= limits["vx_mae"] and abs(row["vy"]) <= limits["vy_mae"] and abs(row["wz"]-cmd[2]) <= limits["yaw_mae"]
                if turn:
                    good = good and math.hypot(row["vx"], row["vy"]) <= limits["turn_xy_speed"]
            consecutive = consecutive + 1 if good and not row["done"] else 0
            if settled is None and consecutive >= round(limits["dwell_s"] / dt):
                settled = row["time"] - seg["start"]
        complete = len(group) == round((seg["end"]-seg["start"]) / dt)
        tracked = vx is not None and vx <= limits["vx_mae"] and vy <= limits["vy_mae"] and yaw <= limits["yaw_mae"]
        if turn:
            tracked = tracked and xy <= limits["turn_xy_speed"]
        distance = sum(math.hypot(r["vx"], r["vy"]) * dt for r in group)
        segments.append({"index": index, **seg, "steps": len(group), "vx_mae": vx, "vy_mae": vy,
                         "yaw_mae": yaw, "xy_speed_mean": xy, "settled_at_s": settled,
                         "travel_distance_m": distance,
                         "success": bool(complete and tracked and settled is not None and settled <= limits["settle_s"] and not any(r["done"] for r in group))})
    return {"steps": len(rows), "terminated_early": failure,
            "success": len(rows) == 1500 and not failure and all(s["success"] for s in segments),
            "max_abs_roll_degrees": max((abs(r["roll"]) for r in rows), default=None),
            "max_abs_pitch_degrees": max((abs(r["pitch"]) for r in rows), default=None),
            "segments": segments}
