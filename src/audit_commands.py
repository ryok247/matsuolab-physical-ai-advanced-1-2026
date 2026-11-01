"""Audit saved CSV/video/Drive copies without rerunning learning or physics."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(session):
    state = json.loads((session / "status.json").read_text())
    assert state["phase"] == "complete"
    destination = Path(state["archive"]).parent
    result = {"scope": "saved trajectory audit; no physics rerun", "evaluations": []}
    # Independently spell out the intended schedule; do not reuse score_trial.
    ends = [3, 7, 11, 15, 19, 23, 27, 30]
    targets = [(0, 0), (.2, 0), (.7, 0), (0, 0), (0, .5), (0, -.5), (.4, .4), (0, 0)]
    formal_jobs = [j for j in state["history"] if j["metrics"].get("formal")]
    assert len(formal_jobs) == 2
    initial_states = []
    for job in formal_jobs:
        run = Path(job["run"])
        groups = {}
        with (run / "rollout.csv").open() as stream:
            for raw in csv.DictReader(stream):
                row = {k: float(raw[k]) for k in ["time", "vx", "vy", "wz", "cmd_vx", "cmd_yaw"]}
                row.update(step=int(raw["step"]), done=raw["done"] == "True")
                assert all(math.isfinite(v) for v in row.values())
                trial = int(raw["trial"])
                groups.setdefault(trial, []).append(row)
        assert len(groups) == 30
        trials = []
        for index, saved in enumerate(job["metrics"]["trials"]):
            rows = groups[index]
            assert [r["step"] for r in rows] == list(range(len(rows)))
            segments = []
            for segment, (vx_target, yaw_target) in enumerate(targets):
                start = ends[segment-1] if segment else 0
                lo, hi = round(start * 50), round(ends[segment] * 50)
                block = rows[lo:hi]
                yaw_target *= saved["mirror"]
                for row in block:
                    assert abs(row["time"] - (row["step"] + 1) / 50) < 1e-8
                    assert abs(row["cmd_vx"] - vx_target) < 1e-8
                    assert abs(row["cmd_yaw"] - yaw_target) < 1e-8
                stable = [r for r in block if r["time"] - start > 1 and not r["done"]]
                maes = [sum(abs(r[key] - target) for r in stable) / len(stable) if stable else None
                        for key, target in [("vx", vx_target), ("vy", 0), ("wz", yaw_target)]]
                streak, settled = 0, None
                stop = vx_target == yaw_target == 0
                turn = vx_target == 0 and yaw_target != 0
                for r in block:
                    xy = math.hypot(r["vx"], r["vy"])
                    good = (xy < .05 and abs(r["wz"]) < .1) if stop else (
                        abs(r["vx"]-vx_target) <= .1 and abs(r["vy"]) <= .1 and abs(r["wz"]-yaw_target) <= .15 and (not turn or xy <= .1))
                    streak = streak + 1 if good and not r["done"] else 0
                    if streak >= 25 and settled is None:
                        settled = r["time"] - start
                xy_mean = sum(math.hypot(r["vx"], r["vy"]) for r in stable) / len(stable) if stable else None
                ok = bool(len(block) == hi-lo and stable and not any(r["done"] for r in block)
                          and all(x <= bound for x, bound in zip(maes, [.1, .1, .15]))
                          and (not turn or xy_mean <= .1) and settled is not None and settled <= 2)
                assert ok == saved["segments"][segment]["success"]
                segments.append({"index": segment, "success": ok, "mae_vx_vy_yaw": maes, "settled_at_s": settled})
            success = len(rows) == 1500 and all(s["success"] for s in segments) and not any(r["done"] for r in rows)
            assert success == saved["success"]
            trials.append({"index": index, "seed": saved["seed"], "yaw_degrees": saved["yaw_degrees"],
                           "mirror": saved["mirror"], "success": success,
                           "failed_segments": [s for s in segments if not s["success"]]})
        initial_states.append(json.loads((run / "initial-states.json").read_text()))
        result["evaluations"].append({"run": run.name, "successes": sum(t["success"] for t in trials),
                                      "trials": 30, "failures": [t for t in trials if not t["success"]]})
    assert initial_states[0] == initial_states[1], "Baseline conditions differ"
    import av
    from PIL import Image, ImageDraw
    recording = Path(state["recording"])
    video = recording / "commands.mp4"
    montage = Image.new("RGB", (960, 720))
    selected = [25, 150, 250, 400, 500, 700]
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        count = 0
        for frame in container.decode(video=0):
            if count in selected:
                tile = frame.to_image().resize((480, 240))
                ImageDraw.Draw(tile).text((5, 220), f"{count/fps:.1f}s", fill="yellow")
                i = selected.index(count)
                montage.paste(tile, ((i % 2)*480, (i//2)*240))
            count += 1
    assert count == 750 and fps == 25.0, (count, fps)
    record_metrics = json.loads((recording / "metrics.json").read_text())
    assert record_metrics["trials"][0]["steps"] == 1500
    archive = session / Path(state["archive"]).name
    assert digest(archive) == digest(Path(state["archive"])) == state["archive_sha256"]
    assert digest(Path(state["checkpoint"])) == digest(Path(state["checkpoint_drive"]))
    assert digest(video) == digest(destination / "runs" / recording.name / video.name)
    result.update(video={"frames": count, "fps": fps, "seconds": count/fps, "sha256": digest(video)},
                  archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                  checkpoint_sha256=digest(Path(state["checkpoint"])),
                  recording_success=record_metrics["success_rate"], status="passed")
    audit = session / "final-audit"
    audit.mkdir(exist_ok=False)
    (audit / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(__file__, audit / "audit_commands.py")
    montage.save(audit / "preview.jpg")
    shutil.copytree(audit, destination / "final-audit", dirs_exist_ok=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
