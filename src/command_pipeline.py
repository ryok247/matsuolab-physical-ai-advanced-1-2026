"""Bounded staged training pipeline; inspect status.json without duplicating jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from command_worker import backup, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=301)
    args = parser.parse_args()
    args.session.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    sequence = 0
    history = []
    chosen = None
    def status(phase, **extra):
        value = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
                 "history": history, **extra}
        dump(args.session / "status.json", value)
        args.backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.session / "status.json", args.backup / "status.json")
        print(json.dumps({k: v for k, v in value.items() if k != "history"}), flush=True)

    def job(mode, *extra, timeout=1800):
        nonlocal sequence
        sequence += 1
        run = args.session / "runs" / f"{sequence:02d}-{mode}-{uuid.uuid4().hex[:6]}"
        log = args.session / "logs" / (run.name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(source / "command_worker.py"), mode,
                   "--run", str(run), "--backup", str(args.backup / "runs"), *map(str, extra)]
        status("running", mode=mode, run=str(run), command=command)
        code = -1
        try:
            with log.open("w") as output:
                process = subprocess.Popen(command, cwd=source, stdout=output, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, MPLBACKEND="Agg", PYOPENGL_PLATFORM="egl",
                                                    OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
                try:
                    code = process.wait(timeout=timeout)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = process.returncode
                    raise
        finally:
            run.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, run / "run.log")
            dump(run / "command.json", command)
            (run / "exit-code.txt").write_text(str(code))
            backup(run, args.backup / "runs")
        if code:
            raise RuntimeError(f"{mode} failed exit={code}: {log}")
        metrics = json.loads((run / "metrics.json").read_text())
        history.append({"mode": mode, "run": str(run), "metrics": metrics})
        status("job_completed", mode=mode, run=str(run), success_rate=metrics.get("success_rate"))
        return run, metrics

    try:
        smoke, _ = job("smoke", "--stage", "full")
        resumed, resumed_metrics = job("smoke", "--resume", smoke / "checkpoint.pt", "--stage", "full")
        assert resumed_metrics["optimizer_restored"] and resumed_metrics["start_iteration"] == 2
        status("resume_verified", run=str(resumed))
        previous = None
        for batch in range(3):
            load = ["--resume", previous / "checkpoint.pt"] if previous else []
            previous, _ = job("train", "--stage", "speed", "--num-envs", args.num_envs,
                              "--iterations", args.iterations, *load)
            _, metrics = job("eval", "--stage", "speed", "--checkpoint", previous / "checkpoint.pt")
            if metrics["success_rate"] == 1.0:
                break
        # Keep the capability scope intact even if stage A needs more work: report its
        # failures and give the full task a bounded attempt; never relabel it success.
        full_previous = None
        for batch in range(3):
            load = ["--resume", full_previous / "checkpoint.pt"] if full_previous else ["--initialize-from", previous / "checkpoint.pt"]
            full_previous, _ = job("train", "--stage", "full", "--num-envs", args.num_envs,
                                   "--iterations", args.iterations, *load)
            _, metrics = job("eval", "--stage", "full", "--checkpoint", full_previous / "checkpoint.pt")
            if metrics["success_rate"] == 1.0:
                break
        if metrics["success_rate"] < 1.0:
            # Plan's first reward adjustment: increase only yaw tracking weight.
            status("yaw_reward_adjustment", old=0.2, new=0.5)
            for batch in range(3):
                load = (["--initialize-from", full_previous / "checkpoint.pt", "--yaw-reward", 0.5]
                        if batch == 0 else ["--resume", full_previous / "checkpoint.pt"])
                full_previous, _ = job("train", "--stage", "full", "--num-envs", args.num_envs,
                                       "--iterations", args.iterations, *load)
                _, metrics = job("eval", "--stage", "full", "--checkpoint", full_previous / "checkpoint.pt")
                if metrics["success_rate"] == 1.0:
                    break
        chosen = full_previous / "checkpoint.pt"
        formal_run, formal = job("eval", "--stage", "full", "--formal", "--checkpoint", chosen)
        baseline_run, baseline = job("eval", "--stage", "full", "--formal", "--checkpoint", args.baseline)
        recording, _ = job("record", "--stage", "full", "--checkpoint", chosen, timeout=1800)
        summary = {"checkpoint": str(chosen), "checkpoint_drive": str(args.backup / "runs" / chosen.parent.name / chosen.name),
                   "formal_run": str(formal_run), "formal_success_rate": formal["success_rate"],
                   "baseline_success_rate": baseline["success_rate"], "recording": str(recording),
                   "target_met": formal["success_rate"] >= 0.9,
                   "scope": "speed, stop, yaw commands on flat ground; no disturbance or navigation"}
        dump(args.session / "summary.json", summary)
        shutil.copy2(args.session / "summary.json", args.backup / "summary.json")
        archive = args.session / "go2-command-results.zip"
        # Export final results and recipe, not every training checkpoint in every batch.
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(args.session / "summary.json", "summary.json")
            for directory in [chosen.parent, formal_run, baseline_run, recording]:
                for path in directory.iterdir():
                    if path.is_file() and (not path.name.startswith("model_") or path.name == "checkpoint.pt"):
                        z.write(path, str(path.relative_to(args.session)))
        shutil.copy2(archive, args.backup / archive.name)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != hashlib.sha256((args.backup / archive.name).read_bytes()).hexdigest():
            raise RuntimeError("Drive archive hash mismatch")
        status("complete", **summary, archive=str(args.backup / archive.name), archive_sha256=digest)
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}", checkpoint=str(chosen) if chosen else None)
        raise


if __name__ == "__main__":
    main()
