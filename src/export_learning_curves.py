"""Export existing TensorBoard scalars and plots, without training or simulation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile


def main(session):
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    status = json.loads((session / "status.json").read_text())
    assert status["phase"] == "complete"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = session / (stamp + "-learning-curves")
    output.mkdir(exist_ok=False)
    rows, runs, raw_files = [], [], []
    for job in status["history"]:
        if job["mode"] != "train":
            continue
        run = Path(job["run"])
        config = json.loads((run / "config.json").read_text())
        events = sorted(run.glob("**/events.out.tfevents.*"))
        assert events, f"Missing TensorBoard events: {run}"
        label = f'{run.name} ({config["command_stage"]}, yaw={config["cfgs"][2]["reward_scales"]["tracking_ang_vel"]})'
        entry = {"run": run.name, "label": label, "events": [], "tags": []}
        for event in events:
            accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = accumulator.Tags()["scalars"]
            entry["tags"] = sorted(set(entry["tags"]) | set(tags))
            target = output / "events" / run.name / event.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(event, target)
            raw_files.append(target)
            entry["events"].append(str(target.relative_to(output)))
            for tag in tags:
                rows.extend({"run": run.name, "tag": tag, "step": e.step, "wall_time": e.wall_time, "value": e.value}
                            for e in accumulator.Scalars(tag))
        runs.append(entry)
        log = run / "run.log"
        if log.exists():
            target = output / "logs" / (run.name + ".log")
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(log, target)
    assert rows
    with (output / "scalars.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)
    tags = sorted({r["tag"] for r in rows})
    # Plot actual recorded tag names; do not infer missing metrics from rewards.
    wanted = []
    for pattern in ["mean_reward", "mean_episode_length", "action_std", "value_function", "value_loss"]:
        found = next((t for t in tags if pattern in t.lower() and t not in wanted), None)
        if found:
            wanted.append(found)
    if not wanted:
        wanted = tags[:4]
    fig, axes = plt.subplots(len(wanted), 1, figsize=(12, 3*len(wanted)), squeeze=False)
    for ax, tag in zip(axes[:, 0], wanted):
        for run in runs:
            values = sorted([r for r in rows if r["run"] == run["run"] and r["tag"] == tag], key=lambda r:r["step"])
            if values:
                ax.plot([v["step"] for v in values], [v["value"] for v in values], label=run["label"], linewidth=1)
        ax.set_title(tag)
        ax.set_xlabel("PPO iteration recorded in each run (optimizer resets keep separate runs)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=140)
    plt.close(fig)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "session": str(session),
                "scope": "existing scalar data; no learning or physics rerun", "scalar_rows": len(rows),
                "runs": runs, "plotted_tags": wanted,
                "files": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.rglob("*") if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in output.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    destination = Path(status["archive"]).parent
    shutil.copy2(archive, destination / archive.name)
    print(json.dumps({"directory": str(output), "archive": str(archive), "drive": str(destination / archive.name),
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size,
                      "scalar_rows": len(rows), "training_runs": len(runs), "tags": wanted}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
