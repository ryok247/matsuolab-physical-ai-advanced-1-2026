"""Audit recorded trajectories, decode videos, and archive a completed Colab run."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import av
from PIL import Image, ImageDraw


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    evaluation, training, backup = map(Path, sys.argv[1:4])
    metrics_path = evaluation / "metrics.json"
    config = json.loads((evaluation / "config.json").read_text())
    initial_x = float(config["cfgs"][0]["base_init_pos"][0])
    original = evaluation / "metrics-before-origin-fix.json"
    if not original.exists():
        shutil.copy2(metrics_path, original)
    metrics = json.loads(original.read_text())
    with (evaluation / "rollout.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for trial in metrics["trials"]:
        group = [r for r in rows if r["control"] == trial["control"] and int(r["seed"]) == trial["seed"]]
        assert group and abs(float(group[0]["x"]) - initial_x) < 0.2
        early = any(r["done"].lower() == "true" for r in group)
        errors = [abs(float(r["vx"]) - 0.5) for r in group if int(r["step"]) >= 50 and r["done"].lower() != "true"]
        mae = sum(errors) / len(errors) if errors else None
        displacement = float(group[-1]["x"]) - initial_x if not early else None
        trial.update(steps=len(group), terminated_early=early, vx_mae_m_s=mae,
                     x_displacement_m=displacement,
                     success=len(group) == 500 and not early and mae is not None and mae < 0.2 and displacement > 1.0)
    metrics["ppo_success_rate"] = sum(t["success"] for t in metrics["trials"] if t["control"] == "ppo") / 3
    metrics["measurement_correction"] = {
        "reason": "reset uses skip_forward=True; pre-step get_pos may return the preceding trial's cached pose",
        "method": "recompute from logged world x at final step minus configured reset x; no physics/training rerun",
        "original_preserved": original.name,
    }
    video_checks = []
    canvas = Image.new("RGB", (960, 520), "white")
    for video in sorted(evaluation.glob("*.mp4")):
        count = 0
        samples = {}
        with av.open(str(video)) as container:
            fps = float(container.streams.video[0].average_rate)
            for count, frame in enumerate(container.decode(video=0), start=1):
                if count in [26, 126, 226]:
                    samples[count] = frame.to_image()
        assert count == 250 and fps == 25, (video.name, count, fps)
        video_checks.append({"file": video.name, "frames": count, "fps": fps,
                             "bytes": video.stat().st_size, "sha256": sha256(video)})
        if video.name in ["standing_pd-1000.mp4", "ppo-1000.mp4"]:
            y = 0 if video.name.startswith("standing") else 260
            for column, (number, image) in enumerate(samples.items()):
                canvas.paste(image.resize((320, 240)), (column * 320, y + 20))
                ImageDraw.Draw(canvas).text((column * 320 + 5, y + 4),
                                           f"{video.stem} t={(number - 1) / 25:.0f}s", fill="black")
    canvas.save(evaluation / "walking-preview.jpg")
    metrics["videos"] = video_checks
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    shutil.copy2(Path(__file__), evaluation / "finalize.py")
    shutil.copytree(evaluation, backup / evaluation.name, dirs_exist_ok=True)
    checkpoint = training / "checkpoint.pt"
    assert sha256(checkpoint) == sha256(backup / training.name / "checkpoint.pt")
    archive = evaluation.parent.parent / "genesis-go2-results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for directory in [training, evaluation]:
            for path in directory.iterdir():
                if path.is_file():
                    z.write(path, f"{directory.name}/{path.name}")
    saved_archive = backup.parent / archive.name
    shutil.copy2(archive, saved_archive)
    assert sha256(archive) == sha256(saved_archive)
    print(json.dumps({"metrics": metrics, "archive": str(saved_archive),
                      "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive),
                      "checkpoint_drive_sha256": sha256(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
