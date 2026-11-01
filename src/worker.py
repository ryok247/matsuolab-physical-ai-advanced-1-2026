"""Bounded CUDA training/evaluation; embedded in the standalone Colab notebook."""
import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "train", "eval"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=301)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("num-envs and iterations must be positive")
    # Never clear or reuse a run directory.
    args.run.mkdir(parents=True, exist_ok=False)
    # Colab exports an inline backend unavailable inside the isolated venv.
    # Set this before Genesis imports matplotlib transitively.
    os.environ["MPLBACKEND"] = "Agg"
    import numpy as np
    import torch
    import genesis as gs
    from go2_record_env import Go2Env
    from go2_train import get_cfgs, get_train_cfg
    from rsl_rl.runners import OnPolicyRunner

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Select a Colab GPU runtime.")
    torch.set_num_threads(2)
    gs.init(backend=gs.cuda, precision="32", logging_level="warning",
            seed=args.seed, performance_mode=True)
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
        saved = json.loads((args.checkpoint.parent / "config.json").read_text())
        current = json.loads((Path(__file__).parent / "upstream.lock.json").read_text())
        if saved["upstream"] != current:
            raise RuntimeError("Checkpoint and current upstream pins differ")
        cfgs = saved["cfgs"]
        train_cfg = saved["train_cfg"]
    else:
        cfgs = get_cfgs()
        train_cfg = get_train_cfg(args.run.name)
        train_cfg["save_interval"] = 25
    spec = json.loads((Path(__file__).parent / "upstream.lock.json").read_text())
    config = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cfgs": cfgs, "train_cfg": train_cfg, "upstream": spec,
              "control_dt_s": 0.02, "physics_substeps": 2,
              "success_rule": "500 steps without reset, vx MAE <0.2 m/s, x displacement >1m",
              "evaluation_seeds": [1000, 1001, 1002]}
    write_json(args.run / "config.json", config)
    write_json(args.run / "dependencies.json", {
        p: importlib.metadata.version(p) for p in
        ["genesis-world", "rsl-rl-lib", "torch", "quadrants", "tensorboard"]})
    for name in ["worker.py", "go2_env.py", "go2_record_env.py", "go2_train.py", "upstream.lock.json"]:
        shutil.copy2(Path(__file__).parent / name, args.run / name)

    def backup():
        if args.backup:
            destination = args.backup / args.run.name
            destination.mkdir(parents=True, exist_ok=True)
            for path in args.run.iterdir():
                if path.is_file():
                    temp = destination / (path.name + ".partial")
                    shutil.copy2(path, temp)
                    temp.replace(destination / path.name)

    class SavedRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            super().save(path, infos)
            backup()

    n = 8 if args.mode == "smoke" else (1 if args.mode == "eval" else args.num_envs)
    env = Go2Env(n, *copy.deepcopy(cfgs), record=args.video)
    obs = env.reset()
    assert obs["policy"].shape == (n, 45)
    runner = SavedRunner(env, copy.deepcopy(train_cfg), str(args.run), device=gs.device)
    if args.checkpoint:
        runner.load(str(args.checkpoint), map_location=str(gs.device))
        # Upstream stores the index of the last completed iteration.
        runner.current_learning_iteration += 1
    started = time.monotonic()
    if args.mode in ["smoke", "train"]:
        policy = runner.alg.get_policy()
        before = [p.detach().clone() for p in policy.parameters()]
        iterations = 2 if args.mode == "smoke" else args.iterations
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        delta = sum((a - b).abs().sum().item() for a, b in zip(policy.parameters(), before))
        assert delta > 0 and np.isfinite(delta), "PPO did not update finite parameters"
        runner.save(str(args.run / "checkpoint.pt"))
        metrics = {"status": "ppo_updates_completed", "iterations_this_run": iterations,
                   "transitions_this_run": n * train_cfg["num_steps_per_env"] * iterations,
                   "parameter_l1_change": delta, "walking_success": "not_evaluated"}
    else:
        if not args.checkpoint:
            raise ValueError("eval requires --checkpoint")
        policy = runner.get_inference_policy(device=gs.device)
        rows, trials = [], []
        # Identical initial pose and command; seed changes alone do not establish generalization.
        for control in ["standing_pd", "ppo"]:
            for trial_seed in [1000, 1001, 1002]:
                random.seed(trial_seed)
                np.random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                obs = env.reset()
                # Upstream reset uses skip_forward=True. get_pos() can still expose
                # the previous trial's cached pose until the first physics step.
                origin = np.asarray(env.env_cfg["base_init_pos"], dtype=float).copy()
                errors = []
                fell = False
                writer = None
                # Streaming avoids keeping 250 RGB frames in RAM.
                if args.video:
                    import av
                    video = args.run / f"{control}-{trial_seed}.mp4"
                    writer = av.open(str(video), "w")
                    stream = writer.add_stream("libx264", rate=25)
                    stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
                try:
                    # Environment buffers survive across trials; inference tensors cannot
                    # be mutated by the next reset outside InferenceMode.
                    with torch.no_grad():
                        for step in range(500):
                            actions = policy(obs) if control == "ppo" else torch.zeros((1, 12), device=gs.device)
                            obs, rewards, dones, _ = env.step(actions)
                            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(rewards).all():
                                raise RuntimeError("Non-finite rollout")
                            done = bool(dones[0].item())
                            velocity = env.base_lin_vel[0].detach().cpu().tolist()
                            pos = env.robot.get_pos()[0].detach().cpu().numpy()
                            rows.append([control, trial_seed, step, (step + 1) * 0.02,
                                         *velocity, *pos.tolist(), float(rewards[0]), done])
                            # Exclude initial 1 second of settling; never count a reset state as tracking.
                            if step >= 50 and not done:
                                errors.append(abs(velocity[0] - 0.5))
                            if writer and step % 2 == 0:
                                env.camera.set_pose(pos=pos + np.array([1.7, -2.0, 1.2]),
                                                    lookat=pos + np.array([0, 0, -0.1]))
                                rgb = env.camera.render()[0]
                                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                                    writer.mux(packet)
                            if done:
                                fell = True
                                break
                finally:
                    if writer:
                        for packet in stream.encode():
                            writer.mux(packet)
                        writer.close()
                mae = float(np.mean(errors)) if errors else None
                displacement = float(pos[0] - origin[0]) if not fell else None
                trials.append({"control": control, "seed": trial_seed, "steps": step + 1,
                               "terminated_early": fell, "vx_mae_m_s": mae,
                               "x_displacement_m": displacement,
                               "success": not fell and mae is not None and mae < 0.2 and displacement > 1.0})
        with (args.run / "rollout.csv").open("w") as output:
            csvwriter = csv.writer(output)
            csvwriter.writerow(["control", "seed", "step", "time_s", "vx", "vy", "vz",
                                "x", "y", "z", "reward", "done"])
            csvwriter.writerows(rows)
        metrics = {"status": "evaluated", "trials": trials,
                   "ppo_success_rate": sum(t["success"] for t in trials if t["control"] == "ppo") / 3,
                   "scope": "fixed initial pose, fixed forward command; no turning or disturbance test"}
    metrics["elapsed_s"] = time.monotonic() - started
    write_json(args.run / "metrics.json", metrics)
    backup()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
