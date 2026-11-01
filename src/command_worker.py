"""CUDA entry point for command-conditioned Go2 training, evaluation and teleoperation."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from commands import CONFIG, cases, command_at, score_trial


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def backup(run, destination):
    if destination:
        target = destination / run.name
        target.mkdir(parents=True, exist_ok=True)
        for path in run.iterdir():
            if path.is_file() and not path.name.endswith(".partial"):
                temp = target / (path.name + ".partial")
                shutil.copy2(path, temp)
                temp.replace(target / path.name)


def evaluate(env, runner, args, evaluation_stage):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    trial_cases = cases(args.formal, evaluation_stage, args.mode in ["record", "teleop"])
    obs = env.reset_with_yaws([c["yaw_degrees"] for c in trial_cases], [c["seed"] for c in trial_cases])
    dump(args.run / "initial-states.json", [dict(c, qpos=q) for c, q in zip(trial_cases, env.evaluation_initial_qpos)])
    policy = runner.get_inference_policy(device=env.device)
    rows = [[] for _ in trial_cases]
    active = [True] * len(trial_cases)
    video_writer, video_stream = None, None
    if args.mode in ["record", "teleop"]:
        import av
        video_writer = av.open(str(args.run / "commands.mp4"), "w")
        video_stream = video_writer.add_stream("libx264", rate=25)
        video_stream.width, video_stream.height, video_stream.pix_fmt = 640, 480, "yuv420p"
    started = time.monotonic()
    try:
        with torch.no_grad():
            for step in range(1500):
                applied, segment_ids = [], []
                for c in trial_cases:
                    segment, cmd = command_at(step, evaluation_stage, c["mirror"])
                    if args.mode == "teleop":
                        requested = json.loads(args.control.read_text())
                        cmd = [max(0, min(0.8, float(requested["vx"]))), 0,
                               max(-0.6, min(0.6, float(requested["yaw"])))]
                        if not all(math.isfinite(v) for v in cmd):
                            raise ValueError("Non-finite control")
                    applied.append(list(cmd))
                    segment_ids.append(segment)
                obs = env.set_commands(applied)
                assert torch.allclose(obs["policy"][:, 6:9], env.commands * env.commands_scale)
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Non-finite policy action")
                obs, reward, _, _ = env.step(actions)
                state = {k: v.detach().cpu().numpy() for k, v in env.terminal.items()}
                if not all(np.isfinite(v).all() for v in state.values()) or not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("Non-finite physics/observation")
                action_values = actions.detach().cpu().tolist()
                reward_terms = {k: v.cpu().tolist() for k, v in env.reward_values.items()}
                for i, c in enumerate(trial_cases):
                    if not active[i]:
                        continue
                    pos = state["base_pos"][i]
                    quat = state["base_quat"][i]
                    vel = state["base_lin_vel"][i]
                    angular = state["base_ang_vel"][i]
                    euler = state["base_euler"][i]
                    done = bool(state["reset_buf"][i])
                    row = {"step": step, "time": (step + 1) * 0.02, "segment": segment_ids[i],
                           "cmd_vx": applied[i][0], "cmd_vy": applied[i][1], "cmd_yaw": applied[i][2],
                           "vx": float(vel[0]), "vy": float(vel[1]), "vz": float(vel[2]),
                           "wz": float(angular[2]), "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                           "roll": float(euler[0]), "pitch": float(euler[1]),
                           "yaw_world": math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2)),
                           "joint_speed_rms": float(np.sqrt(np.mean(state["dof_vel"][i] ** 2))),
                           "reward": float(state["rew_buf"][i]), "done": done,
                           "termination": "solver_error" if state["solver_error"][i] else ("roll_pitch_limit" if done else "none")}
                    row.update({f"action_{j}": a for j, a in enumerate(action_values[i])})
                    row.update({f"reward_{k}": v[i] for k, v in reward_terms.items()})
                    rows[i].append(row)
                    if done:
                        active[i] = False
                if video_writer and step % 2 == 0:
                    row = rows[0][-1]
                    if not row["done"]:
                        pos = state["base_pos"][0]
                        env.camera.set_pose(pos=pos + np.array([1.8, -2.0, 1.3]), lookat=pos)
                        rgb = env.camera.render()[0]
                        picture = Image.fromarray(rgb)
                        draw = ImageDraw.Draw(picture)
                        draw.rectangle((0, 0, 640, 56), fill="black")
                        draw.text((8, 5), f't={row["time"]:.2f}s  command vx={row["cmd_vx"]:.2f}m/s yaw={row["cmd_yaw"]:.2f}rad/s', fill="white")
                        draw.text((8, 24), f'actual vx={row["vx"]:.2f}m/s yaw={row["wz"]:.2f}rad/s  playback 1x', fill="white")
                        for packet in video_stream.encode(av.VideoFrame.from_ndarray(np.array(picture), format="rgb24")):
                            video_writer.mux(packet)
                        if args.mode == "teleop" and step % 10 == 0:
                            picture.save(args.run / "latest.partial.jpg")
                            (args.run / "latest.partial.jpg").replace(args.run / "latest.jpg")
                if not any(active):
                    break
                if args.mode == "teleop":
                    time.sleep(max(0, started + (step+1)*0.02 - time.monotonic()))
    finally:
        if video_writer:
            for packet in video_stream.encode():
                video_writer.mux(packet)
            video_writer.close()
        # Keep partial trajectories even if a renderer/physics failure interrupts the run.
        with (args.run / "rollout.csv").open("w", newline="") as output:
            nonempty = next((r for r in rows if r), None)
            if nonempty:
                writer = csv.DictWriter(output, fieldnames=["trial", "seed", "initial_yaw", "mirror", *nonempty[0]])
                writer.writeheader()
                for i, group in enumerate(rows):
                    for r in group:
                        writer.writerow({"trial": i, "seed": trial_cases[i]["seed"],
                                         "initial_yaw": trial_cases[i]["yaw_degrees"], "mirror": trial_cases[i]["mirror"], **r})
    trials = [{**c, **score_trial(r, evaluation_stage, c["mirror"])} for c, r in zip(trial_cases, rows)]
    if args.mode == "teleop":
        metrics = {"status": "manual_session_completed", "steps": len(rows[0]), "terminated_early": not active[0],
                   "success_rate": None, "scope": "manual commands; not a formal evaluation"}
    else:
        metrics = {"status": "evaluated", "trials": trials,
                   "success_rate": sum(t["success"] for t in trials) / len(trials),
                   "formal": args.formal, "evaluation_stage": evaluation_stage,
                   "scope": "flat ground; initial yaw and mirrored commands varied; no disturbance test"}
        import matplotlib.pyplot as plt
        example = rows[0]
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for axis, measured, commanded, label in [(axes[0], "vx", "cmd_vx", "vx (m/s)"), (axes[1], "wz", "cmd_yaw", "yaw rate (rad/s)")]:
            axis.plot([r["time"] for r in example], [r[commanded] for r in example], label="command")
            axis.plot([r["time"] for r in example], [r[measured] for r in example], label="actual")
            axis.set_ylabel(label)
            axis.legend()
        axes[1].set_xlabel("simulation time (s)")
        fig.tight_layout()
        fig.savefig(args.run / "tracking.png")
        plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "train", "eval", "record", "teleop"])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--stage", choices=["speed", "full"])
    parser.add_argument("--yaw-reward", type=float)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=301)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--control", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--initialize-from", type=Path)
    group.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or (args.mode == "teleop" and args.control is None):
        parser.error("Invalid resource/control arguments")
    if args.mode in ["eval", "record", "teleop"] and not args.checkpoint:
        parser.error("Evaluation/recording requires --checkpoint")
    args.run.mkdir(parents=True, exist_ok=False)
    os.environ["MPLBACKEND"] = "Agg"
    import torch
    import genesis as gs
    from command_env import CommandEnv
    from go2_train import get_cfgs, get_train_cfg
    from rsl_rl.runners import OnPolicyRunner
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    spec = json.loads(Path(__file__).with_name("upstream.lock.json").read_text())
    checkpoint = args.resume or args.initialize_from or args.checkpoint
    saved = None
    if checkpoint:
        saved = json.loads((checkpoint.parent / "config.json").read_text())
        if saved["upstream"] != spec:
            raise ValueError("Upstream pins differ")
    stage = args.stage or (saved.get("command_stage", "speed") if saved else "speed")
    cfgs = get_cfgs()
    train_cfg = get_train_cfg(args.run.name)
    train_cfg["save_interval"] = 25
    if args.resume:
        if stage != saved["command_stage"] or saved["command_config"] != CONFIG:
            raise ValueError("Resume must preserve stage and command configuration; use initialize-from for a new task")
        cfgs, train_cfg = saved["cfgs"], saved["train_cfg"]
        if args.yaw_reward is not None and args.yaw_reward != cfgs[2]["reward_scales"]["tracking_ang_vel"]:
            raise ValueError("Resume cannot change the reward")
    elif saved and args.checkpoint:
        cfgs, train_cfg = saved["cfgs"], saved["train_cfg"]
    elif args.yaw_reward is not None:
        cfgs[2]["reward_scales"]["tracking_ang_vel"] = args.yaw_reward
    if args.mode in ["eval", "record", "teleop"]:
        cfgs = copy.deepcopy(cfgs)
        cfgs[0]["episode_length_s"] = 35.0
    cfgs[0]["resampling_time_s"] = CONFIG["hold_s"]
    config = {"task": "go2-commands-v1", "command_stage": stage, "command_config": CONFIG,
              "cfgs": cfgs, "train_cfg": train_cfg, "upstream": spec, "seed": args.seed,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    dump(args.run / "config.json", config)
    sources = {}
    for name in ["command_worker.py", "commands.py", "command_env.py", "command-config.json", "go2_env.py", "go2_record_env.py", "go2_train.py", "upstream.lock.json"]:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, args.run / name)
        sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(args.run / "source-sha256.json", sources)
    dump(args.run / "dependencies.json", {p: importlib.metadata.version(p) for p in ["genesis-world", "torch", "rsl-rl-lib", "quadrants"]})
    (args.run / "requirements.freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (args.run / "gpu.txt").write_text(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True))
    gs.init(backend=gs.cuda, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    is_eval = args.mode in ["eval", "record", "teleop"]
    n = len(cases(args.formal, stage, args.mode in ["record", "teleop"])) if is_eval else (8 if args.mode == "smoke" else args.num_envs)
    env = CommandEnv(n, cfgs, stage=stage, evaluation=is_eval, record=args.mode in ["record", "teleop"])
    class SavedRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            super().save(path, infos)
            backup(args.run, args.backup)
    runner = SavedRunner(env, copy.deepcopy(train_cfg), str(args.run), device=gs.device)
    if checkpoint:
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False} if args.initialize_from else None
        runner.load(str(checkpoint), load_cfg=load_cfg, map_location=str(gs.device))
        if args.resume:
            runner.current_learning_iteration += 1
    before_iteration = runner.current_learning_iteration
    started = time.monotonic()
    if is_eval:
        metrics = evaluate(env, runner, args, stage)
    else:
        assert env.get_observations()["policy"].shape == (n, 45)
        # Verify command injection and exact zero stop observations before training.
        env.set_commands(torch.zeros(n, 3, device=gs.device))
        assert torch.equal(env.get_observations()["policy"][:, 6:9], torch.zeros(n, 3, device=gs.device))
        env.reset()
        before = [p.detach().clone() for p in runner.alg.get_policy().parameters()]
        iterations = 2 if args.mode == "smoke" else args.iterations
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        delta = sum((a-b).abs().sum().item() for a, b in zip(runner.alg.get_policy().parameters(), before))
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError("No finite PPO update")
        runner.save(str(args.run / "checkpoint.pt"))
        metrics = {"status": "ppo_updates_completed", "iterations_this_run": iterations,
                   "transitions_this_run": n * train_cfg["num_steps_per_env"] * iterations,
                   "start_iteration": before_iteration, "last_completed_iteration": runner.current_learning_iteration,
                   "optimizer_restored": bool(args.resume), "parameter_l1_change": delta}
    metrics["elapsed_s"] = time.monotonic() - started
    dump(args.run / "metrics.json", metrics)
    backup(args.run, args.backup)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Parent orchestrator also persists an exit code when failure precedes imports.
        print(f"COMMAND WORKER FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise
