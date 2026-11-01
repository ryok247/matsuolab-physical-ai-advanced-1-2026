"""Extend the SHA-verified official environment without editing its source."""
import copy
import random
import torch
import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat, quat_to_xyz
from go2_record_env import Go2Env
from commands import CONFIG


class CommandEnv(Go2Env):
    def __init__(self, num_envs, cfgs, stage="full", evaluation=False, record=False):
        self.stage = copy.deepcopy(CONFIG["stages"][stage])
        self.evaluation = evaluation
        self._inside_step = False
        self.terminal = None
        super().__init__(num_envs, *copy.deepcopy(cfgs), record=record)
        self.reward_values = {}
        if evaluation:
            # Capture the actual old-command rewards, before upstream resampling.
            for name, function in list(self.reward_functions.items()):
                def wrapped(fn=function, key=name):
                    value = fn()
                    self.reward_values[key] = (value * self.reward_scales[key]).detach().clone()
                    return value
                self.reward_functions[name] = wrapped

    def _resample_commands(self, envs_idx):
        if self.evaluation:
            # Evaluation commands are injected explicitly, before observation/policy.
            return
        n = self.num_envs
        modes = torch.multinomial(torch.tensor(self.stage["weights"], device=self.device), n, replacement=True)
        vx = 0.1 + torch.rand(n, device=self.device) * (self.stage["vx_max"] - 0.1)
        arc_vx = 0.1 + torch.rand(n, device=self.device) * (self.stage["arc_vx_max"] - 0.1)
        yaw = (0.2 + torch.rand(n, device=self.device) * max(0.0, self.stage["yaw_max"] - 0.2))
        yaw *= torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        values = torch.zeros_like(self.commands)
        values[:, 0] = torch.where(modes == 1, vx, torch.where(modes == 3, arc_vx, 0.0))
        values[:, 2] = torch.where((modes == 2) | (modes == 3), yaw, 0.0)
        if envs_idx is None:
            self.commands.copy_(values)
        else:
            torch.where(envs_idx[:, None], values, self.commands, out=self.commands)

    def set_commands(self, values):
        values = torch.as_tensor(values, device=self.device, dtype=gs.tc_float)
        if values.shape != self.commands.shape or not torch.isfinite(values).all():
            raise ValueError("Invalid commands")
        self.commands.copy_(values)
        self._update_observation()
        return self.get_observations()

    def _reset_idx(self, envs_idx=None):
        if self.evaluation and self._inside_step:
            # Upstream updates these buffers, then calls this hook before resetting.
            self.terminal = {k: getattr(self, k).detach().clone() for k in
                             ["base_pos", "base_quat", "base_euler", "base_lin_vel", "base_ang_vel", "dof_vel", "reset_buf", "rew_buf"]}
            self.terminal["solver_error"] = self.scene.rigid_solver.get_error_envs_mask().clone()
        super()._reset_idx(envs_idx)
        if self.evaluation:
            if envs_idx is None:
                self.commands.zero_()
            else:
                self.commands.masked_fill_(envs_idx[:, None], 0.0)

    def step(self, actions):
        self._inside_step = True
        try:
            return super().step(actions)
        finally:
            self._inside_step = False

    def reset_with_yaws(self, yaw_degrees, seeds):
        self.reset()
        angles = torch.deg2rad(torch.tensor(yaw_degrees, device=self.device, dtype=gs.tc_float))
        qpos = self.init_qpos.repeat(self.num_envs, 1)
        qpos[:, 3] = torch.cos(angles / 2)
        qpos[:, 4:6] = 0
        qpos[:, 6] = torch.sin(angles / 2)
        # Deterministic policy evaluation would otherwise repeat identical rollouts
        # for every seed. Each seed defines a small, recorded initial joint offset.
        offsets = []
        for seed in seeds:
            generator = random.Random(seed)
            width = CONFIG["evaluation_joint_jitter_rad"]
            offsets.append([generator.uniform(-width, width) for _ in range(qpos.shape[1] - 7)])
        qpos[:, 7:] += torch.tensor(offsets, device=self.device, dtype=gs.tc_float)
        self.evaluation_initial_qpos = qpos.detach().cpu().tolist()
        self.robot.set_qpos(qpos, zero_velocity=True, skip_forward=False)
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(self.base_quat, rpy=True, degrees=True)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_quat(self.base_quat))
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel.zero_()
        self._update_observation()
        return self.get_observations()
