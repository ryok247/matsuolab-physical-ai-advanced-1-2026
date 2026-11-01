"""Physical lateral push wrapper; no additional simulation steps or observations."""
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz
from command_env import CommandEnv


class PushEnv(CommandEnv):
    def __init__(self, num_envs, cfgs, *, training_range=None, evaluation=False, record=False):
        self.push_ready = False
        self.training_range = training_range
        super().__init__(num_envs, cfgs, stage='full', evaluation=evaluation, record=record)
        self.trunk = self.robot.get_link('base')
        self.mass = float(self.robot.get_mass().flatten()[0])
        assert 1 < self.mass < 100
        self.push_clock = torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.next_push = torch.zeros_like(self.push_clock)
        self.remaining = torch.zeros_like(self.push_clock)
        self.enabled = torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.force = torch.zeros((num_envs,3),device=self.device)
        self.applied_force = self.force.clone()
        self.pulses_applied = 0
        self.push_ready = True
        self._reset_push(None)

    def _reset_push(self, mask):
        mask = torch.ones(self.num_envs,device=self.device,dtype=torch.bool) if mask is None else mask
        self.push_clock[mask] = 0
        self.remaining[mask] = 0
        self.force[mask] = 0
        self.enabled[mask] = torch.rand(int(mask.sum()),device=self.device)<.5
        self.next_push[mask] = torch.randint(150,301,(int(mask.sum()),),device=self.device)

    def _reset_idx(self, envs_idx=None):
        super()._reset_idx(envs_idx)
        if self.push_ready:
            self._reset_push(envs_idx)

    def start_push(self, mask, alpha, sign):
        yaw = torch.deg2rad(quat_to_xyz(self.robot.get_quat(),rpy=True,degrees=True)[:,2])
        magnitude = torch.as_tensor(alpha,device=self.device)*self.mass*9.81*torch.as_tensor(sign,device=self.device)
        vector = torch.stack((-torch.sin(yaw)*magnitude,torch.cos(yaw)*magnitude,torch.zeros_like(yaw)),dim=1)
        self.force[mask] = vector[mask]
        self.remaining[mask] = 10
        self.pulses_applied += int(mask.sum())

    def step(self, actions):
        if self.training_range is not None:
            onset = self.enabled & (self.push_clock == self.next_push)
            low, high = self.training_range
            alpha = low+(high-low)*torch.rand(self.num_envs,device=self.device)
            sign = torch.where(torch.rand(self.num_envs,device=self.device)<.5,-1.,1.)
            self.start_push(onset,alpha,sign)
            self.next_push[onset] += torch.randint(200,351,(int(onset.sum()),),device=self.device)
        self.applied_force = torch.where((self.remaining>0)[:,None],self.force,0.).clone()
        if bool((self.remaining>0).any()):
            self.scene.rigid_solver.apply_links_external_wrench(
                force=self.applied_force[:,None,:], links_idx=[self.trunk.idx],
                ref=gs.link_ref_frame.link_COM,local=False)
        # Advance independent clock before super: reset hook must leave reset envs at zero.
        self.push_clock += 1
        self.remaining.clamp_(min=0)
        self.remaining.sub_(1).clamp_(min=0)
        return super().step(actions)
