"""Terrain policy wrapper. GPU parity and sampler probes required before training."""
import copy
import numpy as np
import torch
from terrain_command_base import CommandEnv

class TerrainEnv(CommandEnv):
    def __init__(self,num_envs,cfgs,terrain_file,*,evaluation=False,record=False):
        cfgs=copy.deepcopy(cfgs);cfgs[0]['_terrain_file']=str(terrain_file)
        # Allocated before super because reward functions may be bound during initialization.
        self.height_cpu=np.load(terrain_file)['height'].copy()
        self.height_tensor=None
        super().__init__(num_envs,cfgs,stage='full',evaluation=evaluation,record=record)
        self.height_tensor=torch.as_tensor(self.height_cpu,device=self.device,dtype=torch.float32)*.001
    def ground_height(self,xy):
        # Installed Genesis 1.4.5 exposes the actual piecewise-planar surface.
        # A per-environment point dimension avoids querying every point in every env.
        return self.terrain.get_terrain_height(xy[:,None,:]).reshape(-1)
    def _reward_base_height(self):
        return torch.square(self.base_pos[:,2]-self.ground_height(self.base_pos[:,:2])-self.reward_cfg['base_height_target'])
