"""Check the actual CUDA command sampler without building a physics scene."""
import os
os.environ["MPLBACKEND"] = "Agg"
import torch
from command_env import CommandEnv
from commands import CONFIG

torch.manual_seed(13)
env = object.__new__(CommandEnv)
env.num_envs = 10000
env.device = torch.device("cuda")
env.evaluation = False
env.stage = CONFIG["stages"]["full"]
env.commands = torch.zeros(env.num_envs, 3, device=env.device)
env._resample_commands(None)
c = env.commands
assert torch.all(c[:, 1] == 0)
assert torch.all((c[:, 0] >= 0) & (c[:, 0] <= 0.8))
assert torch.all(c[:, 2].abs() <= 0.6)
stop = (c == 0).all(dim=1)
turn = (c[:, 0] == 0) & (c[:, 2] != 0)
assert 0.17 < stop.float().mean().item() < 0.23
assert 0.17 < turn.float().mean().item() < 0.23
assert (c[:, 2] > 0).any() and (c[:, 2] < 0).any()
mask = torch.arange(env.num_envs, device=env.device) % 2 == 0
before = c.clone()
env._resample_commands(mask)
assert torch.equal(env.commands[~mask], before[~mask])
env.stage = CONFIG["stages"]["speed"]
env._resample_commands(None)
assert torch.all(env.commands[:, 2] == 0)
assert torch.all(env.commands[:, 0] <= 0.6)
print("CUDA sampler: exact stops, both yaw signs, bounds and selective reset PASS")
