"""Save policy trajectories for plane/zero terrain parity and weak-terrain contact."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
import genesis as gs
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',choices=['plane','zero','weak'],required=True);a=p.parse_args()
a.run.mkdir(parents=True,exist_ok=False)
from rsl_rl.runners import OnPolicyRunner
saved=json.loads((a.checkpoint.parent/'config.json').read_text());cfgs=copy.deepcopy(saved['cfgs'])
gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=91,performance_mode=True)
if a.kind=='plane':
    from command_env import CommandEnv
    env=CommandEnv(2,cfgs,evaluation=True)
else:
    from terrain_maps import generate
    from terrain_env import TerrainEnv
    height,meta=generate(100,.01 if a.kind=='weak' else 0)
    np.savez(a.run/'terrain.npz',height=height)
    (a.run/'terrain.json').write_text(json.dumps(meta,indent=2))
    env=TerrainEnv(2,cfgs,a.run/'terrain.npz',evaluation=True)
runner=OnPolicyRunner(env,copy.deepcopy(saved['train_cfg']),None,device=gs.device)
runner.load(str(a.checkpoint),load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False),map_location=str(gs.device))
policy=runner.get_inference_policy(device=gs.device)
env.reset_with_yaws([0,90],[3000,3001]);rows=[]
with torch.no_grad():
    for step in range(400 if a.kind=='weak' else 100):
        obs=env.set_commands([[.5,0,0]]*2);action=policy(obs);env.step(action)
        state=torch.cat([env.terminal[k].reshape(2,-1) for k in ['base_pos','base_quat','base_lin_vel','base_ang_vel','dof_vel']],dim=1)
        assert torch.isfinite(state).all() and torch.isfinite(action).all()
        rows.append(state.cpu().numpy());
        if bool(env.terminal['reset_buf'].any()):raise RuntimeError(f'early termination at {step}')
np.save(a.run/'trajectory.npy',np.stack(rows))
(a.run/'metrics.json').write_text(json.dumps(dict(status='trajectory_completed',steps=len(rows),kind=a.kind),indent=2))
print('Trajectory completed',a.kind,len(rows))
