"""Record zero-force trajectories to compare the wrapper against CommandEnv."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,copy,json
from pathlib import Path
import torch
import genesis as gs
from command_env import CommandEnv
from push_env import PushEnv
from rsl_rl.runners import OnPolicyRunner
p=argparse.ArgumentParser();p.add_argument('kind',choices=['command','push']);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
saved=json.loads((a.checkpoint.parent/'config.json').read_text())
gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=81,performance_mode=True)
env=CommandEnv(2,copy.deepcopy(saved['cfgs']),evaluation=True) if a.kind=='command' else PushEnv(2,copy.deepcopy(saved['cfgs']),evaluation=True)
runner=OnPolicyRunner(env,copy.deepcopy(saved['train_cfg']),None,device=gs.device)
runner.load(str(a.checkpoint),load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False),map_location=str(gs.device))
policy=runner.get_inference_policy(device=gs.device)
obs=env.reset_with_yaws([0,90],[91,92]);rows=[]
with torch.no_grad():
    for step in range(100):
        obs=env.set_commands([[.5,0,0]]*2)
        obs,_,_,_=env.step(policy(obs))
        rows.append({k:v.cpu().tolist() for k,v in env.terminal.items()})
if a.kind=='push':
    assert env.pulses_applied==0
    env.start_push(torch.tensor([True,False],device=gs.device),.1,1)
    assert float(env.force[0,1])>0 and float(env.force[1].abs().sum())==0
    env.reset()
    assert int(env.remaining.sum())==0 and float(env.force.abs().sum())==0
    env.reset_with_yaws([0,90],[91,92])
    env.start_push(torch.ones(2,dtype=torch.bool,device=gs.device),.1,1)
    assert abs(float(env.force[0,0]))<1e-4 and float(env.force[0,1])>0
    assert float(env.force[1,0])<0 and abs(float(env.force[1,1]))<1e-4
    print('mass',env.mass,'trunk',env.trunk.name,'idx',env.trunk.idx,'world force',env.force.cpu().tolist())
a.output.write_text(json.dumps(rows,allow_nan=False))
print('ZERO FORCE TRAJECTORY SAVED',a.output)
