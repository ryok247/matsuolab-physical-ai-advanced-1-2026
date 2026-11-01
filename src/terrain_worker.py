"""Isolated bounded terrain training/evaluation; calibration gate required."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,copy,csv,hashlib,json,shutil,time,importlib.metadata,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import genesis as gs
from terrain_maps import generate
from terrain_env import TerrainEnv
from rsl_rl.runners import OnPolicyRunner
p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train','eval','regression']);p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--map-seed',type=int,required=True);p.add_argument('--amplitude',type=float,required=True);p.add_argument('--num-envs',type=int,default=256);p.add_argument('--iterations',type=int,default=301);p.add_argument('--resume',action='store_true');p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--backup',type=Path)
p.add_argument('--initial-count',type=int,choices=[2,4],default=4)
# Parse again after adding the optional preservation destination.
a=p.parse_args()
assert json.loads(a.calibration.read_text())['status']=='passed','calibration gate'
assert 1<=a.num_envs<=1024 and 1<=a.iterations<=301 and a.amplitude in [0,.01,.02,.04]
a.run.mkdir(parents=True,exist_ok=False)
saved=json.loads((a.checkpoint.parent/'config.json').read_text());cfgs=copy.deepcopy(saved['cfgs']);tcfg=copy.deepcopy(saved['train_cfg'])
assert cfgs[2]['reward_scales']['tracking_ang_vel']==.5
assert cfgs[0]['termination_if_roll_greater_than']==cfgs[0]['termination_if_pitch_greater_than']==10
cfgs[0]['episode_length_s']=15.;tcfg['save_interval']=25
if a.mode=='regression':cfgs[0]['episode_length_s']=35.
height,meta=generate(a.map_seed,a.amplitude);np.savez(a.run/'terrain.npz',height=height)
if a.resume:assert saved['terrain']==meta,'resume terrain mismatch'
config=dict(task='go2-terrain-v1',cfgs=cfgs,train_cfg=tcfg,terrain=meta,upstream=saved['upstream'],initialization_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
(a.run/'config.json').write_text(json.dumps(config,indent=2))
for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,a.run/f.name)
shutil.copy2(a.calibration,a.run/'calibration.json')
(a.run/'dependencies.json').write_text(json.dumps({k:importlib.metadata.version(k) for k in ['genesis-world','rsl-rl-lib','torch','quadrants']},indent=2))
(a.run/'requirements.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
def backup():
    if a.backup:
        destination=a.backup/a.run.name;destination.mkdir(parents=True,exist_ok=True)
        for f in a.run.rglob('*'):
            if not f.is_file() or f.name.endswith('.partial'):continue
            target=destination/f.relative_to(a.run);target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(target.name+'.partial');shutil.copy2(f,temp);temp.replace(target)
class SavedRunner(OnPolicyRunner):
    def save(self,path,infos=None):
        super().save(path,infos);backup()
torch.set_num_threads(2);gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=81,performance_mode=True)
n=a.initial_count if a.mode=='eval' else (8 if a.mode=='smoke' else a.num_envs)
if a.mode=='regression':
    from command_env import CommandEnv
    from commands import cases
    a.formal=True;n=len(cases(True,'full',False))
    env=CommandEnv(n,cfgs,stage='full',evaluation=True)
else:env=TerrainEnv(n,cfgs,a.run/'terrain.npz',evaluation=a.mode=='eval')
runner=SavedRunner(env,tcfg,str(a.run),device=gs.device)
runner.load(str(a.checkpoint),load_cfg=None if a.resume else dict(actor=True,critic=True,optimizer=False,iteration=False),map_location=str(gs.device))
started=time.monotonic()
if a.mode=='regression':
    from command_worker import evaluate
    metrics=evaluate(env,runner,a,'full')
elif a.mode!='eval':
    if a.resume:runner.current_learning_iteration+=1
    before=[x.detach().clone() for x in runner.alg.get_policy().parameters()]
    count=2 if a.mode=='smoke' else a.iterations
    runner.learn(num_learning_iterations=count,init_at_random_ep_len=True)
    delta=sum(float((x-y).abs().sum()) for x,y in zip(runner.alg.get_policy().parameters(),before));assert np.isfinite(delta) and delta>0
    runner.save(str(a.run/'checkpoint.pt'));metrics=dict(status='ppo_updates_completed',iterations=count,parameter_l1_change=delta,transitions=n*tcfg['num_steps_per_env']*count)
else:
    env.reset_with_yaws([0]*n,list(range(3000,3000+n)));initial=env.evaluation_initial_qpos
    (a.run/'initial-states.json').write_text(json.dumps(initial))
    policy=runner.get_inference_policy(device=gs.device);active=[True]*n;rows=[[] for _ in range(n)]
    with torch.no_grad():
        for step in range(600):
            command=[0,0,0] if step<50 else [.5,0,0]
            obs=env.set_commands([command]*n);actions=policy(obs);env.step(actions);t=env.terminal
            for i in range(n):
                if not active[i]:continue
                pos=t['base_pos'][i].cpu().tolist();vel=t['base_lin_vel'][i].cpu().tolist();angles=t['base_euler'][i].cpu().tolist()
                row=dict(trial=i,step=step,time=(step+1)*.02,cmd_vx=command[0],x=pos[0],y=pos[1],z=pos[2],vx=vel[0],vy=vel[1],wz=float(t['base_ang_vel'][i,2]),roll=angles[0],pitch=angles[1],done=bool(t['reset_buf'][i]))
                row['ground_height']=float(env.ground_height(t['base_pos'][:,:2])[i])
                row['reward']=float(t['rew_buf'][i]);row['solver_error']=bool(t['solver_error'][i])
                row['outside_map']=max(abs(pos[0]),abs(pos[1]))>11.5
                row.update({f'action_{j}':float(actions[i,j]) for j in range(12)})
                row.update({f'joint_velocity_{j}':float(t['dof_vel'][i,j]) for j in range(12)})
                assert all(np.isfinite(v) for v in row.values());rows[i].append(row)
                if row['done']:active[i]=False
    with (a.run/'rollout.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0][0]);w.writeheader();w.writerows(r for rs in rows for r in rs)
    trials=[]
    for i,rs in enumerate(rows):
        window=[r for r in rs if r['time']>2];mae={k:sum(abs(r[k]-(.5 if k=='vx' else 0)) for r in window)/len(window) if window else None for k in ['vx','vy','wz']}
        completed=len(rs)==600 and not any(r['done'] or r['solver_error'] or r['outside_map'] for r in rs)
        distance=rs[-1]['x']-initial[i][0];lateral=max(abs(r['y']-initial[i][1]) for r in rs)
        success=completed and distance>4 and lateral<.75 and mae['vx']<=.15 and mae['vy']<=.15 and mae['wz']<=.2
        trials.append(dict(trial=i,success=success,completed=completed,distance=distance,lateral=lateral,mae=mae))
    metrics=dict(status='evaluated',trials=trials,success_rate=sum(t['success'] for t in trials)/n)
metrics['elapsed_s']=time.monotonic()-started
(a.run/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
backup()
