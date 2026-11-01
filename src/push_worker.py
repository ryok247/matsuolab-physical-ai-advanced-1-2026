"""Bounded physical push experiments. Saved command cfg is the only initial cfg."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from command_worker import dump, backup
from pushes import cases, score


def evaluate(env, runner, args):
    import torch
    trial_cases=cases(args.formal)
    obs=env.reset_with_yaws([0]*len(trial_cases),[c['seed'] for c in trial_cases])
    dump(args.run/'initial-states.json',[dict(c,qpos=q) for c,q in zip(trial_cases,env.evaluation_initial_qpos)])
    policy=runner.get_inference_policy(device=env.device)
    rows=[[] for _ in trial_cases]
    active=[True]*len(rows)
    # Development pushes vary in onset; formal always starts at t=4s.
    starts=[200 if args.formal else 175+int((c['seed']-90)*18.75) for c in trial_cases]
    try:
        with torch.no_grad():
            for step in range(600):
                mask=torch.tensor([active[i] and step==s for i,s in enumerate(starts)],device=env.device)
                env.start_push(mask,[c['alpha'] for c in trial_cases],[c['sign'] for c in trial_cases])
                obs=env.set_commands([[.5,0,0]]*len(rows))
                assert torch.allclose(obs['policy'][:,6:9],env.commands*env.commands_scale)
                action=policy(obs)
                if not torch.isfinite(action).all(): raise RuntimeError('nonfinite action')
                obs,_,_,_=env.step(action)
                state={k:v.detach().cpu().tolist() for k,v in env.terminal.items()}
                force=env.applied_force.cpu().tolist()
                actions=action.cpu().tolist()
                terms={k:v.cpu().tolist() for k,v in env.reward_values.items()}
                for i,c in enumerate(trial_cases):
                    if not active[i]:continue
                    vel=state['base_lin_vel'][i]; ang=state['base_ang_vel'][i]
                    pos=state['base_pos'][i]; euler=state['base_euler'][i]
                    done=bool(state['reset_buf'][i])
                    row=dict(step=step,time=(step+1)*.02,cmd_vx=.5,cmd_vy=0,cmd_wz=0,
                             vx=vel[0],vy=vel[1],vz=vel[2],wz=ang[2],x=pos[0],y=pos[1],z=pos[2],
                             roll=euler[0],pitch=euler[1],fx=force[i][0],fy=force[i][1],fz=force[i][2],
                             reward=state['rew_buf'][i],done=done,
                             termination='solver_error' if state['solver_error'][i] else ('roll_pitch_limit' if done else 'none'))
                    row.update({f'action_{j}':v for j,v in enumerate(actions[i])})
                    row.update({f'reward_{k}':v[i] for k,v in terms.items()})
                    row['joint_speed_rms']=math.sqrt(sum(v*v for v in state['dof_vel'][i])/12)
                    if not all(math.isfinite(v) for v in row.values() if isinstance(v,(float,int))):
                        raise RuntimeError('nonfinite rollout')
                    rows[i].append(row)
                    if done:active[i]=False
                if not any(active):break
    finally:
        nonempty=next((r for r in rows if r),None)
        if nonempty:
            with (args.run/'rollout.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['trial','alpha','sign','seed',*nonempty[0]])
                w.writeheader()
                for i,rs in enumerate(rows):
                    for row in rs:w.writerow(dict(trial=i,**trial_cases[i],**row))
    trials=[]
    for c,rs,start in zip(trial_cases,rows,starts):
        # Shift the dev onset to the common scoring origin (formal is unchanged).
        shifted=[dict(r,time=r['time']+(200-start)*.02) for r in rs]
        result=score(shifted)
        trials.append(dict(c,**result,steps=len(rs),push_start_step=start,
                           max_roll_pitch=max((max(abs(r['roll']),abs(r['pitch'])) for r in rs),default=None)))
    return dict(status='evaluated',formal=args.formal,trials=trials,
                success_rate=sum(t['success'] for t in trials)/len(trials),
                by_alpha={str(a):sum(t['success'] for t in trials if t['alpha']==a)/sum(t['alpha']==a for t in trials) for a in (.1,.2,.3)})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['smoke','train','eval','regression'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--backup',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=['weak','mixed'],default='weak')
    p.add_argument('--formal',action='store_true')
    p.add_argument('--iterations',type=int,default=301)
    p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--seed',type=int,default=81)
    args=p.parse_args()
    if not 1<=args.iterations<=301 or not 1<=args.num_envs<=1024:raise ValueError('experiment bounds')
    args.run.mkdir(parents=True,exist_ok=False)
    import torch
    import genesis as gs
    from push_env import PushEnv
    from rsl_rl.runners import OnPolicyRunner
    saved=json.loads((args.checkpoint.parent/'config.json').read_text())
    spec=json.loads(Path(__file__).with_name('upstream.lock.json').read_text())
    assert saved['upstream']==spec
    push_config=json.loads(Path(__file__).with_name('push-config.json').read_text())
    if args.resume and saved.get('push_config')!=push_config:raise ValueError('push config changed; cannot resume')
    cfgs=copy.deepcopy(saved['cfgs']); train_cfg=copy.deepcopy(saved['train_cfg'])
    assert cfgs[2]['reward_scales']['tracking_ang_vel']==.5,'must inherit command reward'
    assert cfgs[0]['termination_if_roll_greater_than']==10
    assert cfgs[0]['termination_if_pitch_greater_than']==10
    cfgs[0]['resampling_time_s']=4.
    if args.mode in ('eval','regression'):cfgs[0]['episode_length_s']=35. if args.mode=='regression' else 15.
    if args.resume and (saved.get('push_stage')!=args.stage or saved.get('task')!='go2-push-v1'):
        raise ValueError('resume must preserve task and stage')
    train_cfg['save_interval']=25
    config=dict(task='go2-push-v1',push_config=push_config,push_stage=args.stage,cfgs=cfgs,train_cfg=train_cfg,upstream=spec,
                initialization_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    dump(args.run/'config.json',config)
    names=['push-config.json','push_worker.py','push_env.py','pushes.py','command_env.py','command_worker.py','commands.py',
           'command-config.json','go2_env.py','go2_record_env.py','go2_train.py','upstream.lock.json']
    for name in names:shutil.copy2(Path(__file__).with_name(name),args.run/name)
    dump(args.run/'source-sha256.json',{n:hashlib.sha256((args.run/n).read_bytes()).hexdigest() for n in names})
    dump(args.run/'dependencies.json',{n:importlib.metadata.version(n) for n in ['genesis-world','torch','rsl-rl-lib']})
    (args.run/'requirements.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    torch.set_num_threads(2)
    gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=args.seed,performance_mode=True)
    evaluation=args.mode in ('eval','regression')
    from commands import cases as command_cases
    n=(len(command_cases(args.formal,'full',False)) if args.mode=='regression' else len(cases(args.formal))) if evaluation else (8 if args.mode=='smoke' else args.num_envs)
    env=PushEnv(n,cfgs,training_range=None if evaluation else ((.05,.15) if args.stage=='weak' else (.05,.30)),evaluation=evaluation)
    dump(args.run/'force-api.json',dict(mass_kg=env.mass,link_name=env.trunk.name,global_index=env.trunk.idx,ref='link_COM',local=False))
    class Runner(OnPolicyRunner):
        def save(self,path,infos=None):
            super().save(path,infos)
            backup(args.run,args.backup)
    runner=Runner(env,train_cfg,str(args.run),device=gs.device)
    runner.load(str(args.checkpoint),load_cfg=None if args.resume else dict(actor=True,critic=True,optimizer=False,iteration=False),map_location=str(gs.device))
    if args.resume:runner.current_learning_iteration+=1
    started=time.monotonic()
    if args.mode=='regression':
        from command_worker import evaluate as evaluate_commands
        metrics=evaluate_commands(env,runner,args,'full')
    elif evaluation:metrics=evaluate(env,runner,args)
    else:
        # Smoke explicitly exercises a pulse; training uses independent actual-reset clocks.
        if args.mode=='smoke':env.start_push(torch.ones(n,device=gs.device,dtype=torch.bool),.05,1)
        before=[v.detach().clone() for v in runner.alg.get_policy().parameters()]
        iterations=2 if args.mode=='smoke' else args.iterations
        runner.learn(num_learning_iterations=iterations,init_at_random_ep_len=True)
        delta=sum(float((a-b).abs().sum()) for a,b in zip(runner.alg.get_policy().parameters(),before))
        assert math.isfinite(delta) and delta>0
        runner.save(str(args.run/'checkpoint.pt'))
        metrics=dict(status='ppo_updates_completed',iterations=iterations,parameter_l1_change=delta,
                     pulses_applied=env.pulses_applied,optimizer_restored=args.resume)
    metrics['elapsed_s']=time.monotonic()-started
    dump(args.run/'metrics.json',metrics)
    backup(args.run,args.backup)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
