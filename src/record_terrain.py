"""Prespecified seed200/A=.04, two 12s trials. Freeze frames after termination."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,copy,csv,json
from pathlib import Path
import numpy as np
import torch
import genesis as gs
import av
from PIL import Image,ImageDraw
from terrain_maps import generate
from terrain_env import TerrainEnv
from rsl_rl.runners import OnPolicyRunner
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
a.run.mkdir(parents=True,exist_ok=False)
height,meta=generate(200,.04);np.savez(a.run/'terrain.npz',height=height);(a.run/'terrain.json').write_text(json.dumps(meta,indent=2))
saved=json.loads((a.checkpoint.parent/'config.json').read_text());cfg=copy.deepcopy(saved['cfgs']);cfg[0]['episode_length_s']=15.
torch.set_num_threads(2);gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=81,performance_mode=True)
env=TerrainEnv(1,cfg,a.run/'terrain.npz',evaluation=True,record=True)
r=OnPolicyRunner(env,copy.deepcopy(saved['train_cfg']),None,device=gs.device)
r.load(str(a.checkpoint),load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False),map_location=str(gs.device));policy=r.get_inference_policy(device=gs.device)
video=av.open(str(a.run/'terrain.mp4'),'w');stream=video.add_stream('libx264',rate=25);stream.width,stream.height,stream.pix_fmt=640,480,'yuv420p'
rows=[];initial=[];trials=[]
try:
    for seed in [3000,3001]:
        env.reset_with_yaws([0],[seed]);initial.append(env.evaluation_initial_qpos)
        terminated=False;last_rgb=None;last_pos=env.robot.get_pos()[0].cpu().numpy();end_reason=None
        env.camera.set_pose(pos=last_pos+np.array([1.8,-2.,1.25]),lookat=last_pos+np.array([0,0,-.1]));last_rgb=env.camera.render()[0]
        with torch.no_grad():
            for step in range(600):
                command=[0,0,0] if step<50 else [.5,0,0]
                if not terminated:
                    obs=env.set_commands([command]);action=policy(obs);env.step(action);t=env.terminal
                    pos=t['base_pos'][0].cpu().numpy();vel=t['base_lin_vel'][0].cpu().tolist();angles=t['base_euler'][0].cpu().tolist()
                    assert torch.isfinite(action).all() and np.isfinite(pos).all() and np.isfinite(vel).all()
                    terminated=bool(t['reset_buf'][0]) or bool(t['solver_error'][0])
                    if terminated:end_reason='solver_error' if bool(t['solver_error'][0]) else ('outside_map' if max(abs(pos[:2]))>11.5 else 'termination_roll_pitch_or_timeout')
                    row=dict(seed=seed,step=step,time=(step+1)*.02,cmd_vx=command[0],x=float(pos[0]),y=float(pos[1]),z=float(pos[2]),vx=vel[0],vy=vel[1],wz=float(t['base_ang_vel'][0,2]),roll=angles[0],pitch=angles[1],done=terminated,ground_height=float(env.ground_height(t['base_pos'][:,:2])[0]))
                    rows.append(row)
                    if not terminated:last_pos=pos.copy()
                if step%2==0:
                    if not terminated or last_rgb is None:
                        env.camera.set_pose(pos=last_pos+np.array([1.8,-2.,1.25]),lookat=last_pos+np.array([0,0,-.1]));last_rgb=env.camera.render()[0]
                    picture=Image.fromarray(last_rgb.copy());draw=ImageDraw.Draw(picture);draw.rectangle((0,0,640,76),fill='black')
                    draw.text((8,4),f'{a.label} | seed200 A=+/-4cm | initial {seed} | playback 1x',fill='white')
                    draw.text((8,23),f't={(step+1)*.02:.2f}s | command vx={command[0]:.2f} | actual vx={vel[0]:.2f}',fill='white')
                    draw.text((8,43),f'roll={angles[0]:.1f} pitch={angles[1]:.1f} deg | '+(f'TERMINATED: {end_reason}; FRAME FROZEN' if terminated else 'running'),fill='yellow' if terminated else 'white')
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(picture),format='rgb24')):video.mux(packet)
        trials.append(dict(seed=seed,terminated=terminated,reason=end_reason))
finally:
    for packet in stream.encode():video.mux(packet)
    video.close()
    if rows:
        with (a.run/'record.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (a.run/'initial-states.json').write_text(json.dumps(initial));(a.run/'trials.json').write_text(json.dumps(trials,indent=2))
print('Recorded two trials; terminated segments freeze the last pre-reset rendered frame.')
