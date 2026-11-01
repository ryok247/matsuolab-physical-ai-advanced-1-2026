"""Two prespecified 12s demonstrations (left/right), with measured telemetry overlay."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,copy,csv,json,math
from pathlib import Path
import numpy as np
import torch
import genesis as gs
import av
from PIL import Image,ImageDraw
from push_env import PushEnv
from rsl_rl.runners import OnPolicyRunner
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
a.run.mkdir(parents=True,exist_ok=False)
saved=json.loads((a.checkpoint.parent/'config.json').read_text());cfgs=copy.deepcopy(saved['cfgs']);cfgs[0]['episode_length_s']=15
(torch.set_num_threads(2))
gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=81,performance_mode=True)
env=PushEnv(1,cfgs,evaluation=True,record=True)
runner=OnPolicyRunner(env,copy.deepcopy(saved['train_cfg']),None,device=gs.device)
runner.load(str(a.checkpoint),load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False),map_location=str(gs.device))
policy=runner.get_inference_policy(device=gs.device)
video=av.open(str(a.run/'push.mp4'),'w');stream=video.add_stream('libx264',rate=25)
stream.width,stream.height,stream.pix_fmt=640,480,'yuv420p'
rows=[];initial=[]
try:
    for sign in (-1,1):
        env.reset_with_yaws([0],[2000]);initial.append(env.evaluation_initial_qpos)
        terminated=False; streak=0; recovery=None
        with torch.no_grad():
            for step in range(600):
                if step==200 and not terminated:env.start_push(torch.tensor([True],device=gs.device),.3,sign)
                obs=env.set_commands([[.5,0,0]])
                env.step(policy(obs))
                t=env.terminal
                vel=t['base_lin_vel'][0].cpu().tolist();wz=float(t['base_ang_vel'][0,2]);pos=t['base_pos'][0].cpu().numpy()
                done=bool(t['reset_buf'][0]);terminated=terminated or done
                force=env.applied_force[0].cpu().tolist()
                good=not terminated and abs(vel[0]-.5)<=.1 and abs(vel[1])<=.1 and abs(wz)<=.15
                if step>=210:
                    streak=streak+1 if good else 0
                    if streak>=25 and recovery is None:recovery=(step+1-210)*.02
                row=dict(sign=sign,step=step,time=(step+1)*.02,vx=vel[0],vy=vel[1],wz=wz,fx=force[0],fy=force[1],done=done,terminated=terminated)
                rows.append(row)
                if step%2==0:
                    env.camera.set_pose(pos=pos+np.array([1.8,-2.,1.3]),lookat=pos)
                    picture=Image.fromarray(env.camera.render()[0]);draw=ImageDraw.Draw(picture)
                    draw.rectangle((0,0,640,83),fill='black')
                    draw.text((8,4),f'{a.label} | {"LEFT" if sign==1 else "RIGHT"} | t={(step+1)*.02:.2f}s | playback 1x',fill='white')
                    draw.text((8,22),f'command vx=.50 vy=0 yaw=0 | actual {vel[0]:.2f} {vel[1]:.2f} {wz:.2f}',fill='white')
                    magnitude=math.hypot(force[0],force[1])
                    draw.text((8,40),f'push alpha=.30 | applied {magnitude:.1f}N | '+('PUSHING' if magnitude>0 else 'force off'),fill='yellow' if magnitude else 'white')
                    draw.text((8,58),f'recovery (25 good steps): {recovery if recovery is not None else "pending"} s | terminated={terminated}',fill='white')
                    if magnitude:
                        start=(320,180);end=(320+sign*90,180)
                        draw.line((start,end),fill='red',width=7)
                        draw.polygon((end,(end[0]-sign*18,end[1]-10),(end[0]-sign*18,end[1]+10)),fill='red')
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(picture),format='rgb24')):video.mux(packet)
finally:
    for packet in stream.encode():video.mux(packet)
    video.close()
    if rows:
        with (a.run/'record.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (a.run/'initial-states.json').write_text(json.dumps(initial))
print('Recorded',len(rows),'steps',a.run)
