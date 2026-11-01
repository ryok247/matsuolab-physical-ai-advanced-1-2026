"""Contact-height and two-environment calibration on a known planar height field."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,json
from pathlib import Path
import numpy as np
import torch
import genesis as gs
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
a.run.mkdir(parents=True,exist_ok=False)
gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=91)
scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.02,substeps=2),show_viewer=False)
axis=np.arange(41)*.1
# Surface world z=.15+.05*x+.025*y, morph origin (-2,-2,0).
x,y=np.meshgrid(axis-2,axis-2,indexing='ij');field=(.15+.05*x+.025*y)/.001
terrain=scene.add_entity(gs.morphs.Terrain(height_field=field,horizontal_scale=.1,vertical_scale=.001,pos=(-2,-2,0)))
ball=scene.add_entity(gs.morphs.Sphere(radius=.05,pos=(0,0,.6)),material=gs.materials.Rigid(rho=1000))
scene.build(n_envs=2)
positions=torch.tensor([[0.,0.,.6],[.6,-.4,.6]],device=gs.device)
ball.set_pos(positions,zero_velocity=True)
for _ in range(150):scene.step()
observed=ball.get_pos().cpu().numpy()
# Sphere on a plane rests along the normal: z-center at fixed x,y equals plane height + r*sqrt(1+a²+b²).
expected=.15+.05*observed[:,0]+.025*observed[:,1]+.05*np.sqrt(1+.05**2+.025**2)
error=np.abs(observed[:,2]-expected)
result=dict(observed=observed.tolist(),expected_z=expected.tolist(),error_m=error.tolist(),
            finite=bool(np.isfinite(observed).all()),height_field_shape=list(field.shape))
(a.run/'calibration.json').write_text(json.dumps(result,indent=2))
assert result['finite'] and float(error.max())<.005,result
print(json.dumps(dict(status='passed',**result),indent=2))
