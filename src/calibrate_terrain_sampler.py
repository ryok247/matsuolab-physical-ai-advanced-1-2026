"""Check the installed surface-height API against both triangles in two envs."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,json
from pathlib import Path
import numpy as np
import torch
import genesis as gs
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
gs.init(backend=gs.cuda,precision='32',logging_level='warning',seed=91)
scene=gs.Scene(show_viewer=False)
h=np.array([[0,10,0],[20,50,10],[0,10,0]],dtype=float)
t=scene.add_entity(gs.morphs.Terrain(height_field=h,horizontal_scale=.1,vertical_scale=.001,pos=(-.1,-.1,0)))
scene.build(n_envs=2)
uv=np.array([[[.2,.3],[.7,.8]],[[.6,.1],[.9,.6]]])
xy=torch.tensor(uv*.1-.1,device=gs.device,dtype=torch.float32)
actual=t.get_terrain_height(xy).cpu().numpy()
u,v=uv[...,0],uv[...,1]
expected=np.where(u+v<=1,20*u+10*v,50+(1-v)*(20-50)+(1-u)*(10-50))*.001
error=float(np.abs(actual-expected).max())
assert np.isfinite(actual).all() and error<1e-6,(actual,expected,error)
result=dict(status='passed',method='Genesis Terrain.get_terrain_height; piecewise planar',max_error_m=error,actual=actual.tolist(),expected=expected.tolist())
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
