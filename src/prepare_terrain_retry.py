"""Build an isolated retry revision; never mutate prior sources/results."""
import argparse,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
shutil.copytree(a.source,a.output)
w=(a.source/'terrain_worker.py').read_text()
anchor="p.add_argument('--backup',type=Path)"
assert w.count(anchor)==1
w=w.replace(anchor,anchor+"\np.add_argument('--learning-rate',type=float)\np.add_argument('--posture-scale',type=float)\np.add_argument('--initial-seed',type=int,default=3000)")
anchor="if a.mode=='regression':cfgs[0]['episode_length_s']=35."
assert w.count(anchor)==1
w=w.replace(anchor,anchor+"\nif a.learning_rate is not None:\n    assert 0<a.learning_rate<=.001\n    tcfg['algorithm']['learning_rate']=a.learning_rate\n    tcfg['algorithm']['schedule']='fixed'\nif a.posture_scale is not None:\n    assert -2<=a.posture_scale<=0\n    if a.posture_scale:cfgs[2]['reward_scales']['posture']=a.posture_scale\n    else:cfgs[2]['reward_scales'].pop('posture',None)")
w=w.replace('list(range(3000,3000+n))','list(range(a.initial_seed,a.initial_seed+n))')
(a.output/'terrain_worker.py').write_text(w)
e=(a.source/'terrain_env.py').read_text()
e+='\n    def _reward_posture(self):\n        # Euler buffer is degrees; penalize squared roll/pitch in radians.\n        return torch.square(torch.deg2rad(self.base_euler[:,:2])).sum(dim=1)\n'
(a.output/'terrain_env.py').write_text(e)
r=(a.source/'record_terrain.py').read_text()
r=r.replace('generate(200,.04)','generate(400,.04)').replace('[3000,3001]','[5000,5001]').replace('seed200','seed400')
(a.output/'record_terrain.py').write_text(r)
# The flat CommandEnv has no terrain-only posture reward; inference does not use rewards.
w=(a.output/'terrain_worker.py').read_text()
anchor="if a.mode=='regression':cfgs[0]['episode_length_s']=35."
w=w.replace(anchor,anchor+"\nif a.mode=='regression':cfgs[2]['reward_scales'].pop('posture',None)")
(a.output/'terrain_worker.py').write_text(w)
