"""Create isolated patched copies, preserving original locomotion/command sources."""
import argparse,hashlib,json,difflib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
src=(a.source/'go2_record_env.py').read_text()
start=src.index('        # add plain');end=src.index('        self.robot =',start)
replacement='''        # Terrain-only collision ground; no hidden plane beneath valleys.
        import numpy as np
        terrain_data = np.load(env_cfg['_terrain_file'])
        self.terrain = self.scene.add_entity(gs.morphs.Terrain(
            height_field=terrain_data['height'], horizontal_scale=0.1,
            vertical_scale=0.001, pos=(-12., -12., 0.)))

'''
new=src[:start]+replacement+src[end:]
anchor='        # Compute timeout'
assert new.count(anchor)==1
new=new.replace(anchor,"        self.reset_buf |= (torch.abs(self.base_pos[:, :2]) > 11.5).any(dim=1)\n\n"+anchor)

(a.output/'go2_terrain_base.py').write_text(new)
command=(a.source/'command_env.py').read_text();assert command.count('from go2_record_env import Go2Env')==1
(a.output/'terrain_command_base.py').write_text(command.replace('from go2_record_env import Go2Env','from go2_terrain_base import Go2Env'))
(a.output/'terrain-base.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),new.splitlines(True),fromfile='go2_record_env.py',tofile='go2_terrain_base.py')))
(a.output/'terrain-build-sha256.json').write_text(json.dumps({n:hashlib.sha256((a.output/n).read_bytes()).hexdigest() for n in ['go2_terrain_base.py','terrain_command_base.py']},indent=2))
