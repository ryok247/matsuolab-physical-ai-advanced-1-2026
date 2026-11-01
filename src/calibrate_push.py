"""GPU physical calibration of the inspected Genesis 1.4.5 wrench API."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import json
from pathlib import Path
import torch
import genesis as gs

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
gs.init(backend=gs.cuda, precision='32', logging_level='warning', seed=81,
        performance_mode=True)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.02, substeps=2, gravity=(0,0,0)),
                 show_viewer=False)
body = scene.add_entity(gs.morphs.Box(size=(.2,.2,.2), pos=(0,0,2)),
                        material=gs.materials.Rigid(rho=1000))
scene.build(n_envs=2)
mass = float(body.get_mass().flatten()[0])
link = body.links[0]
force = torch.tensor([[[0.,mass,0.]],[[0.,0.,0.]]], device=gs.device)
rows=[]
for step in range(20):
    # Ten 20ms steps, not twenty physics substeps; world-frame force at link COM.
    if step < 10:
        scene.rigid_solver.apply_links_external_wrench(force=force, links_idx=[link.idx],
                                                       ref=gs.link_ref_frame.link_COM, local=False)
    scene.step()
    velocity=body.get_vel().cpu().tolist()
    rows.append({'step':step,'time_s':(step+1)*.02,'velocity':velocity})
expected=.2
actual=rows[9]['velocity'][0][1]
assert abs(actual-expected)<.01,(actual,expected)
assert max(abs(x) for r in rows for x in r['velocity'][1])<1e-5,'environment force leak'
assert abs(rows[-1]['velocity'][0][1]-actual)<1e-4,'force persisted after release'
# Reset velocity and step without any wrench: no residual force may remain.
body.set_qpos(body.get_qpos(),zero_velocity=True,skip_forward=False)
scene.step()
assert float(body.get_vel().abs().max())<1e-5,'force survived reset'
result={'status':'passed','mass_kg':mass,'force_N':mass,'expected_delta_v':expected,
        'observed_delta_v':actual,'dt':.02,'substeps':2,'pulse_steps':10,
        'force_at':'link_COM','local':False,'rows':rows}
(a.output/'calibration.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
