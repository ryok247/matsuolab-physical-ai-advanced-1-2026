"""Bounded sequential smoke, same-map resume and baseline dev. No training jobs."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,json,subprocess,sys,uuid,time,hashlib,shutil
from pathlib import Path
from terrain_maps import generate
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);a=p.parse_args()
source=Path(__file__).parent
state=dict(phase='preflight',history=[],started=time.time())
def save():
    temp=a.session/'preflight-status.json.partial';temp.write_text(json.dumps(state,indent=2));temp.replace(a.session/'preflight-status.json')
    a.backup.mkdir(parents=True,exist_ok=True);shutil.copy2(a.session/'preflight-status.json',a.backup/'preflight-status.json')
def job(mode,checkpoint,seed,amplitude,extra=()):
    run=a.session/'jobs'/(f'{len(state["history"]):02d}-{mode}-'+uuid.uuid4().hex[:6])
    run.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,str(source/'terrain_worker.py'),mode,'--run',str(run),'--checkpoint',str(checkpoint),'--map-seed',str(seed),'--amplitude',str(amplitude),'--calibration',str(a.calibration),'--backup',str(a.backup/'jobs'),*extra]
    log=run.with_suffix('.log');state.update(current_run=str(run),command=cmd);save()
    with log.open('w') as f:
        process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=source)
        state['child_pid']=process.pid;save()
        try:code=process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            process.terminate();process.wait(timeout=30);raise
    if run.exists():shutil.copy2(log,run/'run.log');(run/'exit-code.txt').write_text(str(code))
    assert code==0,f'{mode} exit {code}; {log}'
    metrics=json.loads((run/'metrics.json').read_text())
    state['history'].append(dict(run=str(run),mode=mode,map_seed=seed,amplitude=amplitude,metrics=metrics));save()
    return run
try:
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()=='748a97c8a9eb1a1debb3d35e464518a850f731bb07764a821bd4ffe6c3435541'
    # Freeze formal maps before any gradient update, without evaluating them.
    formal=a.session/'formal-maps';formal.mkdir(exist_ok=False);manifest=[]
    for amplitude in [.01,.02,.04]:
        for seed in range(200,205):
            h,meta=generate(seed,amplitude);np.savez(formal/f'{seed}-{amplitude}.npz',height=h);manifest.append(meta)
    (formal/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copytree(formal,a.backup/'formal-maps')
    state['phase']='smoke';save();r=job('smoke',a.checkpoint,10,.01)
    state['phase']='resume_smoke';save();job('smoke',r/'checkpoint.pt',10,.01,['--resume'])
    state['phase']='baseline_dev';save()
    for amplitude in [.01,.02,.04]:
        for seed in range(100,103):job('eval',a.checkpoint,seed,amplitude,['--initial-count','2'])
    state['phase']='preflight_complete_training_pending';save()
except BaseException as e:
    state.update(phase='failed',error=f'{type(e).__name__}: {e}');save();raise
