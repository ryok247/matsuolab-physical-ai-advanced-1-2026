"""Two prespecified candidates; fresh holdout, same success criteria."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,json,subprocess,sys,time,uuid,shutil
from pathlib import Path
import numpy as np
from terrain_maps import generate
from audit_terrain import audit
p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);a=p.parse_args()
src=Path(__file__).parent;spec=json.loads((src/'terrain_retry_spec.json').read_text());a.session.mkdir(parents=True,exist_ok=False)
state=dict(phase='freezing',history=[],started=time.time(),training_job_cap=6,spec=spec)
def save():
 t=a.session/'pipeline-status.json.partial';t.write_text(json.dumps(state,indent=2));t.replace(a.session/'pipeline-status.json');a.backup.mkdir(parents=True,exist_ok=True);shutil.copy2(a.session/'pipeline-status.json',a.backup/'pipeline-status.json')
def job(mode,cp,seed,amp,extra=()):
 assert time.time()-state['started']<spec['time_cap_seconds'],'time cap'
 run=a.session/'adaptation'/(f'{len(state["history"]):02d}-{mode}-'+uuid.uuid4().hex[:6]);run.parent.mkdir(exist_ok=True)
 cmd=[sys.executable,str(src/'terrain_worker.py'),mode,'--run',str(run),'--checkpoint',str(cp),'--map-seed',str(seed),'--amplitude',str(amp),'--calibration',str(a.calibration),'--backup',str(a.backup/'adaptation'),*map(str,extra)]
 state.update(current_run=str(run),command=cmd);save()
 with run.with_suffix('.log').open('w') as f:
  proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=src);state['child_pid']=proc.pid;save()
  try:code=proc.wait(timeout=1200)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=30);raise
 if run.exists():shutil.copy2(run.with_suffix('.log'),run/'run.log');(run/'exit-code.txt').write_text(str(code))
 assert code==0,(code,run)
 m=json.loads((run/'metrics.json').read_text())
 if mode=='eval':audit(run)
 state['history'].append(dict(run=str(run),mode=mode,seed=seed,amplitude=amp,metrics=m));save();return run
try:
 formal=a.session/'formal-maps';formal.mkdir();manifest=[]
 for amp in spec['amplitudes']:
  for seed in spec['formal_seeds']:
   h,meta=generate(seed,amp);np.savez(formal/f'{seed}-{amp}.npz',height=h);manifest.append(meta)
 (formal/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copytree(formal,a.backup/'formal-maps')
 scores=[]
 # Smoke uses the new reward and fixed learning rate, isolated from real candidates.
 state['phase']='smoke';save();job('smoke',a.old,11,.02,['--learning-rate',.0001,'--posture-scale',-.5])
 for c in spec['candidates']:
  state.update(phase='training',candidate_name=c['name']);save();cp=a.old
  for seed,amp in spec['training_maps']:
   r=job('train',cp,seed,amp,['--learning-rate',c['learning_rate'],'--posture-scale',c['posture_scale'],'--num-envs',256,'--iterations',301]);cp=r/'checkpoint.pt'
  state['phase']='candidate_dev';save();trials=[]
  for amp in spec['amplitudes']:
   for seed in spec['dev_seeds']:
    r=job('eval',cp,seed,amp,['--initial-count',2]);trials.extend((amp,t) for t in json.loads((r/'metrics.json').read_text())['trials'])
  flat=job('regression',cp,0,0);fm=json.loads((flat/'metrics.json').read_text());fs=sum(t['success'] for t in fm['trials'])
  scores.append(dict(name=c['name'],checkpoint=str(cp),dev=sum(t['success'] for _,t in trials),strong=sum(t['success'] for amp,t in trials if amp==.04),flat=fs,regression=str(flat)))
  state['candidates']=scores;save()
 eligible=[c for c in scores if c['flat']>=27 and c['dev']>=17]
 chosen=max(eligible,key=lambda c:(c['dev'],c['strong'])) if eligible else None
 state.update(candidate=chosen['checkpoint'] if chosen else str(a.old),selection=chosen,phase='flat_regression');save()
 oldflat=job('regression',a.old,0,0);state['old_regression']=str(oldflat)
 state['new_regression']=chosen['regression'] if chosen else str(oldflat);save()
 state['phase']='formal';save()
 for label,cp in [('old',a.old),('new',Path(state['candidate']))]:
  for amp in spec['amplitudes']:
   for seed in spec['formal_seeds']:
    r=job('eval',cp,seed,amp,['--initial-seed',5000])
    cfg=json.loads((r/'config.json').read_text());assert cfg['terrain']==next(m for m in manifest if m['seed']==seed and m['amplitude_m']==amp)
    state['history'][-1]['formal_label']=label;save()
 state['phase']='terrain_evaluation_complete_flat_record_save_pending';save()
except BaseException as e:
 state.update(phase='failed',error=f'{type(e).__name__}: {e}');save();raise
