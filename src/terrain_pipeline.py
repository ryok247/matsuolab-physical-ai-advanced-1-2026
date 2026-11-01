"""One bounded adaptation job, dev/flat checks and fixed formal comparison.

Recording and final preservation are a separate required finishing stage.
"""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,json,subprocess,sys,uuid,time,shutil
from pathlib import Path
from audit_terrain import audit
p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);a=p.parse_args()
source=Path(__file__).parent
assert not (a.session/'pipeline-status.json').exists(),'Inspect existing pipeline before launching'
state=dict(phase='waiting_preflight',history=[],started=time.time(),training_job_cap=1)
def save():
    temp=a.session/'pipeline-status.json.partial';temp.write_text(json.dumps(state,indent=2));temp.replace(a.session/'pipeline-status.json')
    a.backup.mkdir(parents=True,exist_ok=True);shutil.copy2(a.session/'pipeline-status.json',a.backup/'pipeline-status.json')
def job(mode,checkpoint,seed,amplitude,extra=()):
    run=a.session/'adaptation'/(f'{len(state["history"]):02d}-{mode}-'+uuid.uuid4().hex[:6]);run.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,str(source/'terrain_worker.py'),mode,'--run',str(run),'--checkpoint',str(checkpoint),'--map-seed',str(seed),'--amplitude',str(amplitude),'--calibration',str(a.session/'sampler-calibration.json'),'--backup',str(a.backup/'adaptation'),*extra]
    log=run.with_suffix('.log');state.update(current_run=str(run),command=cmd);save()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=source);state['child_pid']=proc.pid;save()
        try:code=proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            proc.terminate();proc.wait(timeout=30);raise
    if run.exists():shutil.copy2(log,run/'run.log');(run/'exit-code.txt').write_text(str(code))
    assert code==0,f'{mode} exit {code}: {log}'
    metrics=json.loads((run/'metrics.json').read_text())
    if mode=='eval':audit(run)
    state['history'].append(dict(run=str(run),mode=mode,seed=seed,amplitude=amplitude,metrics=metrics));save();return run
try:
    save();deadline=time.monotonic()+2400
    while True:
        pre=json.loads((a.session/'preflight-status.json').read_text())
        if pre['phase']=='failed':raise RuntimeError(pre.get('error'))
        if pre['phase']=='preflight_complete_training_pending':break
        if time.monotonic()>deadline:raise TimeoutError('preflight wait exceeded')
        time.sleep(10)
    for item in pre['history']:
        if item['mode']=='eval':audit(item['run'])
    baseline=[t for item in pre['history'] if item['mode']=='eval' for t in item['metrics']['trials']]
    state['old_dev_successes']=sum(t['success'] for t in baseline);state['dev_count']=len(baseline)
    # Cap reduced for the 10:20 UTC outcome-freeze target. No reward search.
    if state['old_dev_successes']==len(baseline):
        state.update(phase='baseline_ceiling_no_adaptation',candidate=str(a.old),adaptation_skipped=True);save()
    else:
        state['phase']='training_weak';save();trained=job('train',a.old,10,.01,['--num-envs','256','--iterations','301'])
        state.update(candidate=str(trained/'checkpoint.pt'),phase='candidate_dev');save()
        for amplitude in [.01,.02,.04]:
            for seed in range(100,103):job('eval',Path(state['candidate']),seed,amplitude,['--initial-count','2'])
    state['phase']='flat_regression';save()
    for label,checkpoint in [('old',a.old),('new',Path(state['candidate']))]:
        if label=='new' and state.get('adaptation_skipped'):
            state['new_regression']=state['old_regression'];state['identical_policy_results_reused']=True;save();continue
        r=job('regression',checkpoint,0,0)
        state[label+'_regression']=str(r);save()
    # Candidate is fixed before formal outcomes, even if it underperforms.
    state['phase']='formal';save();manifest=json.loads((a.session/'formal-maps/manifest.json').read_text())
    for label,checkpoint in [('old',a.old),('new',Path(state['candidate']))]:
        if label=='new' and state.get('adaptation_skipped'):continue
        state['formal_label']=label;save()
        for amplitude in [.01,.02,.04]:
            for seed in range(200,205):
                run=job('eval',checkpoint,seed,amplitude)
                cfg=json.loads((run/'config.json').read_text())
                expected=next(m for m in manifest if m['seed']==seed and m['amplitude_m']==amplitude)
                assert cfg['terrain']==expected,'formal map changed'
                state['history'][-1]['formal_label']=label;save()
    state['phase']='terrain_evaluation_complete_flat_record_save_pending';save()
except BaseException as e:
    state.update(phase='failed',error=f'{type(e).__name__}: {e}');save();raise
