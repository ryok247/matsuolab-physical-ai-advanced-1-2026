"""Wait for evaluation, record, export raw/scalar curves, and verify Drive files."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,csv,hashlib,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
from audit_terrain import audit
p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
source=Path(__file__).parent;out=a.session/'final-artifacts';out.mkdir(exist_ok=False)
def status(phase,**extra):
    (out/'status.json').write_text(json.dumps(dict(phase=phase,**extra),indent=2))
def command(cmd,log):
    with (out/log).open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
try:
    status('waiting_evaluation');deadline=time.monotonic()+5400
    while True:
        state=json.loads((a.session/'pipeline-status.json').read_text())
        if state['phase']=='failed':raise RuntimeError(state.get('error'))
        if state['phase']=='terrain_evaluation_complete_flat_record_save_pending':break
        if time.monotonic()>deadline:raise TimeoutError('evaluation wait exceeded')
        time.sleep(10)
    summary=dict(candidate=state['candidate'],training_job_cap=state['training_job_cap'],formal={},flat={})
    for label in ['old','new']:
        items=[j for j in state['history'] if j.get('formal_label')==label]
        if label=='new' and state.get('identical_policy_results_reused'):items=[j for j in state['history'] if j.get('formal_label')=='old']
        assert len(items)==15,(label,len(items))
        by_amplitude={}
        for amplitude in [.01,.02,.04]:
            subset=[j for j in items if j['amplitude']==amplitude];assert len(subset)==5
            for j in subset:audit(j['run'])
            trials=[t for j in subset for t in j['metrics']['trials']];assert len(trials)==20
            by_amplitude[str(amplitude)]=dict(successes=sum(t['success'] for t in trials),count=len(trials))
        summary['formal'][label]=by_amplitude
        flat=json.loads((Path(state[label+'_regression'])/'metrics.json').read_text());assert len(flat['trials'])==30
        summary['flat'][label]=dict(successes=sum(t['success'] for t in flat['trials']),count=30)
    old_items=[j for j in state['history'] if j.get('formal_label')=='old'];new_items=[j for j in state['history'] if j.get('formal_label')=='new']
    if new_items:
        for old in old_items:
            new=next(j for j in new_items if j['seed']==old['seed'] and j['amplitude']==old['amplitude'])
            assert (Path(old['run'])/'initial-states.json').read_bytes()==(Path(new['run'])/'initial-states.json').read_bytes()
    status('recording',**summary)
    for label,checkpoint in [('old',a.old),('new',Path(state['candidate']))]:
        command([sys.executable,source/'record_terrain.py','--checkpoint',checkpoint,'--run',out/label,'--label',label],'record-'+label+'.log')
    assert (out/'old/initial-states.json').read_bytes()==(out/'new/initial-states.json').read_bytes()
    command(['ffmpeg','-n','-i',out/'old/terrain.mp4','-i',out/'new/terrain.mp4','-filter_complex','hstack=inputs=2','-c:v','libx264','-pix_fmt','yuv420p',out/'comparison.mp4'],'compose.log')
    import av
    with av.open(str(out/'comparison.mp4')) as video:
        fps=float(video.streams.video[0].average_rate);frames=sum(1 for _ in video.decode(video=0))
    assert fps==25 and frames==600,(fps,frames)
    summary.update(video_fps=fps,video_frames=frames)
    status('exporting_curves')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib.pyplot as plt
    rows=[];fig,axes=plt.subplots(2,1,figsize=(10,6))
    for run in [Path(j['run']) for j in state['history'] if j['mode']=='train']:
        dest=out/'raw'/run.name;dest.mkdir(parents=True)
        shutil.copy2(run/'run.log',dest/'run.log')
        for event in run.glob('**/events.out.tfevents.*'):
            shutil.copy2(event,dest/event.name);acc=EventAccumulator(str(event),size_guidance={'scalars':0});acc.Reload()
            for tag in acc.Tags()['scalars']:
                values=acc.Scalars(tag);rows.extend(dict(run=run.name,tag=tag,step=v.step,wall_time=v.wall_time,value=v.value) for v in values)
                if tag in ['Train/mean_reward','Train/mean_episode_length']:
                    axis=axes[0 if tag.endswith('mean_reward') else 1];axis.plot([v.step for v in values],[v.value for v in values],label=run.name);axis.set_ylabel(tag)
    assert rows,'No training scalars'
    with (out/'scalars.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for axis in axes:axis.legend(fontsize=7);axis.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'learning-curves.png');plt.close(fig);summary['scalar_rows']=len(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    status('saving')
    hashes={}
    for path in a.session.rglob('*'):
        if not path.is_file() or path.name.endswith('.partial') or path.name in ['status.json','pipeline-status.json','preflight-status.json']:continue
        relative=path.relative_to(a.session);target=a.backup/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);digest=hashlib.sha256(path.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==digest;hashes[str(relative)]=digest
    (out/'file-sha256.json').write_text(json.dumps(hashes,indent=2));summary['verified_files']=len(hashes)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));status('drive_verified_local_snapshot_pending',**summary)
    shutil.copytree(out,a.backup/'final-artifacts',dirs_exist_ok=True)
    # Small review bundle excludes model binaries and redundant source revisions.
    archive=a.session/'terrain-evidence.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in a.session.rglob('*'):
            if not path.is_file() or path.suffix in ['.pt','.zip','.mp4'] or 'source-before-' in str(path):continue
            z.write(path,path.relative_to(a.session))
    shutil.copy2(archive,a.backup/archive.name);assert hashlib.sha256(archive.read_bytes()).digest()==hashlib.sha256((a.backup/archive.name).read_bytes()).digest()
    print(json.dumps(summary,indent=2))
except BaseException as e:
    status('failed',error=f'{type(e).__name__}: {e}');raise
