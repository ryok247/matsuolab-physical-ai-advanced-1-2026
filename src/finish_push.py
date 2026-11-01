"""Wait for existing evaluation, record comparison, export scalars and preserve files."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,csv,hashlib,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
source=Path(__file__).parent
out=a.session/'final-artifacts';out.mkdir(exist_ok=False)
def save(status,**extra):
    (out/'status.json').write_text(json.dumps(dict(phase=status,**extra),indent=2))
def command(cmd,log):
    with (out/log).open('w') as f:
        subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
try:
    save('waiting_evaluation')
    deadline=time.monotonic()+1800
    while True:
        state=json.loads((a.session/'status.json').read_text())
        if state['phase']=='failed':raise RuntimeError(state.get('error'))
        if state['phase']=='evaluation_complete_recording_pending':break
        if time.monotonic()>deadline:raise TimeoutError('evaluation still running')
        time.sleep(10)
    save('recording')
    for name,checkpoint in [('old',a.old),('new',Path(state['selected']['checkpoint']))]:
        command([sys.executable,source/'record_push.py','--checkpoint',checkpoint,'--run',out/name,'--label',name],'record-'+name+'.log')
    command(['ffmpeg','-n','-i',out/'old/push.mp4','-i',out/'new/push.mp4','-filter_complex','hstack=inputs=2','-c:v','libx264','-pix_fmt','yuv420p',out/'comparison.mp4'],'compose.log')
    import av
    with av.open(str(out/'comparison.mp4')) as v:
        stream=v.streams.video[0];fps=float(stream.average_rate)
        frames=sum(1 for _ in v.decode(video=0))
    assert frames==600 and fps==25,(frames,fps)
    assert json.loads((out/'old/initial-states.json').read_text())==json.loads((out/'new/initial-states.json').read_text())
    save('exporting_curves')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib.pyplot as plt
    rows=[];fig,axes=plt.subplots(2,1,figsize=(10,6))
    for job in state['history']:
        run=Path(job['run']);events=list(run.glob('**/events.out.tfevents.*'))
        if not events:continue
        cfg=json.loads((run/'config.json').read_text())
        label=run.name+' '+cfg['push_stage']
        dest=out/'raw'/run.name;dest.mkdir(parents=True)
        for event in events:
            shutil.copy2(event,dest/event.name)
            acc=EventAccumulator(str(event),size_guidance={'scalars':0});acc.Reload()
            for tag in acc.Tags()['scalars']:
                vals=acc.Scalars(tag)
                rows.extend(dict(run=run.name,stage=cfg['push_stage'],tag=tag,step=x.step,wall_time=x.wall_time,value=x.value) for x in vals)
                if tag in ['Train/mean_reward','Train/mean_episode_length']:
                    idx=0 if tag.endswith('mean_reward') else 1
                    axes[idx].plot([x.step for x in vals],[x.value for x in vals],label=label);axes[idx].set_ylabel(tag)
        if (run/'run.log').exists():shutil.copy2(run/'run.log',dest/'run.log')
    assert rows
    with (out/'scalars.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for axis in axes:axis.legend(fontsize=7);axis.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'learning-curves.png');plt.close(fig)
    save('saving')
    # Existing experiment directories are never deleted. Verify every copied file.
    hashes={}
    for path in a.session.rglob('*'):
        if not path.is_file() or path.name.endswith('.partial') or path.name in ['pipeline.lock','status.json']:continue
        relative=path.relative_to(a.session)
        destination=a.backup/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest()==digest
        hashes[str(relative)]=digest
    (out/'file-sha256.json').write_text(json.dumps(hashes,indent=2))
    summary=dict(selected=state['selected'],video_frames=frames,video_fps=fps,scalar_rows=len(rows),verified_files=len(hashes))
    for name in ['old','new']:
        for mode in ['formal','regression']:
            run=Path(state[name+'_'+mode]);m=json.loads((run/'metrics.json').read_text())
            summary[name+'_'+mode]=dict(run=str(run),successes=sum(t['success'] for t in m['trials']),count=len(m['trials']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    save('artifacts_ready_local_snapshot_pending',**summary)
    shutil.copytree(out,a.backup/'final-artifacts',dirs_exist_ok=True)
    archive=a.session/'push-results.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in a.session.rglob('*'):
            if path.is_file() and path!=archive and not path.name.endswith('.partial'):
                z.write(path,path.relative_to(a.session))
    shutil.copy2(archive,a.backup/archive.name)
    assert hashlib.sha256(archive.read_bytes()).digest()==hashlib.sha256((a.backup/archive.name).read_bytes()).digest()
    print(json.dumps(summary,indent=2))
except BaseException as e:
    save('failed',error=f'{type(e).__name__}: {e}');raise
