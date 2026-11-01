"""Bounded asynchronous pipeline; all jobs create fresh immutable run directories."""
import argparse,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
from command_worker import dump
from audit_pushes import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).parent
    a.session.mkdir(parents=True,exist_ok=True)
    lock=a.session/'pipeline.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state=dict(phase='starting',pid=os.getpid(),history=[],started=time.time())
    def save():
        dump(a.session/'status.json',state)
        a.backup.mkdir(parents=True,exist_ok=True)
        shutil.copy2(a.session/'status.json',a.backup/'status.json')
    def job(mode,checkpoint,stage='weak',resume=False,formal=False,iterations=301):
        name=f'{len(state["history"]):02d}-{mode}-{uuid.uuid4().hex[:6]}'
        run=a.session/'runs'/name;run.parent.mkdir(exist_ok=True)
        command=[sys.executable,'-u',str(source/'push_worker.py'),mode,'--run',str(run),'--checkpoint',str(checkpoint),
                 '--stage',stage,'--iterations',str(iterations),'--backup',str(a.backup/'runs')]
        if resume:command+=['--resume']
        if formal:command+=['--formal']
        state.update(phase=mode,current_run=str(run),command=command);save()
        log=a.session/(name+'.log')
        with log.open('w') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,MPLBACKEND='Agg',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            state['child_pid']=proc.pid;save()
            try:code=proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise RuntimeError('bounded job timeout')
        run.mkdir(exist_ok=True)
        shutil.copy2(log,run/'run.log');dump(run/'command.json',command);(run/'exit-code.txt').write_text(str(code))
        shutil.copytree(run,a.backup/'runs'/name,dirs_exist_ok=True)
        if code:raise RuntimeError(f'{name} failed exit={code}; inspect log')
        metrics=json.loads((run/'metrics.json').read_text())
        if mode=='eval':
            result=audit(run);dump(run/'audit.json',result)
            shutil.copy2(run/'audit.json',a.backup/'runs'/name/'audit.json')
        state['history'].append(dict(run=str(run),metrics=metrics));save()
        return run,metrics
    try:
        assert json.loads((a.session/'calibration/calibration.json').read_text())['status']=='passed'
        assert json.loads((a.session/'dev-baseline/metrics.json').read_text())['status']=='evaluated'
        audit(a.session/'dev-baseline')
        smoke,_=job('smoke',a.checkpoint)
        job('smoke',smoke/'checkpoint.pt',resume=True)
        checkpoint=a.checkpoint
        candidates=[]
        for stage,batches in [('weak',2),('mixed',3)]:
            for batch in range(batches):
                train,_=job('train',checkpoint,stage,resume=batch>0)
                checkpoint=train/'checkpoint.pt'
                dev,metrics=job('eval',checkpoint,stage)
                regression,reg=job('regression',checkpoint,stage)
                candidates.append(dict(checkpoint=str(checkpoint),dev=metrics['success_rate'],regression=reg['success_rate']))
                state['candidates']=candidates;save()
                if stage=='weak' and metrics['by_alpha']['0.1']>=.9:break
                if stage=='mixed' and metrics['by_alpha']['0.2']>=.9 and reg['success_rate']>=.9:break
        eligible=[c for c in candidates if c['regression']>=.9]
        best=max(eligible or candidates,key=lambda c:(c['dev'],c['regression']))
        state['selected']=best;save()
        for name,checkpoint in [('old',a.checkpoint),('new',Path(best['checkpoint']))]:
            run,_=job('eval',checkpoint,formal=True)
            state[name+'_formal']=str(run)
            run,_=job('regression',checkpoint,formal=True)
            state[name+'_regression']=str(run)
        state['phase']='evaluation_complete_recording_pending';save()
    except BaseException as e:
        state.update(phase='failed',error=f'{type(e).__name__}: {e}');save();raise

if __name__=='__main__':main()
