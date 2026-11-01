"""Independent CSV audit: force timing, impulse and continuous recovery criteria."""
import argparse
import csv
import json
import math
from pathlib import Path


def audit(run):
    metrics=json.loads((run/'metrics.json').read_text())
    mass=json.loads((run/'force-api.json').read_text())['mass_kg']
    groups={i:[] for i in range(len(metrics['trials']))}
    with (run/'rollout.csv').open() as f:
        for raw in csv.DictReader(f):
            row={k:float(v) for k,v in raw.items() if k not in ('done','termination')}
            row['done']=raw['done']=='True'
            row['termination']=raw['termination']
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
            groups[int(row['trial'])].append(row)
    trials=[]
    for i,t in enumerate(metrics['trials']):
        rs=groups[i]; start=t['push_start_step']; end=start+10
        assert all(int(r['step'])==j and abs(r['time']-(j+1)*.02)<1e-8 for j,r in enumerate(rs))
        nonzero=[r for r in rs if math.hypot(r['fx'],r['fy'])>1e-5]
        expected=[s for s in range(start,end) if s<len(rs)]
        assert [int(r['step']) for r in nonzero]==expected,'wrong pulse steps'
        magnitude=t['alpha']*mass*9.81
        assert all(abs(math.hypot(r['fx'],r['fy'])-magnitude)<1e-4 and r['fz']==0 for r in nonzero)
        assert all(abs(r[k]-nonzero[0][k])<1e-5 for r in nonzero for k in ('fx','fy','fz'))
        def good(r):return not r['done'] and abs(r['vx']-.5)<=.1 and abs(r['vy'])<=.1 and abs(r['wz'])<=.15
        pre=rs[max(0,start-25):start]
        pre_valid=len(pre)==25 and all(good(r) for r in pre)
        dwell=0; recovered=None
        for r in rs[end:]:
            dwell=dwell+1 if good(r) else 0
            if dwell==25 and recovered is None:recovered=(int(r['step'])+1-end)*.02
        success=len(rs)==600 and not any(r['done'] for r in rs) and pre_valid and recovered is not None and recovered<=2+1e-8
        assert success==t['success'],('score mismatch',i,success,t)
        if recovered is not None:assert abs(recovered-t['recovery_s'])<1e-6
        trials.append(dict(trial=i,success=success,pre_valid=pre_valid,recovery_s=recovered,pulse_steps=len(nonzero),
                           impulse_N_s=sum(math.hypot(r['fx'],r['fy'])*.02 for r in nonzero)))
    return dict(status='passed',trials=trials,successes=sum(t['success'] for t in trials),count=len(trials))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    result=audit(a.run)
    (a.run/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
