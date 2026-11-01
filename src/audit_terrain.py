"""Recompute all terrain evaluation verdicts from before-reset CSV."""
import csv,json,math
from pathlib import Path
def audit(run):
    run=Path(run);m=json.loads((run/'metrics.json').read_text());initial=json.loads((run/'initial-states.json').read_text())
    rows=list(csv.DictReader((run/'rollout.csv').open()));groups={i:[] for i in range(len(initial))}
    for r in rows:groups[int(r['trial'])].append(r)
    verdicts=[]
    for i,rs in groups.items():
        assert rs and [int(r['step']) for r in rs]==list(range(len(rs)))
        assert all(abs(float(r['time'])-(j+1)*.02)<1e-9 for j,r in enumerate(rs))
        assert all(all(math.isfinite(float(v)) for k,v in r.items() if k not in ['done','solver_error','outside_map']) for r in rs)
        assert all(float(r['cmd_vx'])==(0 if j<50 else .5) for j,r in enumerate(rs))
        assert all(r['done']=='False' for r in rs[:-1]),'CSV contains post-reset states'
        window=[r for r in rs if float(r['time'])>2]
        mae={k:sum(abs(float(r[k])-(.5 if k=='vx' else 0)) for r in window)/len(window) if window else None for k in ['vx','vy','wz']}
        complete=len(rs)==600 and not any(r[k]=='True' for r in rs for k in ['done','solver_error','outside_map'])
        distance=float(rs[-1]['x'])-initial[i][0];lateral=max(abs(float(r['y'])-initial[i][1]) for r in rs)
        success=complete and distance>4 and lateral<.75 and mae['vx']<=.15 and mae['vy']<=.15 and mae['wz']<=.2
        assert success==m['trials'][i]['success'],(run,i,'verdict mismatch')
        verdicts.append(success)
    result=dict(status='passed',trials=len(verdicts),successes=sum(verdicts),before_reset=True)
    (run/'audit.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import sys
    print(json.dumps(audit(sys.argv[1]),indent=2))
