"""Descriptive auxiliary metrics from all formal trials; no model selection."""
import csv,json,math,statistics,argparse
from pathlib import Path

def summarize(run):
    groups={}
    with (run/'rollout.csv').open() as f:
        for r in csv.DictReader(f):groups.setdefault(int(r['trial']),[]).append(r)
    results=[]
    for trial,rows in groups.items():
        pulse=[r for r in rows if math.hypot(float(r['fx']),float(r['fy']))>0]
        first=pulse[0]; mag=math.hypot(float(first['fx']),float(first['fy']))
        axis=[float(first['fx'])/mag,float(first['fy'])/mag]
        # World displacement along the actual signed push direction, from just before onset.
        onset=int(first['step']);before=next(r for r in rows if int(r['step'])==onset-1)
        post=[r for r in rows if 4.2<float(r['time'])<=7.2]
        lateral=[(float(r['x'])-float(before['x']))*axis[0]+(float(r['y'])-float(before['y']))*axis[1] for r in rows if int(r['step'])>=onset]
        changes=[sum((float(b[f'action_{j}'])-float(a[f'action_{j}']))**2 for j in range(12))/12 for a,b in zip(rows,rows[1:])]
        results.append(dict(trial=trial,alpha=float(first['alpha']),seed=int(first['seed']),sign=int(first['sign']),
            velocity_error_integral_3s_m=sum(math.hypot(float(r['vx'])-.5,float(r['vy']))*.02 for r in post),
            post_window_steps=len(post),max_abs_roll_pitch_deg=max(max(abs(float(r['roll'])),abs(float(r['pitch']))) for r in rows),
            max_abs_push_axis_displacement_m=max(map(abs,lateral)),final_push_axis_displacement_m=lateral[-1],
            joint_velocity_rms_rad_s=math.sqrt(statistics.mean(float(r['joint_speed_rms'])**2 for r in rows)),
            action_step_delta_rms=math.sqrt(statistics.mean(changes)),terminated=any(r['done']=='True' for r in rows)))
    assert len(results)==60 and all(r['post_window_steps']==150 for r in results)
    fields=['velocity_error_integral_3s_m','max_abs_roll_pitch_deg','max_abs_push_axis_displacement_m','final_push_axis_displacement_m','joint_velocity_rms_rad_s','action_step_delta_rms']
    return dict(trials=results,by_alpha={str(a):{k:dict(median=statistics.median(r[k] for r in results if r['alpha']==a),max=max(r[k] for r in results if r['alpha']==a)) for k in fields} for a in (.1,.2,.3)},scope='All 60 fixed formal trials; descriptive only; lateral displacement uses signed applied-force axis, not path restoration.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.session/'status.json').read_text())
    result={n:summarize(Path(state[n+'_formal'])) for n in ('old','new')}
    target=a.session/'final-artifacts/auxiliary-metrics.json';target.write_text(json.dumps(result,indent=2));print(target)
