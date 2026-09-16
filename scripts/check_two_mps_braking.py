import hashlib,inspect,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path.cwd()/'scripts'))
import compare_field_response as module
# Add an output-only metric to the existing offline replay; control is identical.
source=inspect.getsource(module.replay)
needle='metric=dict(arrival_sec=arrival,'
assert source.count(needle)==1
exec(source.replace(needle,'metric=dict(goal_overshoot_m=float(max(0., (progress-distance).max())), arrival_sec=arrival,'),module.__dict__)
x=json.loads(Path('docs/two_mps_replay_20260915.json').read_text())
def evaluate(c):
 r=module.replay(c['gains'],[.38,.34],c['delay'],c['tau'],c['direction'],0.,1.,overrides=c['tuning'],distance=8.,profile='balanced',duration_sec=30.)
 return dict(session=c['session'],delay=c['delay'],tau=c['tau'],direction=c['direction'],result=r)
if __name__=='__main__':
 cases=[c for c in x['cases'] if c['distance']==8. and c['profile']=='balanced']
 with ProcessPoolExecutor(max_workers=3) as pool: rows=list(pool.map(evaluate,cases))
 for r in rows: print(r['session'],r['delay'],r['direction'],r['result']['goal_overshoot_m'])
 result=dict(kind='Offline long-axis braking diagnostic; output-only metric instrumentation',sources=x['sources'],diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows)
 Path('docs/two_mps_braking_20260915.json').write_text(json.dumps(result,indent=2)+'\n')
