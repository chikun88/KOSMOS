import sys,json,math
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
import check_bucket_gate_replay as route
original=route.tracker.TrajectoryTracker._rate_limit

def frame_rate_limit(node,*args):
 dt=getattr(node,'control_dt',node.period)
 lag=float(node.get_parameter('feedback_delay_sec').value)
 yaw=float(node.pose[2])+float(np.clip(node.velocity[2]*lag,-.35,.35))
 previous=getattr(node,'probe_previous_yaw',yaw);node.probe_previous_yaw=yaw
 angle=route.tracker.wrap(yaw-previous)
 command=node.command.copy()
 try:
  if abs(angle)<2.*node.yaw_limit*dt:
   node.command[:2]=route.tracker.to_body(command[:2],angle)
  return original(node,*args)
 finally:node.command=command

def evaluate(job):
 c,g,d,t=job
 with patch.object(route.tracker,'sprint_turn_clearance',lambda p,y,m,**kw:np.ones(len(p),dtype=bool)),patch.object(route.tracker.TrajectoryTracker,'_rate_limit',frame_rate_limit):
  after=route.replay(c,True,d,t,mode='fast',response_gains=g)
 return dict(case=c['plan_sequence'],delay=d,after=after)
if __name__=='__main__':
 root=Path.cwd();out=[];jobs=[]
 for prefix in ('unrestricted_turn','latest_speed_retry'):
  cs=json.loads((root/f'docs/{prefix}_routes_20260916.json').read_text())['runs'][0]['cases'];g=[a['gain'] for a in json.loads((root/f'docs/{prefix}_audit_20260916.json').read_text())['runs'][0]['exploratory_fit']]
  jobs.extend((c,g,d,t) for c in cs for d,t in ((.2,.12),(.3,.15)))
 with ProcessPoolExecutor(max_workers=2) as pool:
  for r in pool.map(evaluate,jobs):
   out.append(r);v=r['after'];print(r['case'],r['delay'],round(v['peak_speed_m_s'],3),v['arrival_sec'],round(v['min_cad_clearance_m'],3),flush=True)
 (root/'docs/rotating_frame_candidate_20260916.json').write_text(json.dumps(out,indent=2)+'\n')
