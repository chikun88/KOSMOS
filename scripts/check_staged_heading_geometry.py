#!/usr/bin/env python3
"""Non-ROS check usable on the Raspberry Pi gateway without pytest."""
from pathlib import Path
import sys
import math
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'ros2_ws/src/omni_autonomy_next'
sys.path.insert(0,str(PACKAGE))
from omni_autonomy_next.staged_heading import HeadingStage,prepare_stage,rotation_clearance
from omni_autonomy_next.rl_residual import CadClearanceModel
from omni_autonomy_next.route_approaches import load_fixed_goal_approaches
from omni_autonomy_next.configured_goals import load_configured_poses


def main():
    config=PACKAGE/'config'
    model=CadClearanceModel.from_yaml(config/'field_planning.yaml',config/'competition_footprints.yaml')
    goals=load_configured_poses(config/'field_poses.yaml')
    routes=load_fixed_goal_approaches(config/'routes.yaml')
    lanes=[]
    for goal_id in ('4','5'):
        for side in ('upper','lower'):
            points=[[p['x'],p['y']] for p in routes[goal_id][side]['waypoints']]
            goal=goals[goal_id]
            points.append([goal['x'],goal['y']])
            state,path,yaw=prepare_stage(model,points,np.array([*points[0],0.]),HeadingStage(0.,0.))
            clearance=min(model.body_clearance(p,yaw) for p in path)
            assert clearance >= .048 and state.phase == 'TRANSLATE'
            lanes.append(dict(goal_id=goal_id,side=side,minimum_clearance_mm=round(clearance*1000,3)))
    assert rotation_clearance(model,[-1.8,4.75],-math.pi/2,0.) < 0.
    assert rotation_clearance(model,[-1.6,3.4],-math.pi/2,0.) > .12
    print(json.dumps(dict(result='PASS',lanes=lanes),indent=2))


if __name__ == '__main__':
    main()
