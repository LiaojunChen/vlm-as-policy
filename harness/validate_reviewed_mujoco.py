"""Labelled physics/contract controls, never part of VLM evaluation scores."""
import json
from pathlib import Path
import numpy as np
from reviewed_mujoco import ReviewedGrasp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/interface_controls'

def projected_pixel(env,xyz):
    # Ground truth is permitted ONLY in this explicitly labelled control.
    camera=env.model.camera('head').id
    local=env.data.cam_xmat[camera].reshape(3,3).T@(np.array(xyz)-env.data.cam_xpos[camera])
    focal=120/np.tan(np.deg2rad(env.model.cam_fovy[camera])/2)
    return [round(159.5+focal*local[0]/-local[2]),round(119.5-focal*local[1]/-local[2])]

def main():
    rows=[]
    for name in ('contract_rejection','empty_grasp','primitives','intermediate_failure','pick_place_oracle'):
        directory=OUT/name
        env=ReviewedGrasp(0,directory,'pick_place');trace=[]
        def act(action):
            obs=env.observe()
            if 'pixel' in action:action.setdefault('observation_id',obs['observation_id'])
            result=env.execute(action);trace.append({'action':action,'result':result})
            return result
        try:
            if name=='contract_rejection':
                before=env.physics_steps
                for pixel in ([-1,120],[320,120],[12.5,20]):
                    assert act(dict(name='pick_at',arm='right',pixel=pixel))['failure_code']=='invalid_pixel'
                assert act(dict(name='pick_at',arm='right',pixel=[100,100],observation_id=-1))['failure_code']=='stale_observation'
                assert act(dict(name='place_at',arm='right',pixel=[180,100]))['failure_code']=='not_holding'
                assert env.physics_steps==before
                old=env.observe();act({'name':'wait'})
                result=env.execute(dict(name='pick_at',arm='right',pixel=[100,100],observation_id=old['observation_id']))
                trace.append({'stale_after_motion':result});assert result['failure_code']=='stale_observation'
            elif name=='empty_grasp':
                result=act(dict(name='pick_at',arm='right',pixel=projected_pixel(env,[-.15,.1,.7])))
                assert result['failure_code']=='empty_grasp' and not result['skill_success']
                assert not any(s['name']=='lift' for s in result['subactions'])
                assert not env.success
            elif name=='primitives':
                for action in [dict(name='move_delta',delta=[-.02,0,0]),dict(name='move_pose',xyz=[0,0,.95]),
                               dict(name='lift',distance=.04),dict(name='open_gripper'),dict(name='wait',duration=.1),
                               dict(name='reach_at',pixel=projected_pixel(env,[.14,.095,.701]))]:
                    result=act(dict(action,arm='right'));assert result['ok'],result
                result=act(dict(name='close_gripper',arm='right'));assert result['failure_code']=='empty_grasp'
                result=act(dict(name='done'));assert result['terminal'] and not result['success']
            elif name=='intermediate_failure':
                original=env.move
                def failing(xyz=None,opening=None,duration=.65,label='move'):
                    if label=='hover':return dict(name=label,motion_ok=False,position_error_m=.1)
                    return original(xyz,opening,duration,label)
                env.move=failing
                result=act(dict(name='pick_at',arm='right',pixel=projected_pixel(env,env.data.body('cube').xpos)))
                assert result['failure_code']=='motion_failed'
                assert [s['name'] for s in result['subactions']]==['open','hover']
            else:
                result=act(dict(name='pick_at',arm='right',pixel=projected_pixel(env,env.data.body('cube').xpos)))
                assert result['skill_success'] and result['gripper_sensor']['holding'],result
                result=act(dict(name='place_at',arm='right',pixel=projected_pixel(env,[.14,.095,.701])))
                assert result['skill_success'] and env.success,result
            rows.append(dict(control=name,passed=True,task_success=env.success,max_lift_m=env.max_lift_m))
        finally:
            env.close();(directory/'trace.json').write_text(json.dumps(trace,indent=2))
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
