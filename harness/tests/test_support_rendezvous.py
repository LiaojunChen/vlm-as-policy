from copy import deepcopy
from robodawn.mission_safety import cooperative_support_action
from robodawn.mission_planner import RobodawnPlanner


def setup():
    step=dict(operation='transfer',source='item',destination='tray',arm='left',relation='on')
    obs=dict(observation_id=4,sensors={'left':dict(holding=True,remembered_object='item'),
                                    'right':dict(holding=True,remembered_object='tray')},
             endpose={'left_endpose':[-.2,-.1,1.,1.,0.,0.,0.],'right_endpose':[.3,-.1,1.,1.,0.,0.,0.]})
    failure=dict(action=dict(skill='place',arm='left',target='tray',release=True),
                 result=dict(skill_success=False,failure='No reachable placement orientation',subactions=[]))
    acquisition=dict(action=dict(skill='pick',arm='right',target='tray'),
                     result=dict(skill_success=True,subactions=[dict(name='lift',motion_ok=True)]))
    return obs,[acquisition,deepcopy(failure),deepcopy(failure)],step


def test_support_rendezvous_keeps_named_objects_and_both_arm_roles():
    obs,history,step=setup();attempts={}
    action=cooperative_support_action(obs,history,step,'Right arm holds tray while left places item',attempts)
    assert action['arm']=='right' and action['delta']==[-.15,0.,0.]
    assert action['support_rendezvous']['support']=='tray'
    p=RobodawnPlanner(None,'task');p.steps=[step];p.arm='left';p.phase='place';p.failures=2
    p.feedback(action,dict(skill_success=True,sensors=obs['sensors']))
    assert p.arm=='left' and p.index==0 and p.phase=='place' and p.failures==0
    assert cooperative_support_action(obs,history,step,'task',attempts)
    assert cooperative_support_action(obs,history,step,'task',attempts) is None


def test_support_recovery_requires_a_held_support_no_physical_intervention_and_movement_permission():
    obs,history,step=setup()
    assert cooperative_support_action(obs,history,step,'Keep tray stationary',{}) is None
    obs['sensors']['right']['holding']=False
    assert cooperative_support_action(obs,history,step,'task',{}) is None
    obs['sensors']['right']['holding']=True;history[-1]['result']['subactions']=[{}]
    assert cooperative_support_action(obs,history,step,'task',{}) is None


def test_near_midline_support_is_not_moved_indefinitely():
    obs,history,step=setup();obs['endpose']['right_endpose'][0]=.07
    attempts={}
    # Lateral centring alone does not prove mutual reachability. The executor
    # now searches a bounded current-grounded translation in both X and Y.
    assert cooperative_support_action(obs,history,step,'task',attempts)
    assert cooperative_support_action(obs,history,step,'task',attempts)
    assert cooperative_support_action(obs,history,step,'task',attempts) is None


def test_grasped_articulated_handle_does_not_authorize_support_translation():
    obs,history,step=setup();history[0]['action']['skill']='grasp_handle'
    assert cooperative_support_action(obs,history,step,'task',{}) is None


def test_staged_recovery_preserves_phase_and_rechecks_the_next_current_view():
    obs,history,step=setup();p=RobodawnPlanner(None,'task');p.steps=[step];p.arm='left';p.phase='place'
    a=dict(skill='move',arm='right',support_rendezvous={'support':'tray'})
    p.feedback(a,dict(skill_success=True,sensors=obs['sensors'],geometry={'staged_support_translation':True}))
    assert p.support_rendezvous_pending and p.index==0 and p.phase=='place'
    # The second stage does not invent failed placement rows after the move.
    history.append(dict(action=a,result={'skill_success':True}))
    assert cooperative_support_action(obs,history,step,'task',{},resume=True)
    p.feedback(a,dict(skill_success=False,sensors=obs['sensors']))
    assert not p.support_rendezvous_pending
