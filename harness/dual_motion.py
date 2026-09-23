"""Contact-gated coordinated translation using official two-arm EE actions."""
import numpy as np
from robotwin_harness_v3 import SIDES,SkillError


def translate_pair(bridge,delta,steps):
    env=bridge.env
    if not all(bridge.sensors()[side]['holding'] for side in SIDES):
        raise SkillError('DUAL_MOVE needs measured holding contact in BOTH arms')
    origin=bridge.endpose();delta=np.asarray(delta,float)
    for alpha in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(delta)/.02))+1))[1:]:
        if env.eval_success:return
        if env.take_action_cnt>=env.step_lim:raise SkillError('Official action budget exhausted')
        targets={};plans={};originals={side:getattr(env.robot,side+'_plan_path') for side in SIDES}
        for side in SIDES:
            pose=origin[side+'_endpose'].copy()
            pose[:3]=(np.asarray(pose[:3])+alpha*delta).tolist();targets[side]=pose
            plans[side]=originals[side](pose)
            if plans[side].get('status')!='Success':raise SkillError('DUAL_MOVE unreachable for '+side)
        for side in SIDES:
            def cached(target,*args,side=side,**kwargs):
                if np.allclose(target,targets[side],atol=1e-6):return plans[side]
                return originals[side](target,*args,**kwargs)
            setattr(env.robot,side+'_plan_path',cached)
        try:
            env.take_action(targets['left']+[0.]+targets['right']+[0.],action_type='ee')
            if not env.eval_success:
                for _ in range(40):
                    for side in SIDES:env.robot.set_gripper(0.,side)
                    env.robot._entity_qf(env.robot.left_entity)
                    if env.robot.right_entity is not env.robot.left_entity:env.robot._entity_qf(env.robot.right_entity)
                    env.scene.step();env._update_render()
                    if env.check_success():env.eval_success=True;break
        finally:
            for side in SIDES:setattr(env.robot,side+'_plan_path',originals[side])
            bridge.depth_id=None
        after=bridge.endpose()
        errors={side:float(np.linalg.norm(np.array(after[side+'_endpose'][:3])-targets[side][:3])) for side in SIDES}
        steps.append(dict(name='dual_translate',target_ee=targets,position_error_m=errors,
                          endpose_after=after,success=bool(env.eval_success)))
        bridge.capture()
        if env.eval_success:return
        if max(errors.values())>.035:raise SkillError('DUAL_MOVE tracking error '+str(errors))
        if not all(bridge.sensors()[side]['holding'] for side in SIDES):
            raise SkillError('DUAL_MOVE lost contact; stop before moving either arm farther')
