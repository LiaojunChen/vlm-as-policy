"""Same-model semantic plan review using only the task and policy observation.

Review is advice for bounded replanning, not an alternate success checker and
not a source of physical coordinates or privileged object facts.
"""
import json
import numpy as np
from PIL import Image
from robotwin_harness_v3 import SkillError,parse_json
from robodawn.semantic_planner import policy_sensor_view,history_summary
from robodawn.schemas import obj,array,TEXT

REVIEW_SCHEMA=obj(dict(valid=dict(type='boolean'),issues=array(
    obj(dict(requirement=TEXT,problem=TEXT,correction=TEXT)),0,4)))

REVIEW_PROMPT='''Check whether the candidate robot plan preserves the user's request.
You are a plan reviewer, not the executor or task success checker. Use ONLY the
task, current image/contact state, executed history and candidate plan below.
Do not invent evaluator rules, hidden object poses, coordinates or new goals.

Operation semantics:
TRANSFER picks its source if necessary, places it at the destination, RELEASES
and homes the empty hand. LIFT picks and RETAINS its source; stay/centre/side only
describe presentation. BIMANUAL_LIFT holds ONE object with both hands and lifts,
but does not perform later transport or release. HANDOVER acquires/presents with
the donor, verifies receiver contact, then releases only the donor. SLIDE pushes
an object to its destination. TOOL_CONTACT acquires its tool, uses contact_part
on the destination, and retains the tool. ACTION executes only its named skill;
move/rotate do not release; open releases; home requires an empty hand.

Check every requested object and final relation/direction, explicit arm
assignments, required release/retention, and temporal requirements such as
holding a support WHILE the other arm inserts an object. A later support lift
does not mean it was held during an earlier insertion. Acquiring an object alone
does not complete requested transport, orientation, pouring or placement.
Do not demand genuinely simultaneous commands when sequential acquisitions
retain the objects and satisfy the requested overlap. Do not reject harmless
acquisition redundancy or a valid plan just because you prefer another style.
Consider executed history: already completed stages need not be repeated.
The requested result must hold AFTER THE LAST step. Reaching a requested
destination and then moving the object back somewhere else does NOT preserve
the final goal. Read the last named operation for each object below.
Flag only definite missing or contradictory requirements, not speculative
physics or unnecessary extra goals. If there is no definite discrepancy, valid=true.
Return JSON {"valid":true,"issues":[]} or valid=false with at most FOUR issues.
Each issue has requirement (short exact quote), problem, and correction (semantic
change, not invented coordinates or a claim that the task is completed).
'''


def review_plan(client,instruction,steps,observation,history):
    last_named={}
    for index,step in enumerate(steps):
        if step.get('source'):
            last_named[' '.join(step['source'].lower().split())]=dict(step_index=index,operation=step)
    prompt=(REVIEW_PROMPT+'\nTASK: '+instruction+'\nCURRENT CONTACT STATE: '
            +json.dumps(policy_sensor_view(observation['sensors']))+'\nEXECUTED HISTORY: '
            +json.dumps(history_summary(history))+'\nCANDIDATE PLAN: '+json.dumps(steps)
            +'\nLAST NAMED OPERATION PER OBJECT (symbolic plan audit, NOT measured success): '+json.dumps(last_named))
    image=np.asarray(Image.open(observation['image_paths'][0]))
    for attempt in range(2):
        raw=client.complete_text(prompt,image,max_tokens=500,temperature=0,
                                 response_schema=REVIEW_SCHEMA if attempt else None).raw_text
        try:
            result=parse_json(raw)
            if not isinstance(result.get('valid'),bool) or not isinstance(result.get('issues'),list) or len(result['issues'])>4:
                raise SkillError('Need boolean valid and at most four issues')
            for issue in result['issues']:
                if not isinstance(issue,dict) or any(not isinstance(issue.get(k),str) or not issue[k].strip() for k in ('requirement','problem','correction')):
                    raise SkillError('Each issue needs requirement, problem and correction')
                if issue['requirement'].lower() not in instruction.lower():
                    raise SkillError('Each requirement must quote the task exactly; do not invent a requirement')
            if result['valid']!= (not result['issues']):raise SkillError('valid must agree with issues')
            return result
        except (SkillError,ValueError,TypeError) as exc:
            if attempt:raise SkillError('Invalid semantic review: '+str(exc)) from exc
            prompt+='\nInvalid review: '+raw+'\nReason: '+str(exc)+'\nCorrect the review JSON.'
