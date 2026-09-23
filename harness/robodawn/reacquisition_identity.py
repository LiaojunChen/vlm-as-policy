"""Reacquisition checks against an earlier verified visual identity in this episode.

The historical box is a visual reference only. It never supplies current
coordinates, contact geometry, or a simulator identity.
"""
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from robotwin_harness_v3 import SkillError, parse_json
from robodawn.episodic_memory import object_key

SCHEMA=dict(type='object',properties={
    'reference_description':dict(type='string'),
    'candidate_description':dict(type='string'),
    'identity_match':dict(type='string',enum=['same','different','uncertain'])},
    required=['reference_description','candidate_description','identity_match'],additionalProperties=False)


def temporal_crops(reference_path,reference_box,current_path,current_box):
    """Actual magnification of each observed crop; no other scene objects.

    This tests appearance compatibility, not an oracle of physical instance
    identity. Identical-looking instances cannot be distinguished by it.
    """
    canvas=Image.new('RGB',(640,320),'white');draw=ImageDraw.Draw(canvas)
    for i,(path,box) in enumerate(((reference_path,reference_box),(current_path,current_box))):
        frame=Image.open(path).convert('RGB');width,height=frame.size
        bounds=np.rint(np.asarray(box)*[width-1,height-1,width-1,height-1]/1000).astype(int)
        if bounds.shape!=(4,) or np.any(bounds[2:]<=bounds[:2]):raise SkillError('Invalid visual reference crop')
        crop=frame.crop(tuple(bounds))
        crop=ImageOps.contain(crop,(288,264),method=Image.Resampling.BICUBIC)
        canvas.paste(crop,(i*320+(320-crop.width)//2,40+(264-crop.height)//2))
        draw.text((i*320+12,12),'A: EARLIER REFERENCE' if i==0 else 'B: CURRENT CANDIDATE',fill='black')
    return np.asarray(canvas)


def reference_for_reacquisition(obs,memory,action):
    if (memory is None or action.get('skill')!='pick' or action.get('grasp_part','body')!='body'
            or action.get('grounding_role') or action.get('camera','head_camera')!='head_camera'):
        return None
    key=object_key(action.get('target',''));entry=memory.objects.get(key,{})
    visual=entry.get('last_visual',{});reference=visual.get('observation_id');current=obs.get('observation_id')
    if (entry.get('state')!='released_or_lost_reobserve' or entry.get('held_by')
            or visual.get('component_only',True) or visual.get('camera','head_camera')!='head_camera'
            or not isinstance(reference,int) or not isinstance(current,int) or reference>=current
            or entry.get('identity_verification_observation_id')!=reference):
        return None
    if not any(e.get('skill')=='pick' and e.get('skill_success')
               and object_key(e.get('target',''))==key for e in memory.events):
        return None
    current_path=Path(obs['image_paths'][0])
    if current_path.name!=f'{current:04d}_head_camera.png':return None
    reference_path=current_path.parent/f'{reference:04d}_head_camera.png'
    if not reference_path.is_file():return None
    return dict(image_path=str(reference_path),visual=deepcopy(visual),
                observation_id=reference,current_image_path=str(current_path))


def verify_reacquired_identity(client,obs,memory,action,current_box):
    reference=reference_for_reacquisition(obs,memory,action)
    if reference is None:return None
    image=temporal_crops(reference['image_path'],reference['visual']['bbox'],reference['current_image_path'],current_box)
    prompt=('Compare the TWO enlarged object crops independently using their actual pixels. '
            'A on the LEFT is an earlier visual memory reference. B on the RIGHT is a current candidate and MAY BE A DIFFERENT OBJECT. '
            'First describe the visible color, shape and category in EACH crop without copying one description to the other. '
            'Then judge whether B is visually compatible with A after possible translation or rotation. '
            'Different visible categories or incompatible colors/shapes mean different; ambiguous/occluded evidence means uncertain. '
            'Same only means compatible visible appearance, not proof of identity for identical-looking instances. '
            'Return JSON {"reference_description":string,"candidate_description":string,'
            '"identity_match":"same" or "different" or "uncertain"}. Do not return coordinates.')
    for attempt in range(2):
        raw=client.complete_text(prompt,image,max_tokens=220,temperature=0,
                                 response_schema=SCHEMA if attempt else None).raw_text
        try:
            value=parse_json(raw)
            if value.get('identity_match') not in ('same','different','uncertain') or not all(
                    isinstance(value.get(k),str) and value[k].strip() for k in ('reference_description','candidate_description')):
                raise SkillError('Describe both observed crops and return same/different/uncertain')
            break
        except (SkillError,ValueError,TypeError) as exc:
            if attempt:raise SkillError('Invalid visual identity comparison: '+str(exc)) from exc
            prompt+='\nRejected comparison: '+raw+'\nReason: '+str(exc)+'\nCorrect only the comparison JSON.'
    if value['identity_match']!='same':
        raise SkillError('Reacquisition identity '+value['identity_match']+': remembered object is '+value['reference_description']+
                         '; current proposed object is '+value['candidate_description']+'. '
                         'Re-localize the SAME remembered target in the current image; do not grab this different/uncertain candidate. '
                         'Return visible=false if the remembered object is absent from this crop.')
    return dict(source='same_model_temporal_visual_identity_comparison',reference_observation_id=reference['observation_id'],
                current_observation_id=obs['observation_id'],**value)
