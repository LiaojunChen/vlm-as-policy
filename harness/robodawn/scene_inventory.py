"""Perception-first memory for collective goals; only existing current RGB."""
import json
import re
import numpy as np
from PIL import Image
from robotwin_harness_v3 import SkillError,parse_json
from robodawn.schemas import obj,array,TEXT,BOX,POINT,PART
from robodawn.episodic_memory import object_key,same_visual_region

SCHEMA=obj(dict(objects=array(obj(dict(description=TEXT,bbox=BOX,point=POINT,grasp_part=PART)),1,12)))


def needs_inventory(instruction):
    text=instruction.split('Completion requirements:')[0]
    # Explicit comparative identities already enumerate these objects. Their
    # metric-size grounding is done from RGB-D; an RGB-only inventory must not
    # replace "medium" with a second "small" based on projected image size.
    sizes=set(re.findall(r'\b(?:small(?:est)?|medium|large(?:st)?|big(?:gest)?)\b',text,re.I))
    if len(sizes)>=2 and re.search(r'\b(arrange|sort|rank)\b',text,re.I):return False
    return bool(re.search(r'\b(stack(?:ing)?|sort(?:ing)?|rank(?:ing)?|arrang(?:e|ing)|simultaneously)\b|\b(?:at the same time|in parallel)\b|\b(each|every|all)\b(?!\s+(?:arm|hand|gripper)s?\b)',text,re.I))


def validate_inventory(value):
    entries=value.get('objects')
    if not isinstance(entries,list) or not 1<=len(entries)<=12:raise SkillError('Inventory requires 1..12 visible objects')
    checked=[];names=set()
    for item in entries:
        if not isinstance(item,dict) or not isinstance(item.get('description'),str) or not item['description'].strip():
            raise SkillError('Each object needs a distinct visible description')
        key=object_key(item['description'])
        if key in names:raise SkillError('Distinguish identical-looking objects by their initial visible positions')
        box=np.asarray(item.get('bbox'),float);point=np.asarray(item.get('point'),float)
        if box.shape!=(4,) or point.shape!=(2,) or not np.isfinite(box).all() or not np.isfinite(point).all():raise SkillError('Invalid inventory coordinates')
        if np.any(box<0) or np.any(box>1000) or np.any(box[2:]<=box[:2]) or np.any(point<box[:2]) or np.any(point>box[2:]):raise SkillError('Invalid inventory box')
        if item.get('grasp_part') not in ('body','rim','handle','edge'):raise SkillError('Invalid inventory graspable part')
        entry=dict(target=item['description'],bbox=box.tolist(),point=point.tolist(),
                   preferred_grasp_part=item['grasp_part'],inventory_identity=True)
        if any(same_visual_region(entry,old) for old in checked):raise SkillError('Do not count the same visible object twice')
        checked.append(entry);names.add(key)
    return checked


def observe_inventory(client,obs,instruction,memory):
    prompt=('Observe this current image and list each separate task-relevant physical object once. '
            'Do not combine identical-looking objects; distinguish them by their INITIAL left/right/front/back position and visible colour. '
            'Include objects that will be moved AND their destination supports. Ignore robot arms, shadows and background. '
            'Do not plan actions and do not invent hidden objects. TASK: '+instruction.split('Completion requirements:')[0]+'\n'
            'Return JSON {"objects":[{"description":"unique visible object description","bbox":[xmin,ymin,xmax,ymax],'
            '"point":[x,y],"grasp_part":"body" or "rim" or "handle" or "edge"}]}. '
            'Coordinates are normalized 0..1000 in this image; boxes cover the entire visible object.')
    image=np.asarray(Image.open(obs['image_paths'][0]))
    for attempt in range(2):
        raw='[No complete model response received]'
        try:
            raw=client.complete_text(prompt,image,max_tokens=500,temperature=0,response_schema=SCHEMA if attempt else None).raw_text
            entries=validate_inventory(parse_json(raw));break
        except (SkillError,ValueError,TypeError) as exc:
            if attempt:raise SkillError('Scene inventory failed after repair: '+str(exc)) from exc
            prompt+='\nRejected inventory: '+raw+'\nReason: '+str(exc)+'\nCorrect the current-image object list.'
    for entry in entries:memory.remember_visual(entry,obs['observation_id'])
    return entries
