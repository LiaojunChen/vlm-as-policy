"""Verify semantic object bindings using two crops of the existing RGB frame."""
from copy import deepcopy
import json
import numpy as np
from PIL import Image,ImageDraw
from robotwin_harness_v3 import SkillError,parse_json

SCHEMA=dict(type='object',properties={
    'description_A':dict(type='string'),'description_B':dict(type='string'),
    'source':dict(type='string',enum=['A','B','unknown']),
    'destination':dict(type='string',enum=['A','B','unknown']),
    'source_grasp_part':dict(type='string',enum=['body','rim','handle','edge'])},
    required=['description_A','description_B','source','destination','source_grasp_part'],additionalProperties=False)


def paired_image(image_path, first, second, second_image_path=None):
    original=Image.open(image_path).convert('RGB');width,height=original.size
    if second_image_path is not None and second_image_path!=image_path:
        # Each candidate keeps its own full-view context and pixel calibration.
        # Equal pixel coordinates in different cameras do not imply identity.
        canvas=Image.new('RGB',(640,420),'white')
        for i,(path,action) in enumerate(((image_path,first),(second_image_path,second))):
            frame=Image.open(path).convert('RGB');w,h=frame.size
            panel=Image.new('RGB',(320,420),'white');panel.paste(frame.resize((320,240)),(0,0))
            draw=ImageDraw.Draw(panel);label='A' if i==0 else 'B'
            box=tuple(np.rint(np.asarray(action['bbox'])*[319,239,319,239]/1000).astype(int))
            draw.rectangle(box,outline='yellow',width=1)
            draw.text((5,245),label+' / '+action.get('camera','head_camera'),fill='black')
            crop=frame.crop(tuple(np.rint(np.asarray(action['bbox'])*[w-1,h-1,w-1,h-1]/1000).astype(int)))
            crop.thumbnail((300,150));panel.paste(crop,((320-crop.width)//2,265))
            canvas.paste(panel,(i*320,0))
        return np.asarray(canvas)
    # Crops alone erase the spatial evidence needed to distinguish identical
    # instances. Preserve the SAME frame above them, with candidate labels.
    canvas=Image.new('RGB',(320,420),'white');draw=ImageDraw.Draw(canvas)
    canvas.paste(original.resize((320,240)),(0,0))
    for i,action in enumerate((first,second)):
        label='A' if i==0 else 'B'
        overview_box=np.asarray(action['bbox'])*np.array([319,239,319,239])/1000
        overview_box=tuple(np.rint(overview_box).astype(int))
        draw.rectangle(overview_box,outline='yellow',width=1)
        draw.text((overview_box[0]+2,overview_box[1]+2),label,fill='yellow',stroke_width=1,stroke_fill='black')
        box=np.asarray(action['bbox'])*np.array([width-1,height-1,width-1,height-1])/1000
        crop=original.crop(tuple(np.rint(box).astype(int)));crop.thumbnail((150,150))
        canvas.paste(crop,(i*160+(160-crop.width)//2,265+(150-crop.height)//2))
        draw.text((i*160+5,245),label,fill='black')
    return np.asarray(canvas)


def verify_pair(client, obs, source, destination):
    from robodawn.camera_views import camera_of,in_camera
    first_view=in_camera(obs,camera_of(source));second_view=in_camera(obs,camera_of(destination))
    mixed=camera_of(source)!=camera_of(destination)
    image=paired_image(first_view['image_paths'][0],source,destination,second_view['image_paths'][0])
    prompt=('The TOP panel is the full current scene with candidate boxes labeled A and B. '
            'The BOTTOM panels are enlarged crops of those SAME two candidates. '
            'For left/right/top/bottom identities use positions in the TOP full scene, NEVER the left/right order of the crop panels. '
            'Bind these requested identities: SOURCE='+source['target']+'; DESTINATION SUPPORT='+destination['target']+'. '
            'Describe the actual visible shape/category of each crop, then select which crop matches each requested identity. '
            'Do not assume A is the source. Do not rewrite the requested source/destination names. '
            'Also choose a graspable part of the SOURCE: body, rim, handle or edge. A vessel opening is empty, not a solid grasp surface. '
            'If either requested identity is absent or ambiguous, return unknown for that role. '
            'Return JSON {"description_A":string,"description_B":string,"source":"A" or "B" or "unknown",'
            '"destination":"A" or "B" or "unknown","source_grasp_part":"body" or "rim" or "handle" or "edge"}.')
    if mixed:
        prompt=('These are two DIFFERENT calibrated camera views captured in the SAME current observation. '
                'Each column has its own full scene above its candidate crop, tagged A or B and camera name. '
                'Panel order and wrist image left/right DO NOT specify world side or original object identity. '
                'If you cannot distinguish the requested objects by visible appearance, return unknown.\n'+prompt)
    for attempt in range(2):
        raw='[No complete model response received]'
        try:
            raw=client.complete_text(prompt,image,max_tokens=240,temperature=0,response_schema=SCHEMA if attempt else None).raw_text
            value=parse_json(raw)
            if value.get('source')=='unknown' or value.get('destination')=='unknown':
                raise SkillError('Requested object identity remains ambiguous in the paired current-image crops')
            if {value.get('source'),value.get('destination')}!={'A','B'}:
                raise SkillError('Source and destination must select different crops A and B')
            if not all(isinstance(value.get('description_'+key),str) and value['description_'+key].strip() for key in ('A','B')):
                raise SkillError('Describe both actually observed objects')
            if value.get('source_grasp_part') not in ('body','rim','handle','edge'):
                raise SkillError('Choose a valid source graspable part')
            break
        except (SkillError,ValueError,TypeError) as exc:
            if attempt:raise SkillError('Paired identity verification failed: '+str(exc)) from exc
            prompt+='\nRejected: '+raw+'\nReason: '+str(exc)+'\nCorrect this same-image identity verification JSON.'
    candidates={'A':source,'B':destination};bindings=[]
    for role,requested in (('source',source),('destination',destination)):
        key=value[role];binding=deepcopy(candidates[key]);binding['target']=requested['target']
        binding['verified_appearance']=value['description_'+key]
        binding['identity_verification_observation_id']=obs['observation_id']
        binding['grounding_source']='paired_current_image_crops_with_scene_context'
        if role=='source':binding['preferred_grasp_part']=value['source_grasp_part']
        bindings.append(binding)
    return bindings
