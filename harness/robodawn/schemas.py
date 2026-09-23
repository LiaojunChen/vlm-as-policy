"""Executable-interface JSON schemas; no task identities or hidden state."""
def obj(properties,required=None):
    return dict(type='object',properties=properties,required=list(properties) if required is None else required,additionalProperties=False)

def enum(*values):return dict(type='string',enum=list(values))
def array(item,minimum,maximum):return dict(type='array',items=item,minItems=minimum,maxItems=maximum)
def number(lo,hi):return dict(type='number',minimum=lo,maximum=hi)

TEXT=dict(type='string',minLength=1,maxLength=180)
ARM=enum('left','right');AUTO_ARM=enum('left','right','auto')
PART=enum('body','handle','rim','edge');RELATION=enum('on','inside','left_of','right_of','in_front_of','behind')
SUPPORT=enum('table','object','container');AXIS=enum('x','y','z')
FACING=enum('left','right','front','back')
ORIENTATION=obj(dict(from_part=TEXT,to_part=TEXT,facing=FACING))
VECTOR=array(number(-.2,.2),3,3)
BOX=array(number(0,1000),4,4);POINT=array(number(0,1000),2,2)

actions=[]
for skill in ('pick','grasp_handle','reach','press','place'):
    props=dict(skill=enum(skill),arm=ARM,target=TEXT,reason=TEXT)
    required=['skill','arm','target']
    if skill in ('pick','grasp_handle'):props.update(grasp_part=PART,approach=enum('top','side'))
    if skill=='place':props.update(support=SUPPORT,release=dict(type='boolean'),facing=FACING)
    actions.append(obj(props,required))
for skill in ('open','close','home'):
    actions.append(obj(dict(skill=enum(skill),arm=ARM,reason=TEXT),['skill','arm']))
actions += [obj(dict(skill=enum('present'),arm=ARM,location=enum('centre','side'),reason=TEXT),['skill','arm','location']),
            obj(dict(skill=enum('handover'),arm=ARM,donor=ARM,target=TEXT,grasp_part=PART,approach=enum('top','side'),reason=TEXT),['skill','arm','donor','target']),
            obj(dict(skill=enum('move'),arm=ARM,delta=VECTOR,reason=TEXT),['skill','arm','delta']),
            obj(dict(skill=enum('dual_move'),delta=VECTOR,reason=TEXT),['skill','delta']),
            obj(dict(skill=enum('rotate'),arm=ARM,axis=AXIS,angle=number(-90,90),reason=TEXT),['skill','arm','axis','angle']),
            obj(dict(skill=enum('arc'),arm=ARM,target=TEXT,axis=AXIS,angle=number(-90,90),reason=TEXT),['skill','arm','target','axis','angle']),
            obj(dict(skill=enum('shake'),arm=ARM,axis=AXIS,amplitude=number(.02,.06),cycles=dict(type='integer',minimum=1,maximum=4),reason=TEXT),['skill','arm','axis']),
            obj(dict(skill=enum('push'),arm=ARM,target=TEXT,destination=obj(dict(target=TEXT,support=SUPPORT)),reason=TEXT),['skill','arm','target','destination']),
            obj(dict(skill=enum('wait'),reason=TEXT),['skill'])]
ACTION_SCHEMA=dict(anyOf=actions)

steps=[]
for operation in ('transfer','slide','tool_contact'):
    props=dict(operation=enum(operation),source=TEXT,destination=TEXT,relation=RELATION,arm=AUTO_ARM,grasp_part=PART)
    required=['operation','source','destination','relation','arm']
    if operation=='transfer':props['orientation']=ORIENTATION
    if operation=='tool_contact':
        props['contact_part']=TEXT;required.append('contact_part')
    steps.append(obj(props,required))
steps += [obj(dict(operation=enum('lift'),source=TEXT,arm=AUTO_ARM,location=enum('stay','centre','side'),grasp_part=PART),['operation','source','arm','location']),
          obj(dict(operation=enum('handover'),source=TEXT,arm=ARM,receiver=ARM,grasp_part=PART),['operation','source','arm','receiver']),
          obj(dict(operation=enum('press'),source=TEXT,arm=AUTO_ARM)),
          obj(dict(operation=enum('bimanual_lift'),source=TEXT,left_source=TEXT,right_source=TEXT,grasp_part=PART),['operation','source','left_source','right_source']),
          obj(dict(operation=enum('arrange','stack'),sources=array(TEXT,2,5),arms=array(AUTO_ARM,2,5),grasp_part=PART),['operation','sources','arms']),
          obj(dict(operation=enum('action'),action=ACTION_SCHEMA))]
MISSION_SCHEMA=obj(dict(steps=array(dict(anyOf=steps),1,12)))
GROUND_SCHEMA=dict(anyOf=[obj(dict(visible=dict(type='boolean',enum=[False]))),
    obj(dict(bbox=BOX,point=POINT,approach=enum('top','side'),support=SUPPORT,visible=dict(type='boolean',enum=[True])),['bbox','point','approach','support'])])
HINGE_GROUND_SCHEMA=dict(anyOf=[obj(dict(visible=dict(type='boolean',enum=[False]))),
    obj(dict(bbox=BOX,point=POINT,hinge_line=array(POINT,2,2),approach=enum('top','side'),support=SUPPORT,
             visible=dict(type='boolean',enum=[True])),['bbox','point','hinge_line','approach','support'])])
