import json
import pytest
from runtime.json_prefix import JsonPrefix
from robodawn.schemas import MISSION_SCHEMA,ACTION_SCHEMA,GROUND_SCHEMA


@pytest.mark.parametrize('schema,value',[
    (MISSION_SCHEMA,{'steps':[{'operation':'press','source':'button','arm':'left'}]}),
    (MISSION_SCHEMA,{'steps':[{'operation':'tool_contact','source':'tool','contact_part':'tool tip','destination':'surface','relation':'on','arm':'left'}]}),
    (MISSION_SCHEMA,{'steps':[{'operation':'bimanual_lift','source':'pot','left_source':'left handle','right_source':'right handle'}]}),
    (MISSION_SCHEMA,{'steps':[{'operation':'arrange','sources':['small block','big block'],'arms':['left','right']}]}),
    (ACTION_SCHEMA,{'skill':'present','arm':'right','location':'centre'}),
    (ACTION_SCHEMA,{'skill':'move','arm':'left','delta':[0,-.05,.1]}),
    (GROUND_SCHEMA,{'bbox':[10,20,90,95],'point':[30,40],'approach':'top','support':'object'}),
    (GROUND_SCHEMA,{'visible':False}),
])
def test_valid_actions_and_every_partial_prefix_are_accepted(schema,value):
    text=json.dumps(value)
    for length in range(len(text)):
        assert JsonPrefix(schema).status(text[:length])!='invalid',text[:length]
    assert JsonPrefix(schema).status(text)=='complete'


@pytest.mark.parametrize('schema,text',[
    (MISSION_SCHEMA,'{"steps":["PRESS"]}'),
    (MISSION_SCHEMA,'{"steps":[{"operation":"press"}]}'),
    (ACTION_SCHEMA,'{"skill":"tap","arm":"left"}'),
    (ACTION_SCHEMA,'{"skill":"present","arm":"left","location":"front centre"}'),
    (GROUND_SCHEMA,'{"visible":false,"extra":3}'),
])
def test_observed_malformed_outputs_are_rejected(schema,text):
    assert JsonPrefix(schema).status(text)=='invalid'
