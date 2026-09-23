import numpy as np
from container_portal import below_table_portal
from robotwin_harness_v3 import Bridge
from types import SimpleNamespace


def scene():
    x,y=np.meshgrid(np.linspace(-.55,-.32,100),np.linspace(-.2,.1,120))
    return np.dstack([x,y,np.full_like(x,.3)]),np.ones(x.shape,bool)


def test_low_container_uses_clear_above_table_release_not_floor_contact():
    cloud,valid=scene();result=below_table_portal(cloud,valid,[0,0,1000,1000],.74,[.06,.06])
    assert result is not None
    assert result['tcp'][2]==.8 and result['release_portal']['current_column_top_m']==.3
    assert result['release_portal']['clear_radius_m']>=result['release_portal']['required_radius_m']


def test_no_unobserved_void_table_surface_or_oversized_payload_is_accepted():
    cloud,valid=scene()
    assert below_table_portal(cloud,np.zeros_like(valid),[0,0,1000,1000],.74,[.06,.06]) is None
    assert below_table_portal(cloud+[0,0,.44],valid,[0,0,1000,1000],.74,[.06,.06]) is None
    assert below_table_portal(cloud,valid,[0,0,1000,1000],.74,[.24,.24]) is None
    valid[:,45:55]=False
    assert below_table_portal(cloud,valid,[0,0,1000,1000],.74,[.09,.09]) is None


def test_high_obstacle_blocks_observed_column():
    cloud,valid=scene();cloud[:,35:65,2]=.85
    assert below_table_portal(cloud,valid,[0,0,1000,1000],.74,[.08,.08]) is None


def test_empty_place_is_rejected_before_any_grounding_or_motion():
    b=Bridge.__new__(Bridge);b.index=0;b.sensors=lambda:dict(left=dict(holding=False))
    b.env=SimpleNamespace(eval_success=False,check_success=lambda:False)
    def unexpected(action):raise AssertionError('Empty arm must not measure placement geometry')
    b.geometry=unexpected
    result=b.execute(dict(skill='place',arm='left',target='container',bbox=[0,0,1000,1000],point=[500,500]))
    assert result['failure']=='Place requires verified held object' and result['subactions']==[]
