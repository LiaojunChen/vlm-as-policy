import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robodawn.episodic_memory import EpisodicMemory
from robodawn.semantic_planner import ground
from robotwin_harness_v3 import SkillError


def scene(tmp_path):
    paths = []
    for camera in ('head_camera', 'right_camera'):
        path = tmp_path / f'0001_{camera}.png'
        Image.new('RGB', (101, 101), 'white').save(path)
        paths.append(str(path))
        yy, xx = np.indices((101, 101))
        cloud = np.dstack([xx*.002, yy*.002, np.full((101, 101), .8 if camera=='head_camera' else .74)])
        suffix = 'rgbd' if camera=='head_camera' else camera+'_rgbd'
        np.savez_compressed(tmp_path / f'0001_{suffix}.npz', world_xyz=cloud,
                            valid=np.ones((101, 101), bool), robot_self_mask=np.zeros((101, 101), bool))
    return dict(image_paths=paths, camera_paths=dict(zip(('head_camera', 'right_camera'), paths)),
                observation_id=1, table_height_m=.74)


def action():
    return dict(skill='arc', arm='left', target='lid hinge', axis='x', angle=30)


def reply():
    return dict(bbox=[100, 350, 900, 650], point=[500, 500],
                hinge_line=[[200, 500], [800, 500]], approach='top', support='object')


def client(value):
    return SimpleNamespace(complete_text=lambda *a, **kw: SimpleNamespace(raw_text=json.dumps(value)))


def test_hinge_line_reaches_action_with_current_depth_and_original_angle(tmp_path):
    obs = scene(tmp_path)
    result = ground(client(reply()), obs, action())
    assert result['hinge_line']==reply()['hinge_line'] and result['angle']==30
    assert result['contact_view_evidence']['source']=='current_model_line_depth_admission'
    assert result['contact_view_evidence']['observation_id']==1


def test_reversed_pixel_order_does_not_reverse_signed_motion(tmp_path):
    obs = scene(tmp_path); value = reply(); value['hinge_line'].reverse()
    result = ground(client(value), obs, action())
    assert result['hinge_line']==reply()['hinge_line'] and result['angle']==30


def test_table_hinge_rejected_before_memory_commit_and_camera_fallback(tmp_path):
    obs = scene(tmp_path); obs['inspection_cameras']=['right_camera']
    memory = EpisodicMemory(); calls = []
    def complete(prompt, image, **kw):
        assert not memory.objects
        calls.append(prompt)
        return SimpleNamespace(raw_text=json.dumps(reply()))
    result = ground(SimpleNamespace(complete_text=complete), obs, action(), memory=memory)
    assert len(calls)==2 and result['camera']=='head_camera'
    assert 'right_camera' in result['rejected_grounding_views'][0]


def test_a_single_point_can_never_silently_fall_back_to_world_axis(tmp_path):
    obs = scene(tmp_path); value = reply(); value.pop('hinge_line')
    memory = EpisodicMemory()
    with pytest.raises(SkillError, match='two distinct'):
        ground(client(value), obs, action(), memory=memory)
    assert not memory.objects


def test_crop_mapping_applies_to_both_hinge_endpoints(tmp_path):
    obs = scene(tmp_path)
    result = ground(client(reply()), obs, action(), image_region=[250, 250, 750, 750])
    lo_x, lo_y, hi_x, hi_y = result['grounding_crop_bbox_px']
    expected=(np.array(reply()['hinge_line'])*np.array([hi_x-lo_x, hi_y-lo_y])/1000+[lo_x,lo_y])*10
    np.testing.assert_allclose(result['hinge_line'], expected)


def test_perpendicular_measured_axis_cannot_choose_rotation_sign(tmp_path):
    obs = scene(tmp_path)
    with pytest.raises(SkillError, match='inconsistent'):
        ground(client(reply()), obs, dict(action(), axis='y'))
