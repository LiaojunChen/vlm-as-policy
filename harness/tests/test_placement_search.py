import math
import numpy as np
from transforms3d.euler import euler2mat
from robotwin_harness_v3 import placement_yaws, placement_rotations, grasp_quat
from robotwin_harness_v3 import prefer_side_grasp
from robotwin_harness_v3 import grasp_yaws


def test_tall_grasps_cover_all_horizontal_azimuths_before_tilting():
    yaws=grasp_yaws(1.01,True)
    assert yaws[:3]==[math.pi/2,math.pi/3,2*math.pi/3]
    assert any(abs(angle-11*math.pi/12)<1e-8 for angle in yaws)
    assert len(yaws)>=24
    assert len(grasp_yaws(1.01,False))==7


def test_observed_low_loose_handle_uses_top_grasp():
    action=dict(skill='pick',grasp_part='handle',approach='side')
    geometry=dict(tcp=[0,0,.787],tall=False)
    assert not prefer_side_grasp(action,geometry,.74)
    assert prefer_side_grasp(action,{**geometry,'tall':True},.74)
    assert prefer_side_grasp(action,{**geometry,'tcp':[0,0,.90]},.74)
    assert prefer_side_grasp({**action,'skill':'grasp_handle'},geometry,.74)


def test_free_placement_explores_more_than_antipodal_orientations_without_tilting():
    angles = placement_yaws({})
    assert angles[:2] == (0., math.pi)
    assert len({round(a % (2 * math.pi), 5) for a in angles}) == 12
    for angle in angles:
        np.testing.assert_allclose(euler2mat(0, 0, angle) @ [0, 0, 1], [0, 0, 1])


def test_alignment_constraint_is_not_sacrificed_for_reachability():
    for action in ({'align_yaw_deg': 45}, {'preserve_yaw': True}):
        assert placement_yaws(action) == (0., math.pi)


def test_release_search_can_relax_vertical_grasp_but_tool_contact_cannot():
    q=grasp_quat(math.pi/2)
    candidates=list(placement_rotations(q,{}))
    assert len(candidates)==36
    assert all(tilt==0 for _,tilt,_ in candidates[:12])
    assert any(tilt==-15 for _,tilt,_ in candidates)
    assert all(tilt==0 for _,tilt,_ in placement_rotations(q,{'release':False}))
