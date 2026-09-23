from itertools import product
import numpy as np
import pytest
from placement_hand_clearance import HandPart,PlacementHandClearance
from robotwin_harness_v3 import SkillError


def screen(points):
    value=PlacementHandClearance.__new__(PlacementHandClearance)
    value.parts=[HandPart('finger',np.array(list(product((-1,1),repeat=3)))*.01)]
    value.points=np.asarray(points,float);value.calibration_error_m=0.
    return value


def test_local_descent_screens_intermediate_collision_not_only_endpoints():
    check=screen([[0,0,.04]])
    result=check.screen([0,0,.08,1,0,0,0],[0,0,0,1,0,0,0])
    assert not result['accepted'] and result['max_penetration_m']==pytest.approx(.01)
    assert result['link']=='finger'


def test_clear_descent_and_tolerable_surface_noise_pass_without_moving_points():
    points=np.array([[.025,0,.04],[.009,0,.04]])
    check=screen(points);before=points.copy()
    result=check.screen([0,0,.08,1,0,0,0],[0,0,0,1,0,0,0])
    assert result['accepted'] and result['max_penetration_m']==pytest.approx(.001)
    np.testing.assert_array_equal(points,before)


def test_aperture_parts_participate_in_collision_screen():
    check=screen([[.04,0,0]])
    vertices=np.array(list(product((-1,1),repeat=3)))*.01+[.04,0,0]
    check.parts.append(HandPart('open_finger',vertices))
    result=check.screen([0,0,.08,1,0,0,0],[0,0,0,1,0,0,0])
    assert not result['accepted'] and result['link']=='open_finger'


def test_invalid_rotating_descent_is_not_certified():
    with pytest.raises(SkillError,match='constant'):
        screen([[0,0,0]]).screen([0,0,.08,1,0,0,0],[0,0,0,0,1,0,0])
