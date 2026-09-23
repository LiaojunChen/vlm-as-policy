import numpy as np
import pytest

from articulation_geometry import planar_joint_search, measured_line_evidence
from robotwin_harness_v3 import SkillError


def scene():
    x, y = np.meshgrid(np.linspace(-.1, .1, 101), np.linspace(-.08, .08, 81))
    z = .78+np.maximum(0, y)*.6
    return np.stack([x, y, z], axis=-1), np.ones(x.shape, bool)


def test_two_current_faces_produce_search_hint_not_executable_pose():
    cloud, valid = scene()
    hint = planar_joint_search(cloud, valid, [0, 0, 1000, 1000], .74)
    assert hint and hint['executable'] is False
    assert hint['observed_seam_neighbour_pixels'] >= 20
    assert abs(hint['hypothesis_axis'][0]) > .99
    assert abs(hint['hypothesis_origin'][1]) < .004
    assert 'tcp' not in hint and 'hinge_axis' not in hint
    assert hint['bbox'][1] < 500 < hint['bbox'][3]


def test_single_plane_and_background_cannot_invent_joint():
    cloud, valid = scene()
    cloud[:, :, 2] = .8
    assert planar_joint_search(cloud, valid, [0, 0, 1000, 1000], .74) is None
    cloud[:, :, 2] = .74
    assert planar_joint_search(cloud, valid, [0, 0, 1000, 1000], .74) is None


def test_named_current_box_and_validity_are_respected():
    cloud, valid = scene()
    assert planar_joint_search(cloud, valid, [0, 0, 1000, 400], .74) is None
    assert planar_joint_search(cloud, np.zeros_like(valid), [0, 0, 1000, 1000], .74) is None
    assert planar_joint_search(cloud, valid, [0, 0, float('nan'), 1000], .74) is None


def test_observed_line_passes_but_robot_and_table_endpoints_fail():
    cloud, valid = scene()
    line = [[100, 500], [900, 500]]
    result = measured_line_evidence(cloud, valid, line, .74)
    assert result['current_surface_fraction'] == 1
    assert result['axis'][0] == pytest.approx(1)
    robot = np.zeros_like(valid); robot[40, 10] = True
    with pytest.raises(SkillError, match='nonrobot'):
        measured_line_evidence(cloud, valid, line, .74, robot)
    cloud[40, 10, 2] = .74
    with pytest.raises(SkillError, match='above-table'):
        measured_line_evidence(cloud, valid, line, .74)


def test_disconnected_endpoints_or_depth_jump_cannot_define_line():
    cloud, valid = scene()
    cloud[40, 25:70, 2] += .04
    with pytest.raises(SkillError, match='continuous'):
        measured_line_evidence(cloud, valid, [[100, 500], [900, 500]], .74)
    with pytest.raises(SkillError, match='span'):
        measured_line_evidence(cloud, valid, [[100, 500], [101, 500]], .74)
