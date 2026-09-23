import numpy as np
from robot_table_clearance import table_clear_grasp
from robotwin_harness_v3 import grasp_quat,ee_from_tcp
from transforms3d.quaternions import quat2mat


def fingers():
    # Robot EE local X points toward the finger tips, beyond the nominal TCP.
    return np.array([[.155,-.04,-.01],[.155,.04,.01],[.08,-.04,.01],[.08,.04,-.01]])


def test_low_top_grasp_accounts_for_real_finger_tips_not_nominal_tcp():
    q=grasp_quat(.3);tcp=np.array([.1,-.2,.758])
    adjusted,evidence=table_clear_grasp(tcp,q,fingers(),.74)
    np.testing.assert_allclose(adjusted[:2],tcp[:2])
    assert .016<evidence['raise_m']<.021
    pose=ee_from_tcp(adjusted,q);world=fingers()@quat2mat(q).T+pose[:3]
    assert abs(world[:,2].min()-.742)<1e-9
    np.testing.assert_allclose(tcp,[.1,-.2,.758])


def test_already_clear_contact_is_unchanged():
    tcp=[.1,-.2,.9]
    adjusted,evidence=table_clear_grasp(tcp,grasp_quat(0),fingers(),.74)
    np.testing.assert_allclose(adjusted,tcp)
    assert evidence['raise_m']==0


def test_large_corrections_and_bad_robot_geometry_abstain():
    assert table_clear_grasp([0,0,.70],grasp_quat(0),fingers(),.74) is None
    assert table_clear_grasp([0,0,.8],grasp_quat(0),[],.74) is None
    assert table_clear_grasp([0,0,.8],grasp_quat(0),[[float('nan'),0,0]],.74) is None


def test_closed_push_contact_uses_the_same_verified_clearance_rule():
    q=grasp_quat(0)
    raw=np.array([.1,-.2,.758])
    adjusted,evidence=table_clear_grasp(raw,q,fingers(),.74)
    assert evidence['source']=='current_robot_finger_geometry_table_clearance'
    assert adjusted[2]>raw[2]
    # Horizontal push geometry is invariant under the correction.
    np.testing.assert_allclose(adjusted[:2],raw[:2])
