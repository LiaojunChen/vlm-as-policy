import numpy as np
from transforms3d.euler import euler2mat
from visual_registration_v3 import register_grasp

def test_rigid_tracking_recovers_grasp_slip_despite_gray_finger_distractor():
    rng=np.random.default_rng(3)
    source=rng.normal(size=(900,3))*[.025,.008,.008]+[.1,-.1,.76]
    rgb=np.tile([20,50,210],(900,1))+rng.integers(-10,10,(900,3))
    rotation=euler2mat(.10,-.08,.18);centre=source.mean(0)
    shift=np.array([.01,.01,.12]);translation=centre+shift-rotation@centre
    target=source@rotation.T+translation
    clutter=rng.normal(size=(400,3))*[.02,.02,.02]+target.mean(0)
    fit=register_grasp(source,rgb,np.r_[target,clutter],np.r_[rgb,np.tile([170,170,170],(400,1))],shift)
    assert fit is not None
    np.testing.assert_allclose(fit['rotation'],rotation,atol=.025)
    np.testing.assert_allclose(source@fit['rotation'].T+fit['translation'],target,atol=.002)

def test_rigid_tracking_rejects_wrong_colour_robot_surface():
    rng=np.random.default_rng(1);source=rng.normal(size=(100,3))*.01
    fit=register_grasp(source,np.tile([0,0,255],(100,1)),source+[0,0,.1],np.tile([160,160,160],(100,1)),[0,0,.1])
    assert fit is None
