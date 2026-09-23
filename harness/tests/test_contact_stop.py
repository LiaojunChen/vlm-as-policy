import pytest
from contact_geometry import measured_bar_contact_stop


def inputs():
    return (dict(tcp=[0,0,.90],handle_contact_selection={'source':'observed'}),[.01,.01,.945],
            dict(contact_fingers=1),dict(name='contact_refine',position_error_m=.047,planner_status={'right':'Success'}),'right')


def test_measured_contact_stop_preserves_actual_pose_without_claiming_a_grasp():
    result=measured_bar_contact_stop(*inputs())
    assert result and result['measured_tcp']==[.01,.01,.945]
    assert 'holding' not in result and 'success' not in result


@pytest.mark.parametrize('point',[[.06,0,.945],[0,0,.89],[0,0,1.0],[float('nan'),0,.945]])
def test_lateral_overshoot_deep_contact_and_invalid_pose_do_not_close(point):
    args=list(inputs());args[1]=point
    assert measured_bar_contact_stop(*args) is None


def test_unobserved_component_empty_fingers_and_planning_failure_do_not_close():
    args=list(inputs());args[0]=dict(tcp=[0,0,.90])
    assert measured_bar_contact_stop(*args) is None
    args=list(inputs());args[2]=dict(contact_fingers=0)
    assert measured_bar_contact_stop(*args) is None
    args=list(inputs());args[3]=dict(name='contact',position_error_m=.04,planner_status={'right':'Fail'})
    assert measured_bar_contact_stop(*args) is None
