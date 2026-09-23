import numpy as np
from robotwin_harness_v3 import estimate_table_height


def test_low_table_is_not_confused_with_object_tops():
    cloud=np.zeros((100,100,3));cloud[:,:,2]=.64
    cloud[20:45,20:45,2]=.685
    cloud[50:70,50:70,2]=.72
    valid=np.ones((100,100),bool)
    assert abs(estimate_table_height(cloud,valid)-.64)<1e-6
    cloud[:,:,2]+=.1
    assert abs(estimate_table_height(cloud,valid)-.74)<1e-6


def test_missing_depth_retains_last_measured_table():
    cloud=np.zeros((5,5,3))
    assert estimate_table_height(cloud,np.zeros((5,5),bool),.64)==.64
