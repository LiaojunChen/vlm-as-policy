"""Opposing-finger contact checks, using anonymous tactile normals only."""


def opposing_bodies(directions, threshold=.25):
    """Return anonymous bodies pinched along the calibrated closing axis.

    Both fingers merely pressing down on the same object is not a pinch.
    Normals are oriented consistently as forces on each contacting finger.
    """
    fingers=list(directions.values())
    if len(fingers)!=2:return set()
    shared=set(fingers[0])&set(fingers[1]);result=set()
    for body in shared:
        left,right=fingers[0][body],fingers[1][body]
        if not left or not right:continue
        if (max(left)>threshold and min(right)<-threshold) or (min(left)<-threshold and max(right)>threshold):
            result.add(body)
    return result
