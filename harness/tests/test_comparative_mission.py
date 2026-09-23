import pytest
from robotwin_harness_v3 import SkillError
from robodawn.mission_planner import validate_explicit_comparative_order


def arrange(names):
    return [dict(operation='transfer',source=name,relation='row_slot') for name in names]


def test_visual_names_cannot_drop_an_explicit_comparative_identity():
    instruction='Arrange large block, medium block, small block in order'
    with pytest.raises(SkillError,match='EACH requested size identity'):
        validate_explicit_comparative_order(arrange(['large green block','small purple block left','small purple block right']),instruction)
    validate_explicit_comparative_order(arrange(['large green block','medium purple block','small purple block']),instruction)


def test_comparative_check_does_not_impose_a_task_specific_direction_or_recovery_order():
    validate_explicit_comparative_order(arrange(['small cube','medium cube','largest cube']),
                                       'Arrange smallest cube, medium cube, biggest cube')
    validate_explicit_comparative_order(arrange(['red block','green block','blue block']),'Arrange red green blue blocks')
    validate_explicit_comparative_order(arrange(['small cube','large cube']),
                                       'Arrange large cube and small cube',complete_plan=False)
