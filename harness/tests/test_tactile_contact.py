from tactile_contact import opposing_bodies


def test_touching_with_both_fingertips_is_not_a_pinch():
    assert not opposing_bodies({'a':{7:[.03]},'b':{7:[-.04]}})
    assert not opposing_bodies({'a':{7:[.9]},'b':{7:[.8]}})


def test_pinch_requires_opposing_normals_on_the_same_anonymous_body():
    assert opposing_bodies({'a':{7:[.7,0]},'b':{7:[-.8]}})=={7}
    assert opposing_bodies({'a':{7:[-.7]},'b':{7:[.8]}})=={7}
    assert not opposing_bodies({'a':{7:[.7]},'b':{8:[-.8]}})
