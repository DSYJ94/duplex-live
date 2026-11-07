from data.venus.retention_semantics import calibration,calibration_passed,verdict


def test_always_accepting_or_invalid_reviewer_cannot_approve_independent_data_review():
    controls=calibration()
    assert not calibration_passed([dict(r,parsed={'passed':True,'reason':'Always yes'}) for r in controls])
    assert not calibration_passed([dict(r,parsed=None) for r in controls])
    assert calibration_passed([dict(r,parsed={'passed':r['expected'],'reason':'Fixture correct'}) for r in controls])
    assert all(set(r['payload'])=={'user','actions','actual_feedback'} for r in controls)


def test_review_parser_never_coerces_a_string_or_discards_unexplained_response_text():
    assert verdict('{"passed":true,"reason":"Correct"}')['passed'] is True
    assert verdict('{"passed":"true","reason":"Correct"}') is None
    assert verdict('Probably fine {"passed":true,"reason":"Correct"}') is None
    assert verdict('{"passed":true,"reason":"Correct","extra":"ignored"}') is None
