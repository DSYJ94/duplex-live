from data.venus.retention_review_lineage import absolute_clock
import copy


def test_clock_translation_preserves_relative_measurements_words_and_input():
    row=dict(id='case',payload=dict(timing_evidence_protocol='Protocol.',actions=[dict(kind='speak',frame=5,text='path.',native_speech_units=[dict(offset=0,text='path.',final=False),dict(offset=1,text='',final=True)],measured_word_intervals=[dict(text='path.',start=.7,end=1.12)])]))
    before=copy.deepcopy(row);new=absolute_clock(row);a=new['payload']['actions'][0];w=a['measured_word_intervals'][0]
    assert row==before
    assert w['start']==5.7 and w['end']==6.12
    assert w['source_relative_start']==.7 and w['source_relative_end']==1.12
    assert a['native_speech_units']==row['payload']['actions'][0]['native_speech_units']
    assert a['text']==row['payload']['actions'][0]['text']


def test_whole_original_review_is_required_and_only_failed_payload_is_changed(tmp_path):
    import json,hashlib,pytest
    from evals.harness.test_venus_retention_admission import fixture
    from data.venus.retention_review_lineage import origin,payload_sha
    report,controls,rows,files=fixture()
    rows[0]['payload']=dict(timing_evidence_protocol='Protocol.',actions=[dict(kind='speak',frame=5,text='Done.',measured_word_intervals=[dict(text='Done.',start=.1,end=.9)])])
    record=report['results'][0];record['payload_sha256']=payload_sha(rows[0]['payload'])
    record['raw_response']='{"passed":false,"reason":"Clock ambiguity"}';record['parsed']=json.loads(record['raw_response'])
    files[record['id']]=dict(record)
    report['all96_semantic_pass']=False;report['source']='original-producer'
    report['measured_alignment_sha256']='2c85c0aa25f7dc570e515c43b81f9e92a92869e004d979e7275abf8261efac36'
    for name,data in files.items():(tmp_path/(name+'.json')).write_text(json.dumps(data))
    (tmp_path/'report.json').write_text(json.dumps(report));(tmp_path/'calibration.json').write_text(json.dumps(controls))
    digest=hashlib.sha256((tmp_path/'report.json').read_bytes()).hexdigest()
    changed,reused,lineage=origin(tmp_path,digest,rows,'original-manifest')
    assert lineage['failed_ids']==['teacher-0'] and len(reused)==111
    assert changed[1:]==rows[1:] and changed[0]!=rows[0]
    assert 'teacher-0' not in reused
    # The final cohort still needs96 actual passing verdicts with exact inputs.
    from data.venus.retention_admission import validate_review
    final=copy.deepcopy(report);final['all96_semantic_pass']=True
    corrected=copy.deepcopy(record);corrected['payload_sha256']=payload_sha(changed[0]['payload'])
    corrected['raw_response']='{"passed":true,"reason":"Explicit clock resolves ambiguity"}'
    corrected['parsed']=json.loads(corrected['raw_response']);final['results'][0]=corrected
    finalfiles=copy.deepcopy(files);finalfiles['teacher-0']=corrected
    assert validate_review(final,controls,changed,'original-manifest',finalfiles.__getitem__)
    final['results'][0]=record
    with pytest.raises(ValueError):validate_review(final,controls,changed,'original-manifest',finalfiles.__getitem__)
    bad=dict(files['teacher-1']);bad['parsed']={'passed':False,'reason':'Tampered'}
    (tmp_path/'teacher-1.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):origin(tmp_path,digest,rows,'original-manifest')
