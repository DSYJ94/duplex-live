"""Bind every independent verdict to the exact measured96 TRAIN teachers.

This admits native teacher replay only, never optimizer updates or adoption.
"""
import hashlib
import json
from pathlib import Path
from .retention_semantics import calibration_v2,teachers,verdict
from .reviewer_model import REPO,REVISION,sha


def payload_sha(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def validate_review(report,controls,rows,manifest_sha,read_record):
    for key in ('calibration16_complete','calibration16_all_pass','complete96',
                'all96_semantic_pass','frozen_model_weights_unchanged','public_weights_all_hashes'):
        if report.get(key) is not True:raise ValueError('Complete unchanged independent96 review required: '+key)
    if (report.get('manifest_sha256')!=manifest_sha or report.get('required')!=96 or report.get('calibration_version')!='v2'
            or report.get('new_optimizer_updates')!=0 or report.get('external_API_calls')!=0
            or report.get('training_approved') is not False
            or report.get('reviewer',{}).get('repo')!=REPO
            or report.get('reviewer',{}).get('revision')!=REVISION):
        raise ValueError('Exact independent public reviewer, measured inputs and no training claim required')
    expected=calibration_v2()
    if controls.get('all16_pass') is not True or len(controls.get('controls',[]))!=16:
        raise ValueError('All16 unchanged controls required')
    def check(actual,original,expected_pass):
        if (actual['id']!=original['id'] or actual['payload_sha256']!=payload_sha(original['payload'])
                or actual['parsed']!=verdict(actual['raw_response']) or actual['parsed'] is None
                or actual['parsed']['passed'] is not expected_pass):
            raise ValueError('Verdict, exact payload or expected result differs')
        raw=dict(actual);raw.pop('expected',None)
        if read_record(actual['id'])!=raw:raise ValueError('Actual independent verdict file differs')
    for actual,original in zip(controls['controls'],expected):
        if actual.get('expected') is not original['expected']:raise ValueError('Control answer changed')
        check(actual,original,original['expected'])
    actual=report.get('results',[])
    if len(rows)!=96 or len(actual)!=96 or len({r['id'] for r in actual})!=96:
        raise ValueError('All96 distinct teachers required; no passing subset')
    expected_rows={r['id']:r for r in rows}
    if len(expected_rows)!=96 or set(expected_rows)!={r['id'] for r in actual}:
        raise ValueError('Exact complete96 teacher identities required')
    for item in actual:check(item,expected_rows[item['id']],True)
    return True


def admit(timelines,semantic,expected_sha):
    semantic=Path(semantic)
    if sha(semantic/'report.json')!=expected_sha:raise ValueError('Pinned actual semantic report hash required')
    rows,manifest_sha=teachers(timelines)
    report=json.loads((semantic/'report.json').read_text())
    controls=json.loads((semantic/'calibration.json').read_text())
    validate_review(report,controls,rows,manifest_sha,
                    lambda name:json.loads((semantic/(name+'.json')).read_text()))
    return dict(semantic_report_sha256=expected_sha,calibration_sha256=sha(semantic/'calibration.json'),
                timeline_manifest_sha256=manifest_sha,semantic_source=report['source'],
                complete96=True,native_replay_only=True,training_approved=False)
