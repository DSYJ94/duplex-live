"""Preserve complete reviews; re-review only flagged, clock-clarified payloads."""
import copy,hashlib,json
from pathlib import Path
from .retention_semantics import calibration_v2,verdict
from .reviewer_model import REPO,REVISION,sha


def payload_sha(payload):return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def absolute_clock(row):
    result=copy.deepcopy(row)
    for action in result['payload']['actions']:
        if action['kind']!='speak':continue
        for w in action['measured_word_intervals']:
            w['source_relative_start'],w['source_relative_end']=w['start'],w['end']
            w['start'],w['end']=action['frame']+w['start'],action['frame']+w['end']
        action['measured_word_clock']='absolute scene seconds; source_relative_* preserve exact original recording seconds'
    result['payload']['timing_evidence_protocol']+=' Measured start/end now use ABSOLUTE scene seconds, while source_relative_start/end preserve the original recording clock. Speech-unit offsets remain relative to action.frame. These are coordinate translations, not edits to audio or word alignments.'
    return result


def origin(root,expected_sha,rows,manifest_sha):
    root=Path(root)
    if sha(root/'report.json')!=expected_sha:raise ValueError('Pinned complete original review required')
    report=json.loads((root/'report.json').read_text());controls=json.loads((root/'calibration.json').read_text())
    if (not report['complete96'] or len(report['results'])!=96 or not report['calibration16_all_pass']
            or report['manifest_sha256']!=manifest_sha or report.get('measured_alignment_sha256')!='2c85c0aa25f7dc570e515c43b81f9e92a92869e004d979e7275abf8261efac36'
            or report['reviewer']['repo']!=REPO or report['reviewer']['revision']!=REVISION
            or report['new_optimizer_updates']!=0 or report['external_API_calls']!=0
            or not report['frozen_model_weights_unchanged'] or not report['public_weights_all_hashes']
            or report['calibration_version']!='v2' or not controls['all16_pass'] or len(controls['controls'])!=16):
        raise ValueError('Whole original measured96 and calibrated reviewer required')
    expected={r['id']:r for r in rows};records={}
    if len(expected)!=96 or len({r['id'] for r in report['results']})!=96:raise ValueError('No missing or duplicate original teacher')
    for record in controls['controls']+report['results']:
        raw=dict(record);raw.pop('expected',None)
        if raw!=json.loads((root/(record['id']+'.json')).read_text()) or verdict(raw['raw_response'])!=raw['parsed']:
            raise ValueError('Original actual raw verdict differs')
        records[record['id']]=raw
    for record,control in zip(controls['controls'],calibration_v2()):
        if record['id']!=control['id'] or record['payload_sha256']!=payload_sha(control['payload']) or not record['parsed'] or record['parsed']['passed'] is not control['expected'] or record['expected'] is not control['expected']:
            raise ValueError('Original unchanged16 calibration required')
    failed=[]
    for item in report['results']:
        if item['id'] not in expected or item['payload_sha256']!=payload_sha(expected[item['id']]['payload']):raise ValueError('Original source payload differs')
        if not item['parsed'] or not item['parsed']['passed']:failed.append(item['id'])
    if not failed:raise ValueError('No failed records: another model pass is unnecessary')
    changed=[absolute_clock(r) if r['id'] in failed else r for r in rows]
    reused={k:v for k,v in records.items() if k not in failed}
    return changed,reused,dict(report_sha256=expected_sha,source=report['source'],failed_ids=failed,
        reused_record_hashes={k:sha(root/(k+'.json')) for k in reused},original_directory=str(root))
