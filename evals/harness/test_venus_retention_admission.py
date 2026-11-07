import copy
import json
import pytest
from data.venus.retention_admission import validate_review,payload_sha
from data.venus.retention_semantics import calibration_v2 as calibration
from data.venus.reviewer_model import REPO,REVISION


def fixture():
    rows=[dict(id='teacher-'+str(i),payload={'question':'fixture '+str(i)}) for i in range(96)]
    def record(row,passed):
        raw=json.dumps(dict(passed=passed,reason='fixture verdict'))
        return dict(id=row['id'],payload_sha256=payload_sha(row['payload']),raw_response=raw,parsed=json.loads(raw))
    controls=dict(all16_pass=True,controls=[dict(record(r,r['expected']),expected=r['expected']) for r in calibration()])
    report=dict(calibration16_complete=True,calibration16_all_pass=True,complete96=True,
        all96_semantic_pass=True,frozen_model_weights_unchanged=True,public_weights_all_hashes=True,
        manifest_sha256='original-manifest',calibration_version='v2',required=96,new_optimizer_updates=0,external_API_calls=0,
        training_approved=False,reviewer=dict(repo=REPO,revision=REVISION),results=[record(r,True) for r in rows])
    files={r['id']:{k:v for k,v in r.items() if k!='expected'} for r in controls['controls']+report['results']}
    return report,controls,rows,copy.deepcopy(files)


def test_complete_actual_records_bind_controls_and_every_teacher():
    report,controls,rows,files=fixture()
    assert validate_review(report,controls,rows,'original-manifest',files.__getitem__)


@pytest.mark.parametrize('mutation', ['subset','duplicate','payload','raw','record','control','revision','manifest','frozen'])
def test_summary_pass_flags_cannot_hide_incomplete_or_changed_evidence(mutation):
    report,controls,rows,files=fixture()
    if mutation=='subset':report['results'].pop()
    if mutation=='duplicate':report['results'][-1]=report['results'][0]
    if mutation=='payload':rows[-1]['payload']['question']='Changed after review'
    if mutation=='raw':report['results'][0]['raw_response']='{"passed":false,"reason":"wrong"}'
    if mutation=='record':files['teacher-95']['parsed']['reason']='Different file'
    if mutation=='control':controls['controls'][1]['expected']=True
    if mutation=='revision':report['reviewer']['revision']='unapproved-reviewer'
    if mutation=='manifest':report['manifest_sha256']='changed'
    if mutation=='frozen':report['frozen_model_weights_unchanged']=False
    with pytest.raises(ValueError):validate_review(report,controls,rows,'original-manifest',files.__getitem__)
