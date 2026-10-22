"""One diagnosed two-record followup, preserving all38 passing raw histories.

The original four-failure repair scope remains independently validated. This
followup adds two bounded actual sessions with the identical recorder/config,
and keeps every prior attempt. No content rerolls occur within the invocation.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from data.synthetic.native_control_render import sha256, validate_raw
from evals import native_control_teacher as recorder
from evals.native_control_teacher_artifacts import validate_collection, verify_record
from evals.native_control_teacher_reaudit import verify as verify_quality
from evals.native_control_teacher_repair import verify_repair

FAILED_IDS = ['native-control3-train-006', 'native-control3-train-029']


def evidence(raw, source_audit, batch, quality, diagnosis):
    report = verify_quality(raw, source_audit, batch, quality, require_pass=False)
    manifest = json.loads((batch / 'teacher.json').read_text())
    verify_repair(raw, source_audit, batch, manifest)
    diag = json.loads(diagnosis.read_text())
    if (report['passed'] is not False
            or [r['id'] for r in report['results'] if not r['checks']['passed']] != FAILED_IDS
            or diag['status'] != 'complete_two_failed_original_teacher_diagnostics'
            or diag['training_approved'] is not False or diag['diagnostic_only'] is not True
            or diag['original_audio_changed'] is not False
            or [r['id'] for r in diag['results']] != FAILED_IDS):
        raise ValueError('Complete38/40 failure and both actual diagnostic records required')
    records = {r['id']: r for r in manifest['recordings']}
    for row in diag['results']:
        if (row['wav_sha256'] != records[row['id']]['artifacts']['model_out.wav']
                or row['asr_sha256'] != sha256(diagnosis.parent / row['asr_file'])):
            raise ValueError('Followup diagnostic differs from the actual failed waveform/ASR')
    return manifest


def verify_followup(raw, source_audit, batch, manifest):
    lineage = manifest['repair_followup']
    parent_batch, quality, diagnosis = (Path(lineage[k]) for k in ('parent_batch', 'parent_quality', 'diagnosis'))
    parent = evidence(raw, source_audit, parent_batch, quality, diagnosis)
    if (lineage['schema'] != 'dl-control3-diagnosed-two-teacher-followup-v1'
            or lineage['code_sha256'] != sha256(Path(__file__))
            or lineage['parent_batch_sha256'] != sha256(parent_batch / 'teacher.json')
            or lineage['parent_quality_sha256'] != sha256(quality)
            or lineage['diagnosis_sha256'] != sha256(diagnosis)
            or lineage['maximum_new_sessions'] != 2 or lineage['new_session_ids'] != FAILED_IDS
            or lineage['retained_records'] != 38 or lineage['inputs_or_config_or_recorder_changed'] is not False
            or manifest['repair'] != parent['repair']):
        raise ValueError('Two-record followup scope/receipt changed')
    retained = []
    for new, old in zip(manifest['recordings'], parent['recordings']):
        if new['id'] not in FAILED_IDS:
            if new != old:
                raise ValueError('Followup changed a passing original teacher artifact/receipt')
            retained.append(new['id'])
    if len(retained) != 38 or retained != lineage['retained_ids']:
        raise ValueError('Exactly38 byte-identical passing teacher records required')
    return lineage


async def collect(raw, source_audit, parent_batch, quality, diagnosis, out, key, lock):
    parent = evidence(raw, source_audit, parent_batch, quality, diagnosis)
    source, _ = validate_raw(raw)
    with recorder.collection_lock(lock):
        out.mkdir(parents=True, exist_ok=False)
        retained = [r for r in parent['recordings'] if r['id'] not in FAILED_IDS]
        for record in retained:
            folder = out / record['id']
            folder.mkdir()
            for name in recorder.ARTIFACTS:
                shutil.copy2(parent_batch / record['id'] / name, folder / name)
            row = next(r for r in source['cases'] if r['id'] == record['id'])
            verify_record(raw, row, folder, parent['source_word_audit_sha256'], record)
        receipt = dict(parent, created_utc=datetime.now(timezone.utc).isoformat(),
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       status='recording', recordings=retained.copy(),
                       repair_followup=dict(schema='dl-control3-diagnosed-two-teacher-followup-v1',
                           code_sha256=sha256(Path(__file__)), parent_batch=str(parent_batch), parent_quality=str(quality),
                           diagnosis=str(diagnosis), parent_batch_sha256=sha256(parent_batch / 'teacher.json'),
                           parent_quality_sha256=sha256(quality), diagnosis_sha256=sha256(diagnosis),
                           maximum_new_sessions=2, new_session_ids=FAILED_IDS, retained_records=38,
                           retained_ids=[r['id'] for r in retained], inputs_or_config_or_recorder_changed=False))
        target = out / 'teacher.json'
        recorder.atomic_json(target, receipt)
        async def one(row):
            record = await recorder.capture_one(raw, row, out / row['id'], parent['source_word_audit_sha256'], key)
            receipt['recordings'].append(record)
            receipt['recordings'].sort(key=lambda r: r['id'])
            recorder.atomic_json(target, receipt)
            print(json.dumps(dict(id=row['id'], transport_complete=record['transport_complete'],
                                  total_records=len(receipt['recordings']), maximum_new_sessions=2)), flush=True)
            if not record['transport_complete']:
                raise RuntimeError('Followup transport failed; no reroll or subset approval')
        try:
            async with asyncio.TaskGroup() as group:
                for row in source['cases']:
                    if row['id'] in FAILED_IDS:
                        group.create_task(one(row))
        finally:
            receipt['status'] = ('raw_complete_unapproved' if len(receipt['recordings']) == 40
                                 and all(r['transport_complete'] for r in receipt['recordings']) else 'incomplete_unapproved')
            recorder.atomic_json(target, receipt)
        validate_collection(raw, source_audit, out)
        verify_repair(raw, source_audit, out, receipt)
        verify_followup(raw, source_audit, out, receipt)
        return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'source-audit', 'parent-batch', 'quality', 'diagnosis', 'out', 'env-file'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--lock', type=Path, default=Path('runs/local-migration-20261021/.private/control3-capture.lock'))
    args = ap.parse_args()
    result = asyncio.run(collect(args.raw, args.source_audit, args.parent_batch, args.quality,
                                args.diagnosis, args.out, recorder.local_key(args.env_file), args.lock))
    print(json.dumps(dict(status=result['status'], complete=len(result['recordings']), training_approved=False)))


if __name__ == '__main__':
    main()
