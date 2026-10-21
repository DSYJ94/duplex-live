"""One bounded new version: retain36 raw teachers and recapture four diagnosed failures.

Original synthetic input PCM, model, prompt, voice and raw recorder are unchanged.
All40 must subsequently pass independent quality, semantic and native conversion
approvals. No failed teacher is retried within this collection or silently dropped.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from data.synthetic.native_control_render import sha256
from evals import native_control_teacher as recorder
from evals.native_control_teacher_artifacts import validate_collection, verify_record
from evals.native_control_teacher_reaudit import verify as verify_quality

FAILED_IDS = [f'native-control3-train-{i:03d}' for i in (6, 8, 15, 29)]


def parent_evidence(raw, source_audit, parent_batch, quality, diagnosis):
    report = verify_quality(raw, source_audit, parent_batch, quality, require_pass=False)
    parent = json.loads((parent_batch / 'teacher.json').read_text())
    failed = [r['id'] for r in report['results'] if not r['checks']['passed']]
    diagnostic = json.loads(diagnosis.read_text())
    if (report['passed'] is not False or failed != FAILED_IDS
            or diagnostic['status'] != 'complete_five_failed_original_teacher_diagnostics'
            or diagnostic['training_approved'] is not False or diagnostic['diagnostic_only'] is not True
            or diagnostic['original_audio_changed'] is not False):
        raise ValueError('Require complete failed40 with exactly the four independently diagnosed failures')
    by_id = {r['id']: r for r in diagnostic['results']}
    for record in parent['recordings']:
        if record['id'] in failed:
            row = by_id.get(record['id'])
            if not row or row['wav_sha256'] != record['artifacts']['model_out.wav']:
                raise ValueError('Diagnostic does not bind the actual failed teacher WAV')
            response = diagnosis.parent / row['asr_file']
            if row['asr_sha256'] != sha256(response):
                raise ValueError('Original secondary recognizer diagnostic changed')
    return parent, report


def verify_repair(raw, source_audit, batch, manifest):
    repair = manifest['repair']
    parent_batch, quality, diagnosis = (Path(repair[k]) for k in ('parent_batch', 'parent_quality', 'diagnosis'))
    parent, _ = parent_evidence(raw, source_audit, parent_batch, quality, diagnosis)
    if (repair['schema'] != 'dl-control3-diagnosed-teacher-repair-v1'
            or repair['code_sha256'] != sha256(Path(__file__))
            or repair['parent_batch_sha256'] != sha256(parent_batch / 'teacher.json')
            or repair['parent_quality_sha256'] != sha256(quality) or repair['diagnosis_sha256'] != sha256(diagnosis)
            or repair['new_session_ids'] != FAILED_IDS or repair['maximum_new_sessions'] != 4
            or repair['retained_records'] != 36 or repair['original_inputs_or_recorder_or_config_changed'] is not False):
        raise ValueError('Diagnosed-only teacher repair scope or provenance changed')
    retained = []
    for record, old in zip(manifest['recordings'], parent['recordings']):
        if record['id'] not in FAILED_IDS:
            if record != old:
                raise ValueError('Previously passing teacher artifact or receipt was replaced')
            retained.append(record['id'])
    if retained != repair['retained_ids'] or len(retained) != 36:
        raise ValueError('Complete36 byte-identical retained teacher inventory required')
    return repair


async def collect(raw, source_audit, parent_batch, quality, diagnosis, out, key, lock):
    parent, _ = parent_evidence(raw, source_audit, parent_batch, quality, diagnosis)
    from data.synthetic.native_control_render import validate_raw
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
                       repair=dict(schema='dl-control3-diagnosed-teacher-repair-v1', code_sha256=sha256(Path(__file__)),
                                   parent_batch=str(parent_batch), parent_quality=str(quality), diagnosis=str(diagnosis),
                                   parent_batch_sha256=sha256(parent_batch / 'teacher.json'), parent_quality_sha256=sha256(quality),
                                   diagnosis_sha256=sha256(diagnosis), maximum_new_sessions=4, new_session_ids=FAILED_IDS,
                                   retained_records=36, retained_ids=[r['id'] for r in retained],
                                   original_inputs_or_recorder_or_config_changed=False))
        target = out / 'teacher.json'
        recorder.atomic_json(target, receipt)
        semaphore = asyncio.Semaphore(2)
        async def one(row):
            async with semaphore:
                record = await recorder.capture_one(raw, row, out / row['id'], parent['source_word_audit_sha256'], key)
                receipt['recordings'].append(record)
                receipt['recordings'].sort(key=lambda r: r['id'])
                recorder.atomic_json(target, receipt)
                print(json.dumps(dict(id=row['id'], transport_complete=record['transport_complete'],
                                      total_records=len(receipt['recordings']), maximum_new_sessions=4)), flush=True)
                if not record['transport_complete']:
                    raise RuntimeError('Diagnosed teacher repair transport failed; preserve all evidence')
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
