"""One explicitly versioned recovery of the diagnosed request17 read timeout.

Reuses sixteen immutable WAV/receipt pairs. Makes at most24 new requests for the
same approved payloads: one interrupted request plus23 unattempted objects.
The original failure and all41 aggregate attempts remain visible. No automatic
retry, waveform edit, annotation payload, teacher collection or subset approval.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from data.synthetic.recovery_curriculum import MODEL
from evals.native_reasoning_source import ENDPOINT, INSTRUCTIONS, requests_for, sha
from evals.native_reasoning_source import verify as original_verify
from evals.native_control_teacher import atomic_json, collection_lock, local_key

SCHEMA = 'dl-native-reasoning32-user-source-transport-recovery-v2'
ORIGINAL_CODE = '0b1e7b0dfefe49c11991304d53eda398b241234c32fe60829a49ed18b625ec4c'


def failed_source(root, *, require_actual=True):
    plan_path = root/'source-plan.json'; plan = json.loads(plan_path.read_text())
    objects, scenes = requests_for(plan)
    r = json.loads((root/'source-audio.json').read_text())
    if (r['schema'] != 'dl-native-reasoning32-user-source-v1'
            or r['phase'] != 'failed_preserved_no_retry' or r['error_type'] != 'ConnectionError'
            or r['complete32'] is not False or r['code_sha256'] != ORIGINAL_CODE
            or r['plan_sha256'] != sha(plan_path) or r['source_model'] != MODEL
            or r['endpoint'] != ENDPOINT or r['scenes'] != scenes
            or r['actual_requests'] != 17 or len(r['objects']) != 16
            or [x['object_id'] for x in r['objects']] != list(objects)[:16]
            or (require_actual and r['actual_api'] is not True)):
        raise ValueError('Exact preserved sixteen-object/request17 transport failure required')
    for row in r['objects']:
        check_object(root, row, objects[row['object_id']], attempt=0, actual=r['actual_api'])
    return r, plan, objects, scenes


def check_object(root, row, payload, *, attempt, actual):
    identity = row['object_id']; name = 'objects/'+identity+'.wav'
    wav = root/name; receipt = wav.with_suffix('.receipt.json')
    if wav.is_symlink() or receipt.is_symlink():
        raise ValueError('Regular immutable source artifacts required')
    v = json.loads(receipt.read_text())
    expected = dict(request_id=v.get('request_id'), source_model=MODEL, voice=payload['voice'],
        input_text=payload['input'], instructions=INSTRUCTIONS, sha256=sha(wav), attempt=attempt,
        actual_api=actual, request_payload_sha256=identity, response_bytes=wav.stat().st_size)
    if (row != dict(object_id=identity, wav=name, wav_sha256=sha(wav), receipt_sha256=sha(receipt))
            or v != expected or (actual and not v['request_id'])):
        raise ValueError('Source payload, original WAV or receipt changed')


def recover(failed, out, env_file, *, post=None):
    actual = post is None
    prior, plan, objects, scenes = failed_source(failed, require_actual=actual)
    if prior['actual_api'] is not actual:
        raise ValueError('Fixture and actual source provenance cannot be mixed')
    if actual:
        import requests
        post = requests.post; key = local_key(env_file)
    else:
        key = 'fixture-only-no-credential'
    with collection_lock(out.parent/'.private/native-reasoning32-source.lock'):
        out.mkdir(parents=True, exist_ok=False); (out/'objects').mkdir()
        (out/'source-plan.json').write_bytes((failed/'source-plan.json').read_bytes())
        (out/'failed-v1-source-audio.json').write_bytes((failed/'source-audio.json').read_bytes())
        for row in prior['objects']:
            for name in (row['wav'], str(Path(row['wav']).with_suffix('.receipt.json'))):
                shutil.copyfile(failed/name, out/name)
        remaining = list(objects)[16:]
        report = dict(schema=SCHEMA, phase='collecting_unapproved_user_source',
            created_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(out/'source-plan.json'),
            code_sha256=sha(Path(__file__)), source_model=MODEL, endpoint=ENDPOINT, actual_api=actual,
            original_failed_manifest_sha256=sha(out/'failed-v1-source-audio.json'),
            original_attempts=17, reused_objects=16, maximum_new_requests=24, actual_requests=0,
            aggregate_actual_requests=17, new_request_ids=remaining,
            interrupted_request_id=remaining[0], scenes=scenes, objects=list(prior['objects']),
            complete32=False, training_approved=False, source_quality_approved=False,
            teacher_collection_completed=False, operator_or_answer_annotations_sent=False,
            owner_conversations_used=False, original_WAVs_unchanged=True, gain_changed=False,
            resampled=False, automatic_retry=False, diagnosed_manual_transport_replacement=True,
            timeline_assembled=False, actual_native_episode_frame_bound_still_required=1100,
            read_timeout_s=180, maximum_response_bytes=8*1024*1024)
        def save(): atomic_json(out/'source-audio.json', report)
        save()
        try:
            for identity in remaining:
                payload = objects[identity]; wav = out/'objects'/(identity+'.wav')
                partial = out/'failed-response.bin'
                report['actual_requests'] += 1
                report['aggregate_actual_requests'] = 17 + report['actual_requests']; save()
                began = time.monotonic()
                response = post(ENDPOINT, headers={'Authorization': 'Bearer '+key}, json=payload,
                                timeout=(10, 180), allow_redirects=False, stream=True)
                try:
                    count = 0
                    with partial.open('xb') as writer:
                        for block in response.iter_content(65536):
                            if count+len(block) > 8*1024*1024 or time.monotonic()-began > 360:
                                raise RuntimeError('Bounded source response exceeded; no retry')
                            writer.write(block); count += len(block)
                    if response.status_code != 200:
                        report['http_status'] = response.status_code
                        raise RuntimeError('Source transport HTTP failure; raw response preserved')
                    partial.rename(wav)
                    receipt = dict(request_id=response.headers.get('x-request-id'), source_model=MODEL,
                        voice=payload['voice'], input_text=payload['input'], instructions=INSTRUCTIONS,
                        sha256=sha(wav), attempt=1 if identity==remaining[0] else 0, actual_api=actual,
                        request_payload_sha256=identity, response_bytes=count)
                    target = wav.with_suffix('.receipt.json'); atomic_json(target, receipt)
                    report['objects'].append(dict(object_id=identity, wav=str(wav.relative_to(out)),
                        wav_sha256=sha(wav), receipt_sha256=sha(target))); save()
                    print(json.dumps(dict(source_objects=len(report['objects']), expected=40,
                        new_requests=report['actual_requests'], aggregate_requests=report['aggregate_actual_requests'])), flush=True)
                finally:
                    response.close()
            report.update(phase='complete_unapproved_user_source', complete32=True); save()
        except BaseException as error:
            report.update(phase='failed_preserved_no_retry', error_type=type(error).__name__); save(); raise
    return report


def verify(root, *, require_actual=True):
    root = Path(root); r = json.loads((root/'source-audio.json').read_text())
    if r.get('schema') != SCHEMA:
        return original_verify(root, require_actual=require_actual)
    plan = json.loads((root/'source-plan.json').read_text()); objects, scenes = requests_for(plan)
    original_path = root/'failed-v1-source-audio.json'; prior = json.loads(original_path.read_text())
    if (r['phase'] != 'complete_unapproved_user_source' or r['complete32'] is not True
            or r['code_sha256'] != sha(Path(__file__)) or r['plan_sha256'] != sha(root/'source-plan.json')
            or r['source_model'] != MODEL or r['endpoint'] != ENDPOINT
            or r['original_failed_manifest_sha256'] != sha(original_path)
            or prior['code_sha256'] != ORIGINAL_CODE or prior['phase'] != 'failed_preserved_no_retry'
            or prior['error_type'] != 'ConnectionError' or prior['actual_requests'] != 17
            or len(prior['objects']) != 16 or r['objects'][:16] != prior['objects']
            or prior['actual_api'] is not r['actual_api'] or prior['plan_sha256'] != r['plan_sha256']
            or r['scenes'] != scenes or prior['scenes'] != scenes
            or r['new_request_ids'] != list(objects)[16:] or r['interrupted_request_id'] != list(objects)[16]
            or (r['original_attempts'], r['reused_objects'], r['maximum_new_requests'], r['actual_requests'], r['aggregate_actual_requests']) != (17,16,24,24,41)
            or [x['object_id'] for x in r['objects']] != list(objects)
            or (require_actual and r['actual_api'] is not True)
            or any(r[k] is not False for k in ('training_approved','source_quality_approved','teacher_collection_completed',
                'operator_or_answer_annotations_sent','owner_conversations_used','gain_changed','resampled','automatic_retry','timeline_assembled'))
            or r['original_WAVs_unchanged'] is not True or r['diagnosed_manual_transport_replacement'] is not True
            or r['actual_native_episode_frame_bound_still_required'] != 1100):
        raise ValueError('Complete exact source recovery, preserved failed history and all41 attempts required')
    expected = {'source-plan.json', 'source-audio.json', 'failed-v1-source-audio.json'}
    for index, row in enumerate(r['objects']):
        check_object(root, row, objects[row['object_id']], attempt=1 if index==16 else 0, actual=r['actual_api'])
        expected.update((row['wav'], str(Path(row['wav']).with_suffix('.receipt.json'))))
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} != expected:
        raise ValueError('Unexpected or missing full source recovery inventory')
    return r, plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('failed','out','env-file'): p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args(); recover(a.failed, a.out, a.env_file); verify(a.out)
