"""One diagnosed provider take and a complete versioned source corpus.

The canonical object ID continues to identify the approved words/voice. The new
instruction-bearing API payload has a DIFFERENT, explicit hash. Old receipts and
waveforms remain immutable. No training or source-quality approval occurs here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from evals.native_control_teacher import atomic_json, collection_lock, local_key
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_source import ENDPOINT, requests_for, sha
from evals.native_reasoning_source_repair import files, verify as parent_verify
from evals.native_reasoning_source_quality import verify as parent_quality

OBJECT = '8d57491280185df1844bbe328e658cd8f03172c075ea740376e8f63fb3928b77'
PARENT = 'cda077f431fcbbf48306441fce4b94fcd8def086e9cc1a5f5222c89c0064b72d'
QUALITY = 'd22e096126f0df20e413af5b7c8ca2bf3f15e8aa920752c5167ad8299a7cd3c8'
FAILED = '09bb11d17cc0cacfeb3d2c86d3855519571c658c4f8ba0dd86cdd8515488ff60'
FAILED_REPORT = 'b2fd5829d268c8644620255b291f8776b22095141044f02e0d8b1b569acfa2f4'
INSTRUCTIONS = ('Speak at a brisk but natural conversational pace in clear English, '
                'with brief sentence pauses. Read exactly the supplied words, without additions.')
TAKE_SCHEMA = 'dl-native-reasoning32-diagnosed-pacing-take-v1'
SCHEMA = 'dl-native-reasoning32-user-source-pacing-repair-v4'
TAKE_FILES = {'request.json', 'take.json', 'replacement.wav', 'replacement.receipt.json',
              'original.wav', 'original.receipt.json', 'failed-teacher-collection.json', 'failed-teacher-report.json'}


def payload_for(plan):
    objects, _ = requests_for(plan)
    payload = dict(objects[OBJECT], instructions=INSTRUCTIONS)
    if payload['voice'] != 'echo' or canonical_hash(payload) == OBJECT:
        raise ValueError('Same approved echo utterance and explicit new payload hash required')
    return payload


def waveform(path):
    import numpy as np
    import soundfile as sf
    pcm, rate = sf.read(path, dtype='float32')
    duration = len(pcm)/rate
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.
    if (rate != 24000 or pcm.ndim != 1 or not np.isfinite(pcm).all()
            or not .1 <= duration <= 40. or not .001 < peak <= .995):
        raise ValueError('New original provider take must be unclipped mono24kHz and at most40s')
    return dict(duration_s=duration, sample_rate=rate, peak=peak, waveform_modified=False)


def preflight(source, quality, collection, report, *, require_actual=True):
    manifest, plan = parent_verify(source, require_actual=require_actual)
    parent_quality(source, quality)
    value = json.loads(collection.read_text()); failure = json.loads(report.read_text())
    if (sha(source/'source-audio.json') != PARENT or sha(quality) != QUALITY
            or sha(collection) != FAILED or sha(report) != FAILED_REPORT
            or value['source_manifest_sha256'] != PARENT or value['source_quality_sha256'] != QUALITY
            or value['status'] != 'incomplete_failed_preserved_no_retry'
            or failure['native_frame_budget']['full_measured_prefix_live_and_span_frames'] != 1125
            or failure['native_frame_budget']['unchanged_frame_limit'] != 1100
            or failure['native_frame_budget']['original_audio_cropped'] is not False):
        raise ValueError('Exact original source qualification and actual full-frame failure required')
    affected = [s['id'] for s in manifest['scenes'] if any(c['object_id'] == OBJECT for c in s['clips'])]
    if affected != [f'native-reasoning4-train-{i:03d}' for i in (13,14,15)]:
        raise ValueError('Matched question must remain shared by all three original TRAIN scenes')
    row = next(r for r in manifest['objects'] if r['object_id'] == OBJECT)
    return manifest, plan, row, affected


def collect(source, quality, failed, out, env_file, *, post=None):
    actual = post is None
    _, plan, row, affected = preflight(source, quality, failed/'teacher-collection.json',
                                      failed/'native-reasoning4-train-013/report.json', require_actual=actual)
    payload = payload_for(plan)
    if actual:
        import requests
        post = requests.post
        key = local_key(env_file)
    else:
        key = 'fixture-only-no-credential'
    with collection_lock(out.parent/'.private/native-reasoning32-source.lock'):
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out/'request.json', payload)
        shutil.copyfile(source/row['wav'], out/'original.wav')
        shutil.copyfile((source/row['wav']).with_suffix('.receipt.json'), out/'original.receipt.json')
        shutil.copyfile(failed/'teacher-collection.json', out/'failed-teacher-collection.json')
        shutil.copyfile(failed/'native-reasoning4-train-013/report.json', out/'failed-teacher-report.json')
        take = dict(schema=TAKE_SCHEMA, phase='requesting_unapproved_take', code_sha256=sha(Path(__file__)),
                    canonical_object_id=OBJECT, actual_payload_sha256=canonical_hash(payload),
                    source_manifest_sha256=PARENT, source_quality_sha256=QUALITY, failed_collection_sha256=FAILED,
                    affected_scene_ids=affected, actual_api=actual, maximum_requests=1, actual_requests=1,
                    aggregate_prior_requests=42, aggregate_requests=43, automatic_retry=False,
                    words_model_or_voice_changed=False, delivery_instructions_changed=True,
                    waveform_edited=False, whole40_source_quality_approved=False, training_approved=False)
        atomic_json(out/'take.json', take)
        response = None
        try:
            began = time.monotonic()
            response = post(ENDPOINT, headers={'Authorization':'Bearer '+key}, json=payload,
                            timeout=(10,90), allow_redirects=False, stream=True)
            with (out/'response.partial').open('xb') as writer:
                count = 0
                for block in response.iter_content(65536):
                    count += len(block)
                    if count > 8*1024*1024 or time.monotonic()-began > 150:
                        raise RuntimeError('Bounded response exceeded; no retry')
                    writer.write(block)
            if response.status_code != 200 or not response.headers.get('x-request-id'):
                raise RuntimeError('Actual successful provider receipt required; raw response preserved')
            (out/'response.partial').rename(out/'replacement.wav')
            receipt = dict(request_id=response.headers['x-request-id'], actual_api=actual,
                           request_payload_sha256=canonical_hash(payload), source_model=payload['model'],
                           voice=payload['voice'], input_text=payload['input'], instructions=payload['instructions'],
                           sha256=sha(out/'replacement.wav'), response_bytes=count, attempt=1)
            atomic_json(out/'replacement.receipt.json', receipt)
            take.update(phase='collected_unapproved_new_take', metrics=waveform(out/'replacement.wav'),
                        wav_sha256=sha(out/'replacement.wav'), receipt_sha256=sha(out/'replacement.receipt.json'))
            atomic_json(out/'take.json', take)
        except BaseException as error:
            take.update(phase='failed_preserved_no_retry', error_type=type(error).__name__)
            atomic_json(out/'take.json', take)
            raise
        finally:
            if response is not None:
                response.close()
    verify_take(source, quality, out, require_actual=actual)
    return take


def verify_take(source, quality, take_root, *, require_actual=True):
    parent, plan, original, affected = preflight(source, quality, take_root/'failed-teacher-collection.json',
                                                take_root/'failed-teacher-report.json', require_actual=require_actual)
    value = json.loads((take_root/'take.json').read_text()); payload = payload_for(plan)
    receipt = json.loads((take_root/'replacement.receipt.json').read_text())
    expected = dict(request_id=receipt.get('request_id'), actual_api=parent['actual_api'],
                    request_payload_sha256=canonical_hash(payload), source_model=payload['model'], voice=payload['voice'],
                    input_text=payload['input'], instructions=payload['instructions'], sha256=sha(take_root/'replacement.wav'),
                    response_bytes=(take_root/'replacement.wav').stat().st_size, attempt=1)
    if (files(take_root) != TAKE_FILES or value['schema'] != TAKE_SCHEMA
            or value['phase'] != 'collected_unapproved_new_take' or value['code_sha256'] != sha(Path(__file__))
            or value['canonical_object_id'] != OBJECT or value['actual_payload_sha256'] != canonical_hash(payload)
            or value['source_manifest_sha256'] != PARENT or value['source_quality_sha256'] != QUALITY
            or value['failed_collection_sha256'] != FAILED or value['affected_scene_ids'] != affected
            or value['actual_api'] is not parent['actual_api'] or (require_actual and value['actual_api'] is not True)
            or tuple(value[k] for k in ('maximum_requests','actual_requests','aggregate_prior_requests','aggregate_requests')) != (1,1,42,43)
            or any(value[k] is not False for k in ('automatic_retry','words_model_or_voice_changed','waveform_edited',
                                                  'whole40_source_quality_approved','training_approved'))
            or value['delivery_instructions_changed'] is not True
            or json.loads((take_root/'request.json').read_text()) != payload or receipt != expected
            or (require_actual and not receipt['request_id']) or value['metrics'] != waveform(take_root/'replacement.wav')
            or value['wav_sha256'] != expected['sha256'] or value['receipt_sha256'] != sha(take_root/'replacement.receipt.json')
            or sha(take_root/'original.wav') != original['wav_sha256']
            or sha(take_root/'original.receipt.json') != original['receipt_sha256']):
        raise ValueError('Exact one-shot pacing take, explicit payload alias and immutable evidence required')
    return parent, plan


def prepare(source, quality, take, out, *, require_actual=True):
    parent, _ = verify_take(source, quality, take, require_actual=require_actual)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, out/'parent-v3'); shutil.copytree(quality.parent, out/'parent-quality-v3')
    shutil.copytree(take, out/'pacing-take'); shutil.copyfile(source/'source-plan.json', out/'source-plan.json')
    (out/'objects').mkdir(); rows = []
    for old in parent['objects']:
        row = dict(old); changed = row['object_id'] == OBJECT
        wav = take/'replacement.wav' if changed else source/row['wav']
        receipt = take/'replacement.receipt.json' if changed else wav.with_suffix('.receipt.json')
        shutil.copyfile(wav, out/row['wav']); shutil.copyfile(receipt, (out/row['wav']).with_suffix('.receipt.json'))
        row.update(wav_sha256=sha(wav), receipt_sha256=sha(receipt)); rows.append(row)
    r = dict(schema=SCHEMA, phase='complete_unapproved_user_source', complete32=True,
             code_sha256=sha(Path(__file__)), plan_sha256=sha(out/'source-plan.json'),
             source_model=parent['source_model'], actual_api=parent['actual_api'], endpoint=ENDPOINT,
             scenes=parent['scenes'], objects=rows, reused_objects=39, changed_canonical_object_id=OBJECT,
             actual_payload_sha256=canonical_hash(json.loads((take/'request.json').read_text())),
             parent_manifest_sha256=PARENT, parent_quality_sha256=QUALITY, take_sha256=sha(out/'pacing-take/take.json'),
             actual_requests=0, aggregate_actual_requests=43, automatic_retry=False,
             original_waveforms_edited=False, source_quality_approved=False, teacher_collection_completed=False,
             training_approved=False, passing_subset_accepted=False, actual_native_episode_frame_bound_still_required=1100)
    atomic_json(out/'source-audio.json', r); verify(out, require_actual=require_actual)
    return r


def verify(root, *, require_actual=True):
    r = json.loads((root/'source-audio.json').read_text())
    parent, plan = verify_take(root/'parent-v3', root/'parent-quality-v3/source-quality.json',
                              root/'pacing-take', require_actual=require_actual)
    rows = []
    for old in parent['objects']:
        row = dict(old)
        if row['object_id'] == OBJECT:
            row.update(wav_sha256=sha(root/'pacing-take/replacement.wav'),
                       receipt_sha256=sha(root/'pacing-take/replacement.receipt.json'))
        rows.append(row)
    if (r['schema'] != SCHEMA or r['phase'] != 'complete_unapproved_user_source' or r['complete32'] is not True
            or r['code_sha256'] != sha(Path(__file__)) or r['plan_sha256'] != sha(root/'source-plan.json')
            or (root/'source-plan.json').read_bytes() != (root/'parent-v3/source-plan.json').read_bytes()
            or r['source_model'] != parent['source_model'] or r['endpoint'] != ENDPOINT
            or r['actual_api'] is not parent['actual_api'] or r['objects'] != rows or r['scenes'] != parent['scenes']
            or r['parent_manifest_sha256'] != PARENT or r['parent_quality_sha256'] != QUALITY
            or r['take_sha256'] != sha(root/'pacing-take/take.json') or r['changed_canonical_object_id'] != OBJECT
            or r['actual_payload_sha256'] != canonical_hash(payload_for(plan))
            or (r['actual_requests'],r['aggregate_actual_requests'],r['reused_objects']) != (0,43,39)
            or any(r[k] is not False for k in ('automatic_retry','original_waveforms_edited','source_quality_approved',
                                              'teacher_collection_completed','training_approved','passing_subset_accepted'))
            or r['actual_native_episode_frame_bound_still_required'] != 1100):
        raise ValueError('Complete40 source with exact one-object pacing repair required')
    expected = {'source-plan.json','source-audio.json'}
    for name in ('parent-v3','parent-quality-v3','pacing-take'):
        expected |= {name+'/'+p for p in files(root/name)}
    for row in rows:
        wav = root/row['wav']; receipt = wav.with_suffix('.receipt.json')
        if sha(wav) != row['wav_sha256'] or sha(receipt) != row['receipt_sha256']:
            raise ValueError('Changed source WAV or receipt')
        expected.update((row['wav'],str(receipt.relative_to(root))))
    if files(root) != expected:
        raise ValueError('Unexpected or missing full source inventory')
    return r, plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('collect','prepare','verify'))
    for n in ('source','quality','failed','out','env-file','take'):
        p.add_argument('--'+n, type=Path)
    a = p.parse_args()
    if a.action == 'collect':
        collect(a.source,a.quality,a.failed,a.out,a.env_file)
    elif a.action == 'prepare':
        prepare(a.source,a.quality,a.take,a.out)
    else:
        verify(a.source)
