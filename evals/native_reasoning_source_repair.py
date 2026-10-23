"""Compose all40 sources with one diagnosed new take, preserving both parents.

No API or ASR calls, audio editing, gain changes, subset approval or training.
Parent42-attempt history and rejected audio remain inside the new source tree.
The complete resulting corpus still requires independent source qualification.
"""
import argparse
import array
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import wave

from data.synthetic.recovery_curriculum import MODEL
from evals.native_reasoning_source import ENDPOINT, INSTRUCTIONS, requests_for, sha
from evals.native_reasoning_source_recovery import verify as parent_verify, check_object
from evals.native_control_teacher import atomic_json

SCHEMA = 'dl-native-reasoning32-user-source-single-waveform-repair-v3'
OBJECT = '7acae062d4745c22b3f89744c2b963b5b5580d292ebb0c7062eb12d852f15d87'
OLD_QUALITY_CODE = 'af47365e03122406c9a57e438df8bfa56b30a245c3fb61baac2d9606ae749ad8'
TAKE_FILES = {'request.json','take.json','original-rejected.wav','original-rejected.receipt.json',
              'replacement.wav','replacement.receipt.json'}


def peak(path):
    with wave.open(str(path)) as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1,2,24000):
            raise ValueError('Original mono24kHz PCM16 provider WAV required')
        values = array.array('h')
        while True:
            block = reader.readframes(8192)
            if not block: break
            values.frombytes(block)
    if not values: raise ValueError('Empty source WAV')
    return max(abs(v) for v in values)/32768


def files(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Source evidence trees must contain no symlinks')
    return {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}


def replacement(parent_root, take_root, quality, *, require_actual=True):
    parent, plan = parent_verify(parent_root, require_actual=require_actual)
    objects, _ = requests_for(plan); take = json.loads((take_root/'take.json').read_text())
    before = next(x for x in parent['objects'] if x['object_id']==OBJECT)
    prior = json.loads(quality.read_text()); payload = objects[OBJECT]
    if (files(take_root) != TAKE_FILES or take['schema'] != 'dl-native-reasoning32-single-diagnosed-source-take-v1'
            or take['phase'] != 'collected_unapproved_new_take' or take['object_id'] != OBJECT
            or take['request_payload_sha256'] != OBJECT
            or (take['maximum_requests'], take['actual_requests'], take['aggregate_prior_requests'], take['aggregate_requests']) != (1,1,41,42)
            or take['source_manifest_sha256'] != sha(parent_root/'source-audio.json')
            or take['original_quality_sha256'] != sha(quality)
            or any(take[k] is not False for k in ('approved_utterance_or_voice_or_instructions_changed',
                'automatic_retry','training_approved','whole40_source_quality_approved'))
            or json.loads((take_root/'request.json').read_text()) != payload
            or sha(take_root/'original-rejected.wav') != before['wav_sha256']
            or sha(take_root/'original-rejected.receipt.json') != before['receipt_sha256']
            or peak(take_root/'original-rejected.wav') <= .995
            or not .001 < peak(take_root/'replacement.wav') <= .995
            or prior['code_sha256'] != OLD_QUALITY_CODE or prior['passed'] is not False
            or prior['source_manifest_sha256'] != sha(parent_root/'source-audio.json')
            or [x['object_id'] for x in prior['results']] != list(objects)
            or [x['object_id'] for x in prior['results'] if x['peak']>.995] != [OBJECT]):
        raise ValueError('Only the diagnosed over-peak object and one exact-words new take may change')
    receipt_path = take_root/'replacement.receipt.json'; receipt = json.loads(receipt_path.read_text())
    expected = dict(request_id=receipt.get('request_id'), source_model=MODEL, voice=payload['voice'],
        input_text=payload['input'], instructions=INSTRUCTIONS, sha256=sha(take_root/'replacement.wav'),
        attempt=1, actual_api=parent['actual_api'], request_payload_sha256=OBJECT,
        response_bytes=(take_root/'replacement.wav').stat().st_size)
    if (receipt != expected or (require_actual and not receipt['request_id'])
            or take['wav_sha256'] != expected['sha256'] or take['receipt_sha256'] != sha(receipt_path)):
        raise ValueError('Unchanged actual provider replacement receipt required')
    row = dict(object_id=OBJECT, wav='objects/'+OBJECT+'.wav', wav_sha256=expected['sha256'],
               receipt_sha256=sha(receipt_path))
    return parent, plan, row


def prepare(parent_root, take_root, quality, out, *, require_actual=True):
    parent, plan, changed = replacement(parent_root, take_root, quality, require_actual=require_actual)
    files(parent_root); out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent_root, out/'parent-v2'); shutil.copytree(take_root, out/'replacement-take')
    shutil.copyfile(quality, out/'parent-quality.json')
    shutil.copyfile(parent_root/'source-plan.json', out/'source-plan.json'); (out/'objects').mkdir()
    rows = []
    for row in parent['objects']:
        if row['object_id']==OBJECT:
            shutil.copyfile(take_root/'replacement.wav', out/changed['wav'])
            shutil.copyfile(take_root/'replacement.receipt.json', (out/changed['wav']).with_suffix('.receipt.json'))
            rows.append(changed)
        else:
            for name in (row['wav'], str(Path(row['wav']).with_suffix('.receipt.json'))):
                shutil.copyfile(parent_root/name, out/name)
            rows.append(row)
    result = dict(schema=SCHEMA, phase='complete_unapproved_user_source', complete32=True,
        created_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha(Path(__file__)),
        plan_sha256=sha(out/'source-plan.json'), source_model=MODEL, endpoint=ENDPOINT,
        actual_api=parent['actual_api'], actual_requests=0, replacement_actual_requests=1,
        aggregate_actual_requests=42, reused_objects=39, replaced_object_id=OBJECT,
        parent_manifest_sha256=sha(out/'parent-v2/source-audio.json'),
        replacement_take_sha256=sha(out/'replacement-take/take.json'),
        parent_quality_sha256=sha(out/'parent-quality.json'), scenes=parent['scenes'], objects=rows,
        original_WAVs_unchanged=True, gain_changed=False, resampled=False, trimmed=False,
        automatic_retry=False, operator_or_answer_annotations_sent=False, owner_conversations_used=False,
        source_quality_approved=False, teacher_collection_completed=False, training_approved=False,
        actual_native_episode_frame_bound_still_required=1100)
    atomic_json(out/'source-audio.json', result)
    verify(out, require_actual=require_actual)
    return result


def verify(root, *, require_actual=True):
    root = Path(root); r = json.loads((root/'source-audio.json').read_text())
    if r.get('schema') != SCHEMA: return parent_verify(root, require_actual=require_actual)
    parent, plan, changed = replacement(root/'parent-v2', root/'replacement-take',
                                        root/'parent-quality.json', require_actual=require_actual)
    objects, scenes = requests_for(plan)
    expected_rows = [changed if row['object_id']==OBJECT else row for row in parent['objects']]
    if (r['phase'] != 'complete_unapproved_user_source' or r['complete32'] is not True
            or r['code_sha256'] != sha(Path(__file__)) or r['plan_sha256'] != sha(root/'source-plan.json')
            or (root/'source-plan.json').read_bytes() != (root/'parent-v2/source-plan.json').read_bytes()
            or r['source_model'] != MODEL or r['endpoint'] != ENDPOINT or r['scenes'] != scenes
            or r['objects'] != expected_rows or r['actual_api'] is not parent['actual_api']
            or (r['actual_requests'],r['replacement_actual_requests'],r['aggregate_actual_requests'],r['reused_objects']) != (0,1,42,39)
            or r['replaced_object_id'] != OBJECT
            or r['parent_manifest_sha256'] != sha(root/'parent-v2/source-audio.json')
            or r['replacement_take_sha256'] != sha(root/'replacement-take/take.json')
            or r['parent_quality_sha256'] != sha(root/'parent-quality.json')
            or any(r[k] is not False for k in ('gain_changed','resampled','trimmed','automatic_retry',
                'operator_or_answer_annotations_sent','owner_conversations_used','source_quality_approved',
                'teacher_collection_completed','training_approved')) or r['original_WAVs_unchanged'] is not True
            or r['actual_native_episode_frame_bound_still_required'] != 1100):
        raise ValueError('Complete40 one-object repair and immutable original history required')
    for index, row in enumerate(r['objects']):
        check_object(root, row, objects[row['object_id']], attempt=1 if index==16 or row['object_id']==OBJECT else 0,
                     actual=r['actual_api'])
    expected = {'source-plan.json','source-audio.json','parent-quality.json'}
    expected |= {'parent-v2/'+p for p in files(root/'parent-v2')}
    expected |= {'replacement-take/'+p for p in TAKE_FILES}
    for row in expected_rows: expected.update((row['wav'],str(Path(row['wav']).with_suffix('.receipt.json'))))
    if files(root) != expected: raise ValueError('Unexpected or missing complete repair inventory')
    return r, plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','take','quality','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();prepare(a.parent,a.take,a.quality,a.out)
