"""Versioned context teachers: original audio, measured speech, real work IDs.

Dialogue scheduling is authored, not measured provider latency. Local work
receipts witness the outcomes; their monotonic clocks are never relabelled as
native audio clocks. Native replay/cache approval remains a separate gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly
from tokenizers import Tokenizer

from .context_source_quality import inventory
from .context_alignment import PAYLOAD, ROOT, NUMERIC_ID

BASE = Path('runs/venus-training-20261101')
PLAN = Path('evals/fixtures/venus-context-repair-authoring-v2-20261101.json')
DRAFT = Path('evals/fixtures/venus-context-teacher-draft-v1-20261101.json')
TOKENIZER = Path('runs/venus-training-20261029/action-target-audit-v1/tokenizer.json')
FAMILIES = dict(quiet_scope='quiet', simple_or_delegate='delegate', selective_cancel='cancel', correction='delegate', pause_resume='pause', result_delivery='pending')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prerequisites():
    user = json.loads((BASE/'context-word-derived-v2.json').read_text())
    assistant = json.loads((BASE/'source-audit-v47/copyback/quality/report.json').read_text())
    alignment = json.loads((BASE/'context-alignment-v1/report.json').read_text())
    semantic = json.loads((BASE/'context-semantic-v1/report.json').read_text())
    if not (user['all72_pass'] and assistant['all34_pass'] and alignment['all34_exact_word_timings'] and semantic['all36_semantic_pass']):
        raise ValueError('Complete source/word/timing/semantic evidence required')
    if len(assistant['rows']) != 34 or len(alignment['rows']) != 34 or len(user['rows']) != 72:
        raise ValueError('No passing subset')
    original = inventory(ROOT, PAYLOAD, 34)
    for r in original:
        a = next(x for x in alignment['rows'] if x['id'] == r['id'])
        q = next(x for x in assistant['rows'] if x['id'] == r['id'])
        if a['source_sha256'] != r['sha256'] or q['sha256'] != r['sha256'] or a['text'] != r['text']:
            raise ValueError('Original assistant source identity changed')
    numeric = json.loads((BASE/'context-alignment-v1/numeric-diagnostic.json').read_text())
    if '£3,814.41' not in numeric['text'] or '£3,814.041' in numeric['text']:
        raise ValueError('Independent numerical diagnostic requires review')
    # Diagnostic is not a fresh complete34 reapproval or a new word gate.
    local = inventory(BASE/'context-source-v1', Path('docs/NATIVE-VENUS-CONTEXT-REPAIR-SPEECH-PAYLOAD-2026-11-01.json'))
    users = {(r['fold'], r['text']): r for r in local}
    for r in user['rows']:
        if users[r['fold'], r['text']]['sha256'] != r['sha256']:
            raise ValueError('Original user recording changed')
    return users, {(r['fold'], r['text']): r for r in alignment['rows']}


def compile_scene(scene, draft, receipt, users, assistant, tokenizer):
    actions, feedback, turns = [], [], []
    cursor = 0
    occupied = 0
    pending = {}
    results = {r['work_id']: r for r in receipt['all_worker_receipts']}
    dispatch = [e for e in receipt['events'] if e['kind'] == 'teacher_forced_dispatch']
    dispatch_cursor = 0
    def add_speech(frame, text, speech=None):
        source = assistant[scene['fold'], text]
        speech = [dict(u) for u in (speech or source['native_speech_units'])]
        if any(len(tokenizer.encode(u['text'], add_special_tokens=False).ids) > 16 for u in speech):
            raise ValueError('Measured speech exceeds released native capacity')
        actions.append(dict(frame=frame, kind='speak', text=text, final=True, native_speech_units=speech, timing_source_sha256=source['source_sha256']))
        return frame + len(speech)
    for turn_index, text in enumerate(scene['user_turns']):
        source = users[scene['fold'], text]
        start = cursor
        if scene['family'] == 'pause_resume' and turn_index == 1:
            start = actions[0]['frame'] + 1
        turns.append(dict(text=text, start_seconds=start, seconds=source['seconds'], audio_id=source['id'], sha256=source['sha256']))
        user_end = math.ceil(start + source['seconds'])
        frame = max(user_end, occupied)
        for a in [a for a in draft['actions'] if a['after_user_turn'] == turn_index]:
            kind = a['kind']
            if kind in ('delegate', 'replace', 'cancel'):
                action_text = a['text']
                n = len(tokenizer.encode('<delegate>' + action_text + '</delegate>', add_special_tokens=False).ids)
                length = math.ceil(n / 16)
                actions.append(dict(frame=frame, kind='delegate', text=action_text, final=True, teacher_operation=kind))
                frame += length
                label = a['task_label']
                if kind in ('cancel', 'replace'):
                    retired = pending[label]
                    if not any(e['kind'] == 'delivery_retired' and e['work_id'] == retired for e in receipt['events']):
                        raise ValueError('Actual retirement receipt required')
                    feedback.append(dict(frame=frame, text=json.dumps(dict(delivery_retired=retired, remote_execution_aborted=False), sort_keys=True), work_id=retired))
                    frame += 1
                if kind != 'cancel':
                    event = dispatch[dispatch_cursor]
                    dispatch_cursor += 1
                    if event['task_label'] != label or event['revision'] != a.get('revision', 1):
                        raise ValueError('Actual task/revision differs')
                    pending[label] = event['work_id']
            elif kind == 'pause':
                first = actions[0]
                pause = first['frame'] + 2  # One native frame after actual pause audio begins.
                full = first['native_speech_units']
                cut = pause - first['frame']
                if not 0 < cut < len(full):
                    raise ValueError('A real partial spoken prefix and unsaid tail required')
                first['original_teacher_text'] = first['text']
                first['native_speech_units'] = [dict(u, final=False) for u in full[:cut]]
                # Unit-final closes this source phrase; action-final controls
                # turn EOS. The paused phrase must remain an unfinished turn
                # until the separate measured end_turn action, as native
                # schedule() already requires for all original teachers.
                first['native_speech_units'][-1]['final'] = True
                first['text'] = ' '.join(u['text'] for u in full[:cut])
                first['final'] = False
                first['paused_tail'] = full[cut:]
                actions.append(dict(frame=pause, kind='end_turn', text='', final=True))
                occupied = pause + 1
                frame = user_end
            elif kind == 'speak':
                if 'requires_result_for' in a:
                    ident = pending[a['requires_result_for']]
                    if ident not in {r['work_id'] for r in receipt['delivered_receipts']}:
                        raise ValueError('Only actually delivered current results allowed')
                    result = results[ident]
                    if result['revision'] != a.get('revision', 1):
                        raise ValueError('Old revision cannot supervise current speech')
                    feedback.append(dict(frame=frame, text=json.dumps(result, sort_keys=True), work_id=ident, receipt_sha256=result['sha256']))
                    frame += 1
                if a.get('resume_without_restart'):
                    current = assistant[scene['fold'], a['text']]
                    combined = actions[0]['paused_tail'] + current['native_speech_units']
                    combined = [dict(u, offset=i, final=i == len(combined)-1) for i, u in enumerate(combined)]
                    resumed = ' '.join(u['text'] for u in combined)
                    actions.append(dict(frame=frame, kind='speak', text=resumed, final=True, native_speech_units=combined, resumes_unsaid_tail=True, timing_sources_sha256=[actions[0]['timing_source_sha256'], current['source_sha256']]))
                    frame += len(combined)
                else:
                    frame = add_speech(frame, a['text'])
            else:
                raise ValueError('Unrecognized teacher operation')
        occupied = max(occupied, frame)
        cursor = max(user_end, frame) + 3
    if dispatch_cursor != len(dispatch):
        raise ValueError('All actual dispatch identities must be accounted for')
    return dict(scene_id=scene['id'], fold=scene['fold'], family=FAMILIES[scene['family']], context_family=scene['family'], frames=max(cursor, occupied)+2, user_turns=turns, actions=actions, backend_feedback=feedback, teacher_forced=True, serving_oracle_dispatch=False, training_approved=False, dialogue_clock='authored_synthetic_native_frames', worker_clock='separate_actual_monotonic_not_relabelled', worker_receipt_sha256=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest(), native_speech_timing_schema='word-aligned-native-v1')


def main(output):
    if output.exists():
        raise ValueError('Preserve previous timelines')
    users, assistant = prerequisites()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    plan = json.loads(PLAN.read_text())
    drafts = {s['id']: s for s in json.loads(DRAFT.read_text())['scenes']}
    output.mkdir(parents=True)
    rows = []
    for scene in plan['scenes']:
        path = BASE/'context-teacher-work-v1'/(scene['id']+'.json')
        receipt = json.loads(path.read_text())
        timeline = compile_scene(scene, drafts[scene['id']], receipt, users, assistant, tokenizer)
        pcm = np.zeros(timeline['frames']*16000, dtype=np.int16)
        for turn in timeline['user_turns']:
            source = users[scene['fold'], turn['text']]
            with wave.open(source['wav']) as w:
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
            # Sample-rate conversion only, no gain change. Originals remain24k.
            converted = np.rint(resample_poly(raw, 2, 3))
            if np.max(np.abs(converted)) / 32768 > .995:
                raise ValueError('Resampled source clips; never attenuate or clamp it')
            values = converted.astype('<i2')
            start = turn['start_seconds']*16000
            if np.any(pcm[start:start+len(values)]):
                raise ValueError('Unplanned user overlap')
            pcm[start:start+len(values)] = values
        folder = output/scene['id']
        folder.mkdir()
        with wave.open(str(folder/'user.wav'), 'wb') as w:
            w.setparams((1,2,16000,0,'NONE','not compressed'))
            w.writeframes(pcm.tobytes())
        timeline['user_wav_sha256'] = sha(folder/'user.wav')
        (folder/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
        (folder/'real-worker.json').write_bytes(path.read_bytes())
        rows.append(dict(id=scene['id'], fold=scene['fold'], family=timeline['family'], frames=timeline['frames'], timeline_sha256=sha(folder/'timeline.json'), user_wav_sha256=timeline['user_wav_sha256']))
    report = dict(schema='dl-native-context-timelines-v1', scenes=rows, train=24, dev=12, training_approved=False, native_replay_pending=True, native_speech_timing_schema='word-aligned-native-v1', source_plan_sha256=sha(PLAN), draft_sha256=sha(DRAFT), tokenizer_sha256=sha(TOKENIZER), independent_semantic_sha256=sha(BASE/'context-semantic-v1/report.json'), alignment_sha256=sha(BASE/'context-alignment-v1/report.json'), original_user_word_review_sha256=sha(BASE/'context-word-derived-v2.json'), original_assistant_word_review_sha256=sha(BASE/'source-audit-v47/copyback/quality/report.json'), source_audio_preserved=True, per_word_timing_interpolation=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(scenes=len(rows), frames=sum(r['frames'] for r in rows), native_replay_pending=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
