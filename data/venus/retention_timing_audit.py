"""Reproduce measured teacher timing without substituting a semantic verdict."""
import hashlib
import json
from pathlib import Path
from .pacing import units


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_action(action, aligned):
    if (action['text'] != aligned['text']
            or action['timing_source_sha256'] != aligned['source_sha256']):
        raise ValueError('Exact authored words and original aligned audio required')
    measured = [dict(text=w['text'], timestamp=[w['start'], w['end']])
                for w in aligned['measured_words']]
    expected = units(action['text'], measured)
    if expected != action['native_speech_units']:
        raise ValueError('Native units differ from measured word onsets and ends')
    last = next(w for w in reversed(aligned['measured_words']) if w['text'].strip())
    return dict(source_sha256=aligned['source_sha256'], frame=action['frame'],
                unit_count=len(expected), empty_final=expected[-1]['text'] == '',
                last_word=last, exact_measured_reconstruction=True)


def audit(root, alignment, *, manifest_sha, alignment_sha):
    root, alignment = Path(root), Path(alignment)
    if sha(root/'manifest.json') != manifest_sha or sha(alignment) != alignment_sha:
        raise ValueError('Pinned original timeline and measured-alignment manifests required')
    manifest = json.loads((root/'manifest.json').read_text())
    ar = json.loads(alignment.read_text())
    if (not ar['complete'] or not ar['all73_exact_authored_word_alignments']
            or len(ar['rows']) != 73 or not manifest['complete96']
            or len(manifest['scenes']) != 96 or manifest['train'] != 96 or manifest['dev'] != 0
            or len({r['id'] for r in manifest['scenes']}) != 96):
        raise ValueError('Whole original96 teachers and73 measured alignments required')
    bysha = {r['source_sha256']: r for r in ar['rows']}
    rows = []
    for row in manifest['scenes']:
        path = root/row['id']/'timeline.json'
        if sha(path) != row['timeline_sha256']:
            raise ValueError('Original timeline changed')
        t = json.loads(path.read_text())
        if t['scene_id'] != row['id'] or t['fold'] != 'TRAIN':
            raise ValueError('Original TRAIN identity required')
        actions = [verify_action(a, bysha[a['timing_source_sha256']])
                   for a in t['actions'] if a['kind'] == 'speak']
        rows.append(dict(id=row['id'], timeline_sha256=row['timeline_sha256'], speak_actions=actions))
    return dict(status='ALL96_TIMING_RECONSTRUCTION_PASS_NOT_SEMANTIC_APPROVAL',
                manifest_sha256=manifest_sha, alignment_sha256=alignment_sha, scenes=len(rows),
                speak_actions=sum(len(r['speak_actions']) for r in rows),
                empty_final_actions=sum(a['empty_final'] for r in rows for a in r['speak_actions']),
                scenes_with_empty_final=sum(any(a['empty_final'] for a in r['speak_actions']) for r in rows),
                all_original_audio_words_timings_unchanged=True,
                independent_semantic_review_pending=True, training_approved=False, rows=rows)
