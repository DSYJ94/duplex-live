"""Verify raw actual TRAIN conditioning; never approve it as target labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from evals.moshi_model_download import sha256


def validate(folder, data, assets, source_commit):
    import torch
    from evals.recovery_data import validate_conversion
    report = json.loads((folder / 'report.json').read_text())
    record = json.loads((folder / 'student-history.json').read_text())
    source = record['source']
    if (record['schema'] not in ('dl-context-student-history-v1','dl-context-student-history-v2')
            or record['source_split'] != 'train'
            or source['split'] != 'train' or report['source_split'] != 'train'
            or record['training_approved'] is not False or report['training_approved'] is not False
            or record['conditioning_only'] is not True or record['past_labels_must_all_be_masked'] is not True
            or record['generated_reply_tokens_forced'] or report['word_or_acoustic_tokens_forced']
            or report['response_start_forced'] or report['scheduled_dispatch']
            or not report['actual_lm_and_pcm_executed_together']):
        raise RuntimeError('History scope is not unapproved actual TRAIN conditioning')
    if not re.fullmatch('[a-f0-9]{40}', source_commit) or source['repo_commit'] != source_commit:
        raise RuntimeError('Unexpected collector source commit')
    repo = Path(__file__).resolve().parents[1]
    for file, key in [('context_student_history.py', 'recorder_code_sha256'),
                      ('context_spanning_probe.py', 'probe_code_sha256'),
                      ('context_spanning_native.py', 'native_run_code_sha256')]:
        content = subprocess.check_output(['git', 'show', source_commit + ':evals/' + file], cwd=repo)
        if hashlib.sha256(content).hexdigest() != source[key]:
            raise RuntimeError('Collector source hash differs from the specified immutable commit')
    receipt = validate_conversion(data)
    row = next(r for r in receipt['accepted'] if r['scenario_id'] == source['scenario'])
    if (row['split'] != 'train' or source['conversion'] != str(data)
            or source['ingestion_sha256'] != sha256(data / 'ingestion.json')
            or any(source[k] != row[k] for k in ('wav_sha256', 'sidecar_sha256', 'kind'))
            or source['assets_manifest_sha256'] != sha256(assets)
            or report['loading']['assets_manifest_sha256'] != sha256(assets)
            or source['voice_codes_sha256'] != sha256(folder / 'student-voice.pt')):
        raise RuntimeError('Approved source, model or actual voice provenance changed')
    summary = report['student_history']
    if (summary['file'] != 'student-history.json' or record['file'] != 'student-history.pt'
            or summary['sha256'] != sha256(folder / summary['file'])
            or record['sha256'] != sha256(folder / record['file'])
            or summary['history_sha256'] != record['sha256']
            or report['audio_sha256'] != sha256(folder / 'model.wav')):
        raise RuntimeError('Actual history or recording hash changed')
    saved = torch.load(folder / record['file'], map_location='cpu', weights_only=True)
    inputs = saved['delayed_inputs']
    keys = {'delayed_inputs', 'states'} if record['schema'].endswith('v2') else {'delayed_inputs'}
    if (set(saved) != keys or list(inputs.shape) != record['shape']
            or inputs.shape[:2] != (1, 17) or inputs.dtype != torch.int64
            or (inputs < -1).any() or (inputs[:, 0] > 32000).any() or (inputs[:, 1:] > 2048).any()):
        raise RuntimeError('Invalid actual delayed conditioning tensor')
    calls = record['calls']
    cursor = 0
    for index, call in enumerate(calls):
        if (call['start'] != cursor or call['frames'] < 1
                or (call['mode'], call['path']) not in (('prefix', 'direct'), ('span', 'direct'), ('live', 'graph'))
                or (call['mode'] == 'prefix') != (index == 0)
                or (call['mode'] == 'live' and call['frames'] != 1)):
            raise RuntimeError('Noncontiguous or unexpected native forward accounting')
        cursor += call['frames']
    rows = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    live = [c for c in calls if c['mode'] == 'live']
    events = json.loads((folder / 'worker-events.json').read_text())
    spans = [e for e in events if e.get('applied_at_s') is not None and 'virtual_span_frames' in e]
    if (cursor != inputs.shape[2] or summary['frames'] != cursor or len(live) != len(rows)
            or events != report['worker_events']
            or [e['virtual_span_frames'] for e in spans] != [c['frames'] for c in calls if c['mode'] == 'span']
            or sum(r['input_samples'] for r in rows) != report['real_user_samples_expected']
            or report['real_user_samples_processed'] != report['real_user_samples_expected']):
        raise RuntimeError('Actual input, receipt or virtual-frame accounting changed')
    for call, row in zip(live, rows):
        if (call['start'] != row['history_virtual_frames_before_step']
                or call['start'] + 1 != row['history_virtual_frames_after_step']):
            raise RuntimeError('Real input frame no longer maps to its actual native forward')
    if record['schema'].endswith('v2'):
        if len(saved['states']) != len(rows) or not record['snapshot_after_every_real_frame']:
            raise RuntimeError('Incomplete actual generator snapshots')
        for state, row in zip(saved['states'], rows):
            if (state['virtual_frames'] != row['history_virtual_frames_after_step']
                    or state['offset'] != state['virtual_frames'] + 1
                    or state['cache'].shape != state['provided'].shape
                    or state['cache'].shape[:2] != (1, 17) or state['cache'].dtype != torch.int64
                    or state['provided'].dtype != torch.bool):
                raise RuntimeError('Invalid native generator state or clock')
            pos = state['offset'] % state['cache'].shape[2]
            if state['provided'][:, 1:9, pos].any() or int(state['provided'][:, 9:17, pos].sum()) != 7:
                raise RuntimeError('Unobserved assistant acoustics were marked provided')
    return {'scenario': source['scenario'], 'shape': record['shape'], 'live_forwards': len(live),
        'actual_spans': len(spans), 'history_sha256': record['sha256'],
        'report_sha256': sha256(folder / 'report.json'), 'raw_integrity_passed': True,
        'training_approved': False, 'future_targets_approved': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--history', type=Path, action='append', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--source-commit', required=True)
    args = ap.parse_args()
    print(json.dumps({'schema': 'dl-context-history-integrity-v1', 'results': [
        validate(folder, args.data, args.assets, args.source_commit) for folder in args.history]}, indent=2))


if __name__ == '__main__': main()
