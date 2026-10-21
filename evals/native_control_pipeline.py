"""Follow ONE existing source collection through complete independent approvals.

Never launches generation, compute, conversion or optimizer work. Each downstream
API stage requires the complete upstream verifier, not merely a zero CLI exit.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from data.synthetic.native_control_render import sha256
from evals.native_control_teacher import atomic_json, collection_lock, local_key


def generation_complete(generation, plan):
    try:
        ledger = json.loads((generation / 'generation.json').read_text())
    except json.JSONDecodeError:
        # The existing collector writes a live progress snapshot. A partial
        # read during that write cannot authorize anything; retry within bound.
        return False
    if ledger['status'] == 'collecting_unapproved':
        return False
    if (ledger['status'] != 'complete_raw_unapproved' or len(ledger['clips']) != 168
            or ledger.get('failures') or ledger['plan_sha256'] != sha256(plan)
            or ledger['training_approved'] is not False
            or ledger['implementation_sha256'] != sha256(Path('data/synthetic/native_control_generate.py'))
            or ledger['speech_helper_sha256'] != sha256(Path('data/synthetic/recovery_curriculum.py'))):
        raise ValueError('Existing source generation is failed, incomplete or changed; no downstream submissions')
    if ledger.get('new_speech_helper_sha256'):
        if (ledger.get('new_clip_model') != 'tts-1-hd'
                or ledger['new_speech_helper_sha256'] != sha256(Path('evals/native_control_tts1hd.py'))):
            raise ValueError('Explicit repair synthesizer provenance changed')
    if 'repair_provenance' in ledger:
        from data.synthetic.native_control_generate import retained_source, source_tasks
        parent = Path(ledger['repair_provenance']['parent'])
        extra = {}
        if ledger['repair_provenance'].get('repair_kind') == 'complete-word-audit-v1':
            value = ledger['repair_provenance']
            extra = dict(raw=Path(value['parent_raw']), word_audit=Path(value['word_audit']),
                         diagnostic=Path(value['independent_diagnostic']))
        retained, provenance = retained_source(parent, plan, source_tasks(json.loads(plan.read_text())), **extra)
        if provenance != ledger['repair_provenance']:
            raise ValueError('Changed failed-source repair provenance')
        for row in retained:
            wav = Path(row['path'])
            if (sha256(generation / wav) != row['waveform_sha256']
                    or sha256(generation / wav.with_suffix('.receipt.json')) != row['receipt_sha256']):
                raise ValueError('A retained original source clip or receipt changed in the repair')
    return True


def stages(plan, generation, out, weights, env_file, reuse_raw=None, reuse_audit=None):
    raw, source, teachers = out / 'raw-source', out / 'source-audit/source-word-audit.json', out / 'teachers'
    quality, semantic = out / 'teacher-audit/teacher-quality.json', out / 'semantic/semantic.json'
    source_args = ['--raw', str(raw), '--source-audit', str(source)]
    teacher_args = source_args + ['--batch', str(teachers)]
    cached = ['--reuse-raw', str(reuse_raw), '--reuse-audit', str(reuse_audit)] if reuse_raw and reuse_audit else []
    return [
        ('assemble40', 'data.synthetic.native_control_render',
         ['--plan', str(plan), '--clips', str(generation / 'clips'), '--out', str(raw)], 300),
        ('source_asr168', 'evals.native_control_source_quality',
         ['--raw', str(raw), '--weights', str(weights), '--out', str(source.parent), '--threads', '4'] + cached, 7200),
        ('verify_source168', 'evals.native_control_source_quality', ['--raw', str(raw), '--verify', str(source)], 300),
        ('capture_teachers40', 'evals.native_control_teacher',
         source_args + ['--out', str(teachers), '--env-file', str(env_file)], 4800),
        ('teacher_asr40', 'evals.native_control_teacher_quality',
         teacher_args + ['--weights', str(weights), '--out', str(quality.parent), '--threads', '4'], 7200),
        ('verify_teacher40', 'evals.native_control_teacher_quality', teacher_args + ['--verify', str(quality)], 300),
        ('semantic40', 'evals.native_control_semantic',
         teacher_args + ['--quality', str(quality), '--out', str(semantic.parent), '--env-file', str(env_file)], 4800),
        ('verify_semantic40', 'evals.native_control_semantic',
         teacher_args + ['--quality', str(quality), '--verify', str(semantic)], 300),
    ]


def run(plan, generation, out, weights, env_file, lock, reuse_raw=None, reuse_audit=None):
    # Preflight all dependencies and key presence without network or key output.
    import requests
    import soundfile
    import torch
    import whisper_timestamped
    from websockets.asyncio.client import connect
    local_key(env_file)
    if bool(reuse_raw) != bool(reuse_audit):
        raise ValueError('Cached ASR needs both complete original source and audit')
    with collection_lock(lock):
        out.mkdir(parents=True, exist_ok=False)
        state = dict(schema='dl-native-control3-pipeline-v1',
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     code_sha256=sha256(Path(__file__)), generation=str(generation),
                     started_utc=datetime.now(timezone.utc).isoformat(),
                     maximum_wallclock_s=6 * 3600, source_wait_bound_s=2 * 3600,
                     phase='waiting_existing_source', stages=[], training_approved=False,
                     new_gpu_submitted=False, credentials_exported=False)
        target = out / 'pipeline.json'
        atomic_json(target, state)
        started = time.monotonic()
        try:
            while not generation_complete(generation, plan):
                if time.monotonic() - started >= state['source_wait_bound_s']:
                    raise TimeoutError('Existing generation still incomplete at bounded wait deadline')
                time.sleep(5)
            for name, module, arguments, limit in stages(plan, generation, out, weights, env_file, reuse_raw, reuse_audit):
                remaining = state['maximum_wallclock_s'] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('Pipeline deadline reached before another stage could start')
                state['phase'] = name
                command = [sys.executable, '-u', '-m', module, *arguments]
                row = dict(stage=name, command=command,
                           started_utc=datetime.now(timezone.utc).isoformat(), status='running')
                state['stages'].append(row)
                atomic_json(target, state)
                print(json.dumps(dict(stage=name, status='running')), flush=True)
                with (out / (name + '.log')).open('x') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=min(limit, remaining), check=False)
                row.update(returncode=result.returncode,
                           finished_utc=datetime.now(timezone.utc).isoformat(),
                           status='exit0' if result.returncode == 0 else 'failed')
                atomic_json(target, state)
                if result.returncode:
                    raise RuntimeError('Upstream collection or approval failed; no subsequent submission')
            state['phase'] = 'semantic_complete_conversion_and_native_approvals_pending'
        except Exception as exc:
            state['failed_phase'] = state['phase']
            state['phase'] = 'stopped_preserve_all_artifacts'
            state['error_type'] = type(exc).__name__
            raise
        finally:
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(target, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'generation', 'out', 'weights', 'env-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--lock', type=Path,
                        default=Path('runs/local-migration-20261021/.private/control3-pipeline.lock'))
    parser.add_argument('--reuse-raw', type=Path)
    parser.add_argument('--reuse-audit', type=Path)
    args = parser.parse_args()
    result = run(args.plan, args.generation, args.out, args.weights, args.env_file, args.lock,
                 args.reuse_raw, args.reuse_audit)
    print(json.dumps(dict(phase=result['phase'], training_approved=False)))


if __name__ == '__main__':
    main()
