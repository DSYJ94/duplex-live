"""Bounded complete40 post-collection chain; never provision or train a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from data.synthetic.native_control_render import sha256
from evals.native_control_teacher import atomic_json, collection_lock


def stages(raw, source_audit, batch, reuse_batch, reuse_quality, out, weights, env_file, exact_teacher=False):
    quality = out / 'quality/teacher-quality.json'
    semantic = out / 'semantic/semantic.json'
    common = ['--raw', str(raw), '--source-audit', str(source_audit), '--batch', str(batch)]
    quality_module = ('evals.native_control_teacher_exact_quality' if exact_teacher
                      else 'evals.native_control_teacher_reaudit')
    yield ('quality40', quality_module, common + [
        '--out', str(quality.parent), '--reuse-batch', str(reuse_batch),
        '--reuse-quality', str(reuse_quality), '--weights', str(weights)], 7200)
    yield ('verify_quality40', quality_module, common + ['--verify', str(quality)], 1800)
    yield ('semantic40', 'evals.native_control_semantic', common + [
        '--quality', str(quality), '--out', str(semantic.parent), '--env-file', str(env_file)], 7200)
    yield ('verify_semantic40', 'evals.native_control_semantic', common + [
        '--quality', str(quality), '--verify', str(semantic)], 1800)
    yield ('convert40', 'evals.native_control_convert', common + [
        '--quality', str(quality), '--semantic', str(semantic), '--out', str(out / 'converted')], 3600)
    yield ('verify_conversion40', 'evals.native_control_convert', ['--verify', str(out / 'converted')], 1800)


def run(raw, source_audit, batch, reuse_batch, reuse_quality, out, weights, env_file, lock, exact_teacher=False):
    # The caller must first observe its collection handle finish. The controller
    # independently validates the sealed40 on stage1; no generation or retry lives here.
    manifest = json.loads((batch / 'teacher.json').read_text())
    if (manifest['status'] != 'raw_complete_unapproved' or len(manifest['recordings']) != 40
            or not all(r['transport_complete'] for r in manifest['recordings'])):
        raise ValueError('Complete40 actual raw collection required before downstream work')
    with collection_lock(lock):
        out.mkdir(parents=True, exist_ok=False)
        state = dict(schema='dl-control3-postteacher-pipeline-v1',
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     code_sha256=sha256(Path(__file__)), started_utc=datetime.now(timezone.utc).isoformat(),
                     maximum_wallclock_s=6 * 3600, teacher_manifest_sha256=sha256(batch / 'teacher.json'),
                     batch=str(batch), phase='starting', stages=[], training_approved=False,
                     optimizer_updates=0, new_gpu_submitted=False, credentials_exported=False)
        state['exact_teacher_protocol'] = exact_teacher
        target = out / 'pipeline.json'
        atomic_json(target, state)
        start = time.monotonic()
        try:
            for name, module, args, bound in stages(raw, source_audit, batch, reuse_batch, reuse_quality, out, weights, env_file, exact_teacher):
                if sha256(batch / 'teacher.json') != state['teacher_manifest_sha256']:
                    raise ValueError('Sealed collection changed between independent stages')
                remaining = state['maximum_wallclock_s'] - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Pipeline deadline reached before another stage')
                command = [sys.executable, '-u', '-m', module, *args]
                row = dict(stage=name, command=command, status='running', started_utc=datetime.now(timezone.utc).isoformat())
                state['phase'] = name
                state['stages'].append(row)
                atomic_json(target, state)
                print(json.dumps(dict(stage=name, status='running')), flush=True)
                with (out / (name + '.log')).open('x') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=min(bound, remaining), check=False)
                row.update(status='exit0' if result.returncode == 0 else 'failed', returncode=result.returncode,
                           finished_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(target, state)
                if result.returncode:
                    raise RuntimeError('Upstream complete40 approval failed; no later stage submitted')
            state['phase'] = 'converted40_native_cache_numerical_and_pilot_pending'
        except Exception as error:
            state['failed_phase'] = state['phase']
            state['phase'] = 'stopped_preserve_all_artifacts'
            state['error_type'] = type(error).__name__
            raise
        finally:
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(target, state)
        return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'source-audit', 'batch', 'reuse-batch', 'reuse-quality', 'out', 'weights', 'env-file'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--lock', type=Path, default=Path('runs/local-migration-20261021/.private/control3-pipeline.lock'))
    ap.add_argument('--exact-teacher', action='store_true')
    args = ap.parse_args()
    result = run(args.raw, args.source_audit, args.batch, args.reuse_batch, args.reuse_quality,
                 args.out, args.weights, args.env_file, args.lock, args.exact_teacher)
    print(json.dumps(dict(phase=result['phase'], training_approved=False, optimizer_updates=0)))


if __name__ == '__main__':
    main()
