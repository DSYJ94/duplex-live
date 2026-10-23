"""ONE diagnosed001 recapture: unchanged sources/prompt, quiescent recorder.

This is component evidence only. It neither edits nor approves the whole corpus.
Original32 failed speech and all previous recordings remain untouched.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evals import native_reasoning_teacher_quality as old_quality
from evals import native_reasoning_teacher_v7 as recorder
from evals.native_control_teacher import atomic_json, collection_lock
from evals.native_reasoning_source import sha
from evals.native_reasoning_teacher import geometry
from evals.native_reasoning_teacher_lifecycle_review import case_review

CASE = 'native-reasoning4-train-001'
PARENT_SHA = 'a958eac2d5b2634d62010be3b35bd48e996ea5d1607ec79aa3b086256770ec5a'


async def probe(source, quality, parent, failed, out, key, engine, tokenizer_path):
    with collection_lock(out.parent / '.private/native-reasoning32-teacher.lock'):
        previous = old_quality.verify(source, quality, parent, engine, tokenizer_path, failed, require_pass=False)
        if sha(parent / 'teacher-collection.json') != PARENT_SHA:
            raise ValueError('Exact failed preserved whole32 parent required')
        row = next(r for r in previous['results'] if r['id'] == CASE)
        text = old_quality.expected_output(parent / CASE)
        if row['passed'] is not False or text != 'Okay. Eight tokens. Badge, candle, and poster; total cost 16 tokens; total':
            raise ValueError('Exact independently failed truncated001 required')
        _, manifest, plan = old_quality.whole_collection(source, quality, parent, engine, tokenizer_path)
        scene = next(s for s in manifest['scenes'] if s['id'] == CASE)
        case = next(r for r in plan['cases'] if r['id'] == CASE)
        tokenizer = geometry(engine, tokenizer_path)
        out.mkdir(parents=True, exist_ok=False)
        code = dict(recorder.code_hashes(), **{Path(__file__).name: sha(Path(__file__))})
        value = dict(schema='dl-native-reasoning32-output-close-component-v1',
                     at_utc=datetime.now(timezone.utc).isoformat(),
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     code_hashes=code, parent_collection_sha256=PARENT_SHA,
                     failed_whole32_speech_sha256=sha(failed),
                     source_manifest_sha256=sha(source / 'source-audio.json'), source_quality_sha256=sha(quality),
                     case_id=CASE, maximum_new_voice_sessions=1, maximum_concurrent_sessions=1,
                     prior_voice_attempts=38, maximum_aggregate_voice_attempts=39,
                     prior_backend_request_upper_bound=73, maximum_backend_requests=6,
                     maximum_aggregate_backend_request_upper_bound=79,
                     automatic_retry=False, diagnostic_only=True,
                     teacher_approved=False, training_approved=False, status='capturing_one_diagnosed_case')
        target = out / 'cutoff-component.json'; atomic_json(target, value)
        try:
            record = await recorder.capture_one(source, scene, case, out / CASE, sha(quality), 'gpt-6-luna', key, tokenizer)
            value['recording'] = record; atomic_json(target, value)
            recorder.verify_record(source, scene, out / CASE, record, tokenizer)
            lifecycle = case_review(case, json.loads((out / CASE / 'lifecycle.json').read_text()))
            value['lifecycle_diagnostic'] = lifecycle
            if (record['frame_count_diagnostic_pass'] is not True or lifecycle['structural_pass'] is not True
                    or len(json.loads((out / CASE / 'lifecycle.json').read_text())['tasks']) > 6):
                raise ValueError('Actual unchanged frame/lifecycle/backend bound failed')
            if code != dict(recorder.code_hashes(), **{Path(__file__).name: sha(Path(__file__))}):
                raise ValueError('Recorder implementation changed')
            value['status'] = 'raw_component_complete_independent_speech_and_semantics_required'
        except Exception as error:
            value.update(status='failed_preserved_no_retry', error_type=type(error).__name__, reason=str(error))
        finally:
            atomic_json(target, value)
        return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'quality', 'parent', 'failed', 'out', 'key', 'engine', 'tokenizer'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    r = asyncio.run(probe(a.source, a.quality, a.parent, a.failed, a.out, a.key, a.engine, a.tokenizer))
    print(json.dumps(dict(status=r['status'], teacher_approved=False, training_approved=False)))
    if r['status'] == 'failed_preserved_no_retry': raise SystemExit(1)
