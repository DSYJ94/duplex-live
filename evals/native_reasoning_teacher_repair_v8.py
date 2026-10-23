"""One bounded attempt for ONLY six independently diagnosed failed teachers.

001 already has its preserved quiescent replacement and independent speech pass.
The passing25 are frozen. This component collection cannot approve any corpus.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evals import native_reasoning_teacher_v8 as recorder
from evals.native_control_teacher import atomic_json, collection_lock
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_pacing import verify as source_manifest
from evals.native_reasoning_source_quality_v4 import verify as source_quality
from evals.native_reasoning_teacher import geometry
from evals.native_reasoning_teacher_lifecycle_review import case_review

FAILED = [f'native-reasoning4-train-{i:03d}' for i in (1,5,6,25,26,28,30)]
REPAIR = FAILED[1:]
PARENT = 'a958eac2d5b2634d62010be3b35bd48e996ea5d1607ec79aa3b086256770ec5a'
SPEECH = 'cfe45df2d97aa03b29381c6f0ce8618c28bb1a0a7b441aa5c8e9f655dc34ec01'


def preflight(parent, failed, proof):
    value = json.loads(failed.read_text()); checked = json.loads(proof.read_text())
    if (sha(parent/'teacher-collection.json') != PARENT or sha(failed) != SPEECH
            or value['complete_cases'] != 32 or len(value['results']) != 32
            or value['passed'] is not False
            or [r['id'] for r in value['results'] if not r['passed']] != FAILED
            or checked['phase'] != 'complete_all32_speech_only'
            or checked['original_failed32_reproduced_exactly'] is not True
            or checked['unchanged_input_snapshot'] is not True
            or checked['whole_speech_result_sha256'] != SPEECH
            or checked['source_commit'] != 'f551ca793168bee4f7123e0093ca2abe36420231'
            or checked['runner_sha256'] != '694dc110d29166f6993575b6dec453784512e5e758b2dfa7a46f5908bd16e565'
            or checked['complete_cases'] != 32 or checked['passed_cases'] != 25
            or checked['whole_speech_passed'] is not False):
        raise ValueError('Exact independently recomputed complete32 failure required')
    return value


async def run(a):
    # Canonical repository lock also protects captures spooled outside OneDrive.
    with collection_lock(a.lock):
        preflight(a.parent,a.failed,a.proof)
        source_quality(a.source,a.quality); manifest,plan=source_manifest(a.source)
        scenes={s['id']:s for s in manifest['scenes']}; cases={r['id']:r for r in plan['cases']}
        tokenizer=geometry(a.engine,a.tokenizer)
        a.out.mkdir(parents=True,exist_ok=False)
        code=recorder.code_hashes()
        value=dict(schema='dl-native-reasoning32-diagnosed-repair-components-v8',
                   at_utc=datetime.now(timezone.utc).isoformat(),
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   collector_code_sha256=sha(Path(__file__)),
                   code_hashes=code,original_parent_sha256=PARENT,failed_whole32_speech_sha256=SPEECH,
                   independent_failed_readback_sha256=sha(a.proof),source_manifest_sha256=sha(a.source/'source-audio.json'),
                   source_quality_sha256=sha(a.quality),diagnosed_failed_ids=FAILED,attempt_order=REPAIR,
                   maximum_new_voice_sessions=6,maximum_concurrent_sessions=2,
                   prior_voice_attempts=39,maximum_aggregate_voice_attempts=45,
                   prior_backend_request_upper_bound=79,maximum_backend_requests_per_session=6,
                   maximum_aggregate_backend_request_upper_bound=115,automatic_retry=False,
                   passing25_recaptured=False,originals_or_source_words_changed=False,
                   whole32_approved=False,semantic_approved=False,training_approved=False,
                   attempted_ids=[],recordings=[],failures=[],status='capturing_diagnosed_components')
        def save(): atomic_json(a.out/'repair-components.json',value)
        save(); semaphore=asyncio.Semaphore(2); stop=False
        async def one(identity):
            nonlocal stop
            async with semaphore:
                if stop: return
                value['attempted_ids'].append(identity); save()
                clarity=identity!='native-reasoning4-train-005'
                capture,verify=recorder.implementation(clarity)
                try:
                    record=await capture(a.source,scenes[identity],cases[identity],a.out/identity,
                                         sha(a.quality),'gpt-6-luna',a.key,tokenizer)
                    record.update(recorder_version='v8',clarity_profile=clarity)
                    value['recordings'].append(record); save()
                    verify(a.source,scenes[identity],a.out/identity,record,tokenizer)
                    lifecycle=case_review(cases[identity],json.loads((a.out/identity/'lifecycle.json').read_text()))
                    if (record['frame_count_diagnostic_pass'] is not True
                            or lifecycle['structural_pass'] is not True):
                        atomic_json(a.out/(identity+'-failed-lifecycle.json'),lifecycle)
                        raise ValueError('Full unchanged frame/lifecycle gate failed')
                    print(json.dumps(dict(id=identity,actual_transport_and_lifecycle=True,
                                          independent_speech_required=True)),flush=True)
                except Exception as error:
                    stop=True; value['failures'].append(dict(id=identity,error_type=type(error).__name__,reason=str(error)));save()
        try:
            await asyncio.gather(*(one(identity)for identity in REPAIR))
        finally:
            value['status']=('raw_components_complete_unapproved'if len(value['recordings'])==6
                             and not value['failures']else'incomplete_failed_preserved_no_retry')
            if code != recorder.code_hashes() or value['collector_code_sha256'] != sha(Path(__file__)):
                value['status']='capture_code_changed_failed'
            save()
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','quality','parent','failed','proof','engine','tokenizer','key','lock','out'):
        p.add_argument('--'+name,type=Path,required=True)
    value=asyncio.run(run(p.parse_args()));print(json.dumps(value))
    if value['status']!='raw_components_complete_unapproved':raise SystemExit(1)
