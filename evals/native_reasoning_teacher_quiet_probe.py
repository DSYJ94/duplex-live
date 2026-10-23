"""One diagnosed006 experiment: silent listening instead of acknowledgments.

The failed V9 waveform and absent-caption interval remain immutable. This tests
only the documented backchannel prompt policy; no audio filtering, gate change,
reference answer, request forcing or automatic retry is allowed.
"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from evals import native_reasoning_teacher_ack_probe as parent
from evals import native_reasoning_component_speech as component
from evals.native_control_teacher import atomic_json,collection_lock
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_quality import assess
from evals.native_reasoning_source_quality_v4 import verify as source_quality
from evals.native_reasoning_teacher_quality import expected_output
from evals.native_reasoning_teacher_lifecycle_review import case_review

SCHEMA='dl-native-reasoning32-silent-listening-component-v10'
CASE=parent.CASE
POLICY=('Backchannel policy: Do not use backchannels. Listen silently while the user speaks. '
        'While waiting, remain silent except when answering an independent basic question. '
        'Give complete answers without listening acknowledgments. After an actual cancellation '
        'receipt, confirm its state with the sentence "The search is off."')


def configuration():
    value=deepcopy(parent.stream.configuration())
    if value['instructions'].count(parent.BACKCHANNEL)!=1:raise ValueError('Exact original policy required')
    value['instructions']=value['instructions'].replace(parent.BACKCHANNEL,POLICY)
    return value


def code_hashes():return dict(parent.code_hashes(),**{Path(__file__).name:sha(Path(__file__))})


def implementation():
    bind=parent.previous.bind;capture=bind(parent.stream.capture,configuration=configuration)
    return (bind(parent.original.capture_one,capture=capture,configuration=configuration,code_hashes=code_hashes),
            bind(parent.readback.verify_record,configuration=configuration,code_hashes=code_hashes))


def diagnosis(original,failed):
    import soundfile as sf
    header=json.loads((original/'ack-component.json').read_text());value=json.loads(failed.read_text())
    folder=original/CASE;record=header['recording'];path=failed.parent/'asr.json'
    pcm,rate=sf.read(folder/'model_out.wav',dtype='float32')
    actual=assess(expected_output(folder),json.loads(path.read_text()),pcm,rate)
    if (header['status']!='raw_component_complete_independent_speech_and_semantics_required'
            or header['code_hashes']!=parent.code_hashes() or header['case_id']!=CASE
            or header['failed_component_speech_sha256']!=parent.FAILED_SPEECH_SHA
            or header['maximum_aggregate_voice_attempts']!=46
            or value['code_sha256']!=sha(Path(component.__file__))
            or value['original_wav_sha256']!=record['artifacts']['model_out.wav']['sha256']
            or value['original_wav_sha256']!=sha(folder/'model_out.wav') or value['asr_sha256']!=sha(path)
            or value['actual_provider_transcript_sha256']!=sha(folder/'transcripts.json')
            or actual!=value['component_speech_verdict'] or actual['passed']is not False
            or actual['reasons']!=['Independent exact-word agreement failed']):
        raise ValueError('Exact independently failed V9 acknowledgment recording required')
    return actual


async def run(a):
    with collection_lock(a.lock):
        failed=diagnosis(a.parent,a.failed);source_quality(a.source,a.quality)
        manifest,plan=parent.source_manifest(a.source);tokenizer=parent.geometry(a.engine,a.tokenizer)
        scene=next(s for s in manifest['scenes']if s['id']==CASE);case=next(c for c in plan['cases']if c['id']==CASE)
        a.out.mkdir(parents=True,exist_ok=False)
        state=dict(schema=SCHEMA,at_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=code_hashes(),
            source_manifest_sha256=sha(a.source/'source-audio.json'),source_quality_sha256=sha(a.quality),
            parent_ack_manifest_sha256=sha(a.parent/'ack-component.json'),failed_component_speech_sha256=sha(a.failed),
            diagnosed_failure=failed,case_id=CASE,maximum_new_voice_sessions=1,maximum_backend_requests=6,
            prior_voice_attempts=46,maximum_aggregate_voice_attempts=47,
            prior_backend_request_upper_bound=121,maximum_aggregate_backend_request_upper_bound=127,
            automatic_retry=False,original_audio_or_source_words_changed=False,teacher_approved=False,
            semantic_approved=False,training_approved=False,status='capturing_one_silent_listening_experiment')
        target=a.out/'quiet-component.json';atomic_json(target,state);capture,verify=implementation()
        try:
            record=await capture(a.source,scene,case,a.out/CASE,sha(a.quality),'gpt-6-luna',a.key,tokenizer)
            state['recording']=record;atomic_json(target,state);verify(a.source,scene,a.out/CASE,record,tokenizer)
            lifecycle=case_review(case,json.loads((a.out/CASE/'lifecycle.json').read_text()));state['lifecycle']=lifecycle
            if record['frame_count_diagnostic_pass']is not True or lifecycle['structural_pass']is not True:
                raise ValueError('Unchanged full frame/lifecycle gate failed')
            if state['code_hashes']!=code_hashes():raise ValueError('Capture source changed')
            state['status']='raw_component_complete_independent_speech_and_semantics_required'
        except Exception as error:state.update(status='failed_preserved_no_retry',error_type=type(error).__name__,reason=str(error))
        finally:atomic_json(target,state)
        return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','quality','parent','failed','engine','tokenizer','key','lock','out'):
        p.add_argument('--'+name,type=Path,required=True)
    v=asyncio.run(run(p.parse_args()));print(json.dumps(dict(status=v['status'],teacher_approved=False,training_approved=False)))
    if v['status']=='failed_preserved_no_retry':raise SystemExit(1)
