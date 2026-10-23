"""One diagnosed006 acknowledgment-clarity experiment, not a corpus approval."""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from evals import native_reasoning_teacher_v8 as previous
from evals import native_reasoning_teacher_v7 as original
from evals import native_reasoning_stream_v6 as stream
from evals import native_reasoning_teacher_cutoff_readback as readback
from evals import native_reasoning_component_speech as component_speech
from evals.native_control_teacher import atomic_json,collection_lock
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_pacing import verify as source_manifest
from evals.native_reasoning_source_quality_v4 import verify as source_quality
from evals.native_reasoning_source_quality import assess
from evals.native_reasoning_teacher import geometry
from evals.native_reasoning_teacher_quality import expected_output
from evals.native_reasoning_teacher_lifecycle_review import case_review

CASE='native-reasoning4-train-006'
FAILED_SPEECH_SHA='76c9e76447f43a83672f9f3a673fc6a5b28c767a9a863d79f6f257aafe2de682'
BACKCHANNEL='Backchannel policy: Keep acknowledgments brief and avoid competing with the user.'
ACK=('Backchannel policy: Until you have a concrete answer, be silent or say exactly '
     '"Okay." once. Use that single complete acknowledgment and then keep listening. '
     'After an actual cancellation receipt, confirm its state with the complete sentence '
     '"The search is off." Answer any accompanying basic question directly once. '
     'These speech choices do not change the delegation or interruption policy.')


def configuration():
    value=deepcopy(stream.configuration())
    if value['instructions'].count(BACKCHANNEL)!=1:raise ValueError('Exact original policy required')
    value['instructions']=value['instructions'].replace(BACKCHANNEL,ACK)
    return value


def code_hashes():
    return dict(previous.code_hashes(),**{Path(__file__).name:sha(Path(__file__))})


def implementation():
    capture=previous.bind(stream.capture,configuration=configuration)
    return (previous.bind(original.capture_one,capture=capture,configuration=configuration,code_hashes=code_hashes),
            previous.bind(readback.verify_record,configuration=configuration,code_hashes=code_hashes))


def diagnosis(repairs,failed):
    import soundfile as sf
    header=json.loads((repairs/'repair-components.json').read_text());value=json.loads(failed.read_text())
    record=next(r for r in header['recordings']if r['id']==CASE);folder=repairs/CASE
    raw_path=failed.parent/'asr.json';pcm,rate=sf.read(folder/'model_out.wav',dtype='float32')
    actual=assess(expected_output(folder),json.loads(raw_path.read_text()),pcm,rate)
    if (header['status']!='raw_components_complete_unapproved' or header['failures']
            or header['source_commit']!='e762d82a2a83a53cffaae7597a05f000090a6c3e'
            or header['code_hashes']!=previous.code_hashes() or sha(failed)!=FAILED_SPEECH_SHA
            or value['schema']!='dl-native-reasoning-component-speech-diagnostic-v1'
            or value['code_sha256']!=sha(Path(component_speech.__file__))
            or value['original_wav_sha256']!=record['artifacts']['model_out.wav']['sha256']
            or value['original_wav_sha256']!=sha(folder/'model_out.wav')
            or value['asr_sha256']!=sha(raw_path)
            or value['actual_provider_transcript_sha256']!=sha(folder/'transcripts.json')
            or actual!=value['component_speech_verdict'] or actual['passed']is not False
            or actual['reasons']!=['Independent exact-word agreement failed']):
        raise ValueError('Exact independently failed original006 speech required')
    return record,actual


async def run(a):
    with collection_lock(a.lock):
        _,failure=diagnosis(a.repairs,a.failed)
        source_quality(a.source,a.quality)
        manifest,plan=source_manifest(a.source);tokenizer=geometry(a.engine,a.tokenizer)
        scene=next(s for s in manifest['scenes']if s['id']==CASE);case=next(c for c in plan['cases']if c['id']==CASE)
        a.out.mkdir(parents=True,exist_ok=False)
        state=dict(schema='dl-native-reasoning32-ack-clarity-component-v9',
                   at_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   code_hashes=code_hashes(),source_manifest_sha256=sha(a.source/'source-audio.json'),source_quality_sha256=sha(a.quality),
                   parent_repair_manifest_sha256=sha(a.repairs/'repair-components.json'),failed_component_speech_sha256=sha(a.failed),
                   diagnosed_failure=failure,case_id=CASE,maximum_new_voice_sessions=1,maximum_backend_requests=6,
                   prior_voice_attempts=45,maximum_aggregate_voice_attempts=46,
                   prior_backend_request_upper_bound=115,maximum_aggregate_backend_request_upper_bound=121,
                   automatic_retry=False,original_audio_or_words_changed=False,
                   teacher_approved=False,semantic_approved=False,training_approved=False,status='capturing_one_ack_policy_experiment')
        target=a.out/'ack-component.json';atomic_json(target,state)
        capture,verify=implementation()
        try:
            record=await capture(a.source,scene,case,a.out/CASE,sha(a.quality),'gpt-6-luna',a.key,tokenizer)
            state['recording']=record;atomic_json(target,state)
            verify(a.source,scene,a.out/CASE,record,tokenizer)
            lifecycle=case_review(case,json.loads((a.out/CASE/'lifecycle.json').read_text()));state['lifecycle']=lifecycle
            if record['frame_count_diagnostic_pass']is not True or lifecycle['structural_pass']is not True:
                raise ValueError('Complete unchanged frame/lifecycle gate failed')
            if state['code_hashes']!=code_hashes():raise ValueError('Capture source changed')
            state['status']='raw_component_complete_independent_speech_and_semantics_required'
        except Exception as error:
            state.update(status='failed_preserved_no_retry',error_type=type(error).__name__,reason=str(error))
        finally:atomic_json(target,state)
        return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','quality','repairs','failed','engine','tokenizer','key','lock','out'):
        p.add_argument('--'+name,type=Path,required=True)
    v=asyncio.run(run(p.parse_args()));print(json.dumps(dict(status=v['status'],teacher_approved=False,training_approved=False)))
    if v['status']=='failed_preserved_no_retry':raise SystemExit(1)
