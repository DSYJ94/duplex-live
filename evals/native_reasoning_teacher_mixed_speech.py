"""Recompute every original-waveform speech gate in a typed complete32 view.

Unchanged recordings reuse their exact independent ASRs. Every replacement uses
its own pinned independent recognition. The entire32 is reassessed; no subset,
new normalization, semantic approval or native training is permitted here.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from evals import native_reasoning_teacher_mixed as mixed
from evals import native_reasoning_teacher_speech_v2 as region
from evals import native_reasoning_component_speech as component
from evals import native_reasoning_source_quality as quality
from evals.native_reasoning_source import sha
from evals.native_reasoning_teacher_quality import expected_output
from evals.native_reasoning_verification_session import source_readbacks

SCHEMA='dl-native-reasoning32-derived-whole-speech-v3'


def recognition(c,kind,identity):
    if kind=='v6':
        worker=json.loads((c.region_worker/'recognition.json').read_text())
        record=next(r for r in worker['recordings']if r['id']==identity)
        return c.region_worker/record['asr_file'],dict(kind='unchanged-v6',
            wav_sha256=record['wav_sha256'],asr_sha256=record['asr_sha256'])
    folder={'cutoff-v7':c.cutoff_speech,'repair-v8':c.repair_speech/identity}
    if hasattr(c,'ack_speech'):folder['ack-v9']=c.ack_speech
    if kind not in folder:raise ValueError('Unsupported independent recognition lineage')
    root=folder[kind];value=json.loads((root/'component-speech.json').read_text())
    if (value['schema']!='dl-native-reasoning-component-speech-diagnostic-v1'
            or value['code_sha256']!=sha(Path(component.__file__)) or value['policy']!=region.POLICY
            or value['asr_weight_sha256']!=region.WEIGHT or value['asr_packages']!=region.PACKAGES
            or value['original_waveform_changed']is not False or value['provider_requests']!=0
            or any(value[k]is not False for k in ('whole32_approved','semantic_approved','training_approved'))):
        raise ValueError('Exact same-method independent replacement recognition required')
    return root/'asr.json',dict(kind=kind,wav_sha256=value['original_wav_sha256'],
        asr_sha256=value['asr_sha256'],component_speech_sha256=sha(root/'component-speech.json'),
        actual_transcript_sha256=value['actual_provider_transcript_sha256'],
        component_verdict=value['component_speech_verdict'])


def rows(c,batch,header):
    import soundfile as sf
    result=[]
    if (header['complete_cases']!=32 or [r['id']for r in header['recordings']]!=mixed.IDS):
        raise ValueError('All32 typed records required')
    for record in header['recordings']:
        identity=record['id'];folder=batch/identity;wav=folder/'model_out.wav'
        path,binding=recognition(c,record['recorder'],identity)
        if (sha(wav)!=binding['wav_sha256']or sha(path)!=binding['asr_sha256']
                or binding['wav_sha256']!=record['recording']['artifacts']['model_out.wav']['sha256']):
            raise ValueError('Original WAV and independent ASR byte binding changed')
        pcm,rate=sf.read(wav,dtype='float32');raw=json.loads(path.read_text());spans=region.regions(pcm,rate)
        mapped=[dict(start=round(a*16000)/16000,end=round(b*16000)/16000)for a,b in spans]
        if raw.get('speech_activity')!=mapped:raise ValueError('Original waveform-only clock map changed')
        text=expected_output(folder);verdict=quality.assess(text,raw,pcm,rate)
        if record['recorder']!='v6':
            if (binding['actual_transcript_sha256']!=sha(folder/'transcripts.json')
                    or binding['component_verdict']!=verdict):
                raise ValueError('Independently recomputed component speech verdict differs')
        report=json.loads((folder/'report.json').read_text())
        if not text:verdict['reasons'].append('No actual provider speech transcript')
        if report['native_frame_budget']['frame_count_diagnostic_pass']is not True:
            verdict['reasons'].append('Complete prefix/live/actually sent context spans exceed unchanged1100-frame bound')
        verdict['passed']=not verdict['reasons']
        result.append(dict(id=identity,recorder=record['recorder'],
            original_rendered_wav_sha256=sha(wav),actual_provider_transcript=text,
            recognition_binding={k:v for k,v in binding.items()if k!='component_verdict'},
            original_clock_regions=spans,**verdict))
    return result


def body(c,batch,results):
    return dict(schema=SCHEMA,code_sha256=sha(Path(__file__)),
        collection_verifier_sha256=sha(Path(mixed.__file__)),assessor_sha256=sha(Path(quality.__file__)),
        normalization_sha256=sha(quality.NORMALIZATION_CODE),teacher_collection_sha256=sha(batch/'teacher-collection.json'),
        input_manifest_hashes=mixed.source_pins(c),policy=region.POLICY,
        asr_weight_sha256=region.WEIGHT,asr_packages=region.PACKAGES,
        complete_cases=32,results=results,passed=all(r['passed']for r in results),
        original_audio_changed=False,passing_subset_accepted=False,
        semantic_approved=False,training_approved=False,native_behavior_qualified=False)


def verify_value(c,batch,value,results,require_pass=True):
    expected=body(c,batch,results)
    if (set(value)!=set(expected)|{'at_utc'}or any(value[k]!=v for k,v in expected.items())
            or (require_pass and value['passed']is not True)):
        raise ValueError('Complete unchanged whole32 speech verdict required; no passing subset')
    return value


def finalize(c,batch,out):
    out.mkdir(parents=True,exist_ok=False)
    with source_readbacks([*mixed.input_roots(c),batch],include_old_speech=True)as memo:
        header,_,_=mixed.verify(c,batch);results=rows(c,batch,header)
        value=dict(body(c,batch,results),at_utc=datetime.now(timezone.utc).isoformat())
        target=out/'teacher-speech-quality.json';region.write(target,value)
        # Every speech decision is independently reread/recomputed, while the
        # already checked immutable transport stays bound by the full snapshot.
        verify_value(c,batch,json.loads(target.read_text()),rows(c,batch,header),require_pass=False)
    proof=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        complete_cases=32,all_original_transports_verified=True,all_speech_decisions_recomputed_twice=True,
        unchanged_input_snapshot=True,report_sha256=sha(target),whole_speech_passed=value['passed'],
        source_checker_executions=memo.calls,duplicate_reads_reused=memo.hits,training_approved=False)
    region.write(out/'independent-readback.json',proof)
    return value


def verify(c,batch,approval,*,require_pass=True):
    with source_readbacks([*mixed.input_roots(c),batch],include_old_speech=True):
        header,_,_=mixed.verify(c,batch)
        value=verify_value(c,batch,json.loads(approval.read_text()),rows(c,batch,header),require_pass)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','batch','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();v=finalize(mixed.load(a.config),a.batch,a.out)
    print(json.dumps(dict(complete_cases=32,passed_cases=sum(r['passed']for r in v['results']),whole_passed=v['passed'],training_approved=False)))
    if not v['passed']:raise SystemExit(1)
