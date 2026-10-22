"""Reapprove all40 with one byte-bound WebRTC006 repair and unchanged scene bars.

All original failed approvals remain intact. The decoded server input receives an
additional independent lexical check; client output keeps its actual playout clock.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil

from data.synthetic.native_control_render import sha256, validate_raw
from evals import native_control_teacher_scoped_quality as parent
from evals.native_control_teacher import atomic_json
from evals.native_control_teacher_artifacts import input_availability
from evals.native_control_webrtc_artifacts import verify as verify_capture
from evals.native_control_webrtc_capture import SID
from evals.native_control_words import agreement, spoken_words

SCHEMA = 'dl-native-control3-teacher-quality-v7'
COLLECTION = 'dl-native-control3-teacher-collection-v6'


def pins():
    return dict(parent=parent.pins(), **{name:sha256(Path('evals')/name) for name in (
        'native_control_webrtc_quality.py','native_control_webrtc_capture.py',
        'native_control_webrtc_artifacts.py','native_control_webrtc_probe.py')})


def reflected_words(folder, scene, response):
    import math
    import numpy as np
    import soundfile as sf
    pcm,rate = sf.read(folder/'server-input.wav',dtype='float32')
    expected = ' '.join(t['text'] for t in scene['turns'] if t['role']=='user')
    words = [w for s in response['segments'] for w in s.get('words',[])]
    comparison = agreement(expected,response.get('text',''))
    valid = bool(words) and all(type(w['start']) in (int,float) and type(w['end']) in (int,float)
        and math.isfinite(w['start']+w['end']) and 0<=w['start']<w['end']<=len(pcm)/rate+.08 for w in words)
    tokens = [t for w in words for t in spoken_words(w['text'])]
    exact = tokens==spoken_words(expected)==spoken_words(response.get('text',''))
    return dict(passed=bool(rate==24000 and pcm.ndim==1 and np.isfinite(pcm).all()
                           and float(np.abs(pcm).max())<=.995 and valid and exact),
                expected_text=expected,asr_text=response.get('text',''),word_agreement=comparison,
                exact_spoken_words=exact,valid_word_intervals=valid,peak=float(np.abs(pcm).max()),
                waveform_sha256=sha256(folder/'server-input.wav'),expected_text_prompt_used=False,
                source_audio_changed=False,training_approved=False)


def parent_evidence(raw,source_audit,batch,quality):
    report = parent.verify(raw,source_audit,batch,quality,require_pass=False)
    failed = [r['id'] for r in report['results'] if not r['checks']['passed']]
    if failed!=[SID] or report['passed'] is not False:
        raise ValueError('Only diagnosed39/40 parent with failed006 may receive this repair')
    return report,json.loads((batch/'teacher.json').read_text())


def validate_collection(raw,source_audit,batch):
    manifest=json.loads((batch/'teacher.json').read_text())
    lineage=manifest['parent_evidence']
    old_batch,old_quality=Path(lineage['batch']),Path(lineage['quality'])
    report,old=parent_evidence(raw,source_audit,old_batch,old_quality)
    source,plan=validate_raw(raw)
    ids=[s['id'] for s in plan['cases']]
    if (manifest['schema']!=COLLECTION or manifest['code_pins']!=pins()
            or manifest['source_raw_manifest_sha256']!=sha256(raw/'raw-source.json')
            or manifest['source_word_audit_sha256']!=sha256(source_audit)
            or manifest['selected_ids']!=ids or len(ids)!=40
            or [r['id'] for r in manifest['recordings']]!=ids or manifest['retained_records']!=39
            or manifest['new_protocol_ids']!=[SID] or manifest['new_api_sessions_created']!=0
            or manifest['training_approved'] is not False or manifest['status']!='raw_complete_unapproved'
            or lineage['batch_sha256']!=sha256(old_batch/'teacher.json')
            or lineage['quality_sha256']!=sha256(old_quality)):
        raise ValueError('Complete40 unchanged source and parent provenance required')
    for row,record,previous in zip(source['cases'],manifest['recordings'],old['recordings']):
        folder=batch/row['id']
        if record['id']!=row['id'] or record['dir']!=row['id'] or record['training_approved'] is not False:
            raise ValueError('Teacher record identity changed')
        if row['id']!=SID and record!=previous:
            raise ValueError('A passing retained teacher was changed')
        for name,digest in record['artifacts'].items():
            if Path(name).name!=name or (folder/name).is_symlink() or sha256(folder/name)!=digest:
                raise ValueError('Complete teacher artifact inventory changed')
        if row['id']==SID:
            expected=json.loads((folder/'artifacts.json').read_text())
            expected['artifacts.json']=sha256(folder/'artifacts.json')
            if record['artifacts']!=expected:
                raise ValueError('WebRTC original inventory differs')
            verify_capture(raw,row,source_audit,folder)
    return manifest,source,plan,report


def verify(raw,source_audit,batch,approval,*,require_pass=True):
    manifest,source,plan,old=validate_collection(raw,source_audit,batch)
    report=json.loads(approval.read_text());checks=json.loads(source_audit.read_text())
    if (report['schema']!=SCHEMA or report['code_pins']!=pins() or report['all40_complete'] is not True
            or report['teacher_manifest_sha256']!=sha256(batch/'teacher.json')
            or report['source_raw_manifest_sha256']!=sha256(raw/'raw-source.json')
            or report['source_word_audit_sha256']!=sha256(source_audit)
            or report['asr_weight_sha256']!=parent.original.ASR_WEIGHT
            or report['expected_text_prompt_used'] is not False or report['original_audio_changed'] is not False
            or report['training_approved'] is not False or report['model_behavior_qualified'] is not False
            or report['semantic_approval_required'] is not True or report['native_conversion_and_numerical_required'] is not True
            or report['reused_recognition_count']!=39 or report['recognition_calls']!=1
            or report['additional_server_input_recognition_calls']!=1
            or [r['id'] for r in report['results']]!=manifest['selected_ids']):
        raise ValueError('Complete unchanged40 quality evidence required')
    for source_row,scene,source_checks,row,previous in zip(source['cases'],plan['cases'],checks['results'],report['results'],old['results']):
        path=approval.parent/(scene['id']+'.asr.json')
        if (row['asr_file']!=path.name or row['asr_sha256']!=sha256(path)
                or row['internal_silence_removed'] is not False or row['original_clock_restored'] is not True
                or (scene['id']!=SID and row['asr_sha256']!=previous['asr_sha256'])):
            raise ValueError('Original independent recognition evidence changed')
        actual=parent.decision(raw,source_row,scene,batch/scene['id'],source_checks['clips'],json.loads(path.read_text()))
        if any(row[k]!=v for k,v in actual.items()):
            raise ValueError('Recomputed teacher decision differs')
    path=approval.parent/'server-input.turbo.json'
    scene=next(s for s in plan['cases'] if s['id']==SID)
    if (report['server_input_asr_sha256']!=sha256(path)
            or report['server_input_checks']!=reflected_words(batch/SID,scene,json.loads(path.read_text()))):
        raise ValueError('Server-reflected input recognition or source words changed')
    passed=all(r['checks']['passed'] for r in report['results']) and report['server_input_checks']['passed']
    if type(report['passed']) is not bool or report['passed']!=passed or (require_pass and not passed):
        raise ValueError('All40 quality and independent received-input checks must pass')
    return report


def reapprove(raw,source_audit,old_batch,old_quality,capture,out,weights,threads=2):
    old,manifest=parent_evidence(raw,source_audit,old_batch,old_quality)
    source,plan=validate_raw(raw)
    verify_capture(raw,next(r for r in source['cases'] if r['id']==SID),source_audit,capture)
    if sha256(weights)!=parent.original.ASR_WEIGHT or not 1<=threads<=8:
        raise ValueError('Pinned independent recognizer with bounded threads required')
    packages={p:importlib.metadata.version(p) for p in old['asr_packages']}
    if packages!=old['asr_packages']:raise ValueError('Recognizer packages changed')
    out.mkdir(parents=True,exist_ok=False);batch=out/'teachers';batch.mkdir();quality=out/'quality';quality.mkdir()
    records=[]
    for previous in manifest['recordings']:
        sid=previous['id'];folder=batch/sid
        shutil.copytree(capture if sid==SID else old_batch/sid,folder)
        if sid==SID:
            artifacts=json.loads((folder/'artifacts.json').read_text());artifacts['artifacts.json']=sha256(folder/'artifacts.json')
            record=dict(id=sid,dir=sid,artifacts=artifacts,transport_complete=True,training_approved=False)
        else:record=previous
        records.append(record)
    collected=dict(schema=COLLECTION,code_pins=pins(),model='gpt-live-1',
        source_raw_manifest_sha256=sha256(raw/'raw-source.json'),source_word_audit_sha256=sha256(source_audit),
        selected_ids=[s['id'] for s in plan['cases']],recordings=records,retained_records=39,new_protocol_ids=[SID],
        new_api_sessions_created=0,training_approved=False,status='raw_complete_unapproved',
        parent_evidence=dict(batch=str(old_batch),quality=str(old_quality),batch_sha256=sha256(old_batch/'teacher.json'),quality_sha256=sha256(old_quality)))
    atomic_json(batch/'teacher.json',collected)
    import numpy as np
    import soundfile as sf
    import torch
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from evals.moshi_speech_audit import asr_window,restore_clock
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    model=whisper.load_model(str(weights),device='cpu')
    def recognize(wav,path):
        pcm,rate=sf.read(wav,dtype='float32')
        if rate!=24000:raise ValueError('Original24k waveform required')
        window,head,tail,silent=asr_window(resample_poly(pcm,2,3).astype(np.float32),True,True,True)
        response=dict(text='',segments=[]) if silent else whisper.transcribe(model,window,language='en',vad=False,beam_size=5,best_of=5,temperature=0.,fp16=False,verbose=None)
        atomic_json(path,restore_clock(response,head/16000));return head/16000,tail/16000
    checks=json.loads(source_audit.read_text());results=[]
    for source_row,scene,source_checks,previous in zip(source['cases'],plan['cases'],checks['results'],old['results']):
        sid=scene['id'];path=quality/(sid+'.asr.json')
        if sid==SID:head,tail=recognize(batch/sid/'model_out.wav',path)
        else:
            shutil.copyfile(old_quality.parent/previous['asr_file'],path)
            head,tail=previous['leading_window_s'],previous['trailing_window_s']
        row=parent.decision(raw,source_row,scene,batch/sid,source_checks['clips'],json.loads(path.read_text()))
        row.update(asr_file=path.name,asr_sha256=sha256(path),leading_window_s=head,trailing_window_s=tail,internal_silence_removed=False,original_clock_restored=True)
        results.append(row);print(json.dumps(dict(id=sid,passed=row['checks']['passed'],complete=len(results))),flush=True)
    path=quality/'server-input.turbo.json';recognize(batch/SID/'server-input.wav',path)
    received=reflected_words(batch/SID,next(s for s in plan['cases'] if s['id']==SID),json.loads(path.read_text()))
    report=dict(old,schema=SCHEMA,at_utc=datetime.now(timezone.utc).isoformat(),code_pins=pins(),teacher_manifest_sha256=sha256(batch/'teacher.json'),
                results=results,passed=all(r['checks']['passed'] for r in results) and received['passed'],recognition_calls=1,reused_recognition_count=39,
                additional_server_input_recognition_calls=1,server_input_asr_sha256=sha256(path),server_input_checks=received)
    report.pop('derivation',None)
    atomic_json(quality/'teacher-quality.json',report)
    verify(raw,source_audit,batch,quality/'teacher-quality.json')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('raw','source-audit','parent-batch','parent-quality','capture','out','weights'):
        ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();report=reapprove(a.raw,a.source_audit,a.parent_batch,a.parent_quality,a.capture,a.out,a.weights)
    print(json.dumps(dict(passed=report['passed'],complete=len(report['results']),training_approved=False)))


if __name__=='__main__':main()
