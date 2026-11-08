"""Record all48 original ordinary TRAIN inputs through the actual native Engine.

No synthesis or weight updates. A disjoint TRAIN-only George reference matches
the target speaker; neither RPBen nor the uncertified released main voice is
silently substituted. This is input ownership evidence, not speech acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from evals.moshi_model_download import sha256 as sha

BASE=Path('/projects/u6ada/duplex-live')
PP=BASE/'public-models/personaplex-fdaf-v1'
VENDOR=BASE/'vendor/context-spanning-79df-v1'
WEIGHTS=BASE/'public-models/context-spanning-e62a-v1'

CODEC_SHA = 'f30a71b00f39a3b4e5b3baa4487783cc6117e6dbbc6db39181247374b4e72162'
CODEC_SOURCE = '4c025ad3ff8c1d64c735b3a4b861ce76ec4803a7'
AUTH = Path('docs/NATIVE-MOSHI-MATCHED-TRAINING-AUTHORIZATION-2026-11-08.json')
GEORGE = 'JBFqnCBsd6RMkjVDRZzb'
SCHEMA = 'dl-personaplex-ordinary48-native-inputs-v1'


def reference_sources(rows, ordinary_hashes):
    """Select whole, approved, non-overlapping TRAIN recordings deterministically."""
    candidates = sorted((r for r in rows if r['profile'] == 'assistant-british'
                         and r['source_sha256'] not in ordinary_hashes), key=lambda r:r['id'])
    selected, seconds = [], 0.
    for row in candidates:
        if row['fold'] != 'TRAIN' or not row['peak_pass'] or not 0 < row['seconds'] < 60:
            raise ValueError('Only complete approved TRAIN reference speech')
        selected.append(row)
        seconds += row['seconds']
        if seconds >= 8:
            return selected
    raise ValueError('Insufficient disjoint same-speaker TRAIN reference')


def preflight(a):
    import torch
    import sentencepiece
    from training.personaplex.codec_cache import inputs, EXPORT_SHA
    from data.venus.retention_quality import inventory
    from data.venus.retention_orthography import words
    from evals.context_spanning_download import SOURCE_COMMIT
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() != a.source:
        raise ValueError('Exact committed producer required')
    auth=json.loads(AUTH.read_text())
    if auth['status']!='OWNER_SELECTED_MATCHED_TRAINING_COMPARISON' or auth['new_external_model_API_calls']!=0:
        raise ValueError('Exact approved matched training authority required')
    if (subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(VENDOR),'status','--porcelain'],text=True).strip()):
        raise ValueError('Pinned clean native engine required')
    _, rows=inputs(a.data)
    report_file=a.cache/'cache.json'
    if sha(report_file)!=CODEC_SHA:
        raise ValueError('Exact complete48 codec report required')
    cache=json.loads(report_file.read_text())
    if (cache['source']!=CODEC_SOURCE or not cache['complete48'] or not cache['actual_CUDA']
            or cache['source_export_sha256']!=EXPORT_SHA or cache['optimizer_updates']!=0
            or cache['training_approved'] is not False or len(cache['rows'])!=48
            or [r['id'] for r in rows]!=[r['id'] for r in cache['rows']]):
        raise ValueError('All48 exact original codec records required')
    spm=sentencepiece.SentencePieceProcessor(str(PP/'tokenizer_spm_32k_3.model'))
    plan={s['id']:s for s in json.loads(Path('evals/fixtures/venus-retention-request-source-proposal-v2-20261107.json').read_text())['scenes']}
    for row,encoded in zip(rows,cache['rows']):
        p=a.cache/'codes'/encoded['file']
        if p.parent!=a.cache/'codes' or sha(p)!=encoded['sha256'] or encoded['fold']!='TRAIN':
            raise ValueError('Exact original TRAIN cache artifact required')
        saved=torch.load(p,map_location='cpu',weights_only=True);codes=saved['codes']
        if (saved['source_record']!=row or codes.dtype!=torch.int64
                or codes.shape!=(1,17,encoded['frames']) or (codes[:,0]==4).any()
                or (codes[:,0]<0).any() or (codes[:,0]>=32000).any()
                or (codes[:,1:]<0).any() or (codes[:,1:]>=2048).any()):
            raise ValueError('Full ordinary48 code geometry/words/ownership required')
        decoded=spm.decode([int(t) for t in codes[0,0] if int(t)>=5])
        if words(decoded)!=words(plan[row['id']]['teacher_program']['answer']):
            raise ValueError('Every original target word must survive')
    quality=a.sources/'teacher-inputs-v1/quality.json'
    q=json.loads(quality.read_text())
    expected=json.loads(Path('docs/NATIVE-RETENTION96-DERIVED182-REVIEW-2026-11-07.json').read_text())['derived_report_sha256']
    if sha(quality)!=expected or not q['all_sources_pass'] or q['passed']!=182:
        raise ValueError('Complete independent original182 qualification required')
    payload=Path('docs/NATIVE-RETENTION96-SPEECH-PAYLOAD-2026-11-07.json')
    source_rows=inventory(a.sources/'original',a.sources/'resumed',payload)
    selected=reference_sources(source_rows,{r['assistant_source_sha256'] for r in rows})
    requests={r['id']:r for r in json.loads(payload.read_text())['requests']}
    if any(requests[r['id']]['voice_id']!=GEORGE for r in selected):
        raise ValueError('Same George voice as the unchanged48 assistant targets required')
    proof=dict(source=a.source,complete48=True,codec_report_sha256=CODEC_SHA,export_sha256=EXPORT_SHA,
        all_words_exact=True,all_reference_sources_TRAIN=True,reference_disjoint_from_all48=True,
        reference_voice_id=GEORGE,reference_sources=selected,source_quality_sha256=expected,
        reference_policy='Concatenate complete original disjoint TRAIN clips; zero tail padding to native frame only; no resampling, trimming or gain change',
        optimizer_updates=0,training_approved=False)
    return rows,cache,proof


def capture(a):
    import numpy as np
    import soundfile as sf
    import torch
    from evals.native_control_engine import assemble,delayed,trace,validate_trace,DELAYS,GRAPH_POLICY
    from evals.context_native_audio49_data import digest
    from evals.context_execution import math_attention,snapshot
    from evals.context_native_layout import install
    from evals.context_spanning_probe import audited_loads
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_spanning_download import WEIGHT_SHA256
    from training.venus.isambard_native import guarded
    from portable.venus.native_app import ensure_guard
    ensure_guard();guarded(a.lease,240)
    rows,cache,proof=preflight(a)
    if json.loads((a.root/'preflight.json').read_text())!=proof:
        raise ValueError('Actual complete CPU preflight changed')
    out=a.root/'capture48';out.mkdir(exist_ok=False)
    os.environ.update(CS_BASE_DIR=str(PP),CS_WEIGHTS_DIR=str(WEIGHTS),CS_VOICES_DIR=str(WEIGHTS/'voices'))
    sys.path.insert(0,str(VENDOR))
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-B','-m','evals.cuda_linker_path','--out',str(out/'driver')],text=True).strip()
    if sha(WEIGHTS/'context_spanning_7b.pt')!=WEIGHT_SHA256:
        raise ValueError('Exact published capture weights required')
    from contextspan.model.engine import Engine,seed_all
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    seed_all(42424242);loads=[]
    with audited_loads(loads):engine=Engine(checkpoint=str(WEIGHTS/'context_spanning_7b.pt'),device='cuda',temp=.8,temp_text=.7)
    final=[r for r in loads if r['module_identity']==id(engine.lm)]
    if not final or not final[-1]['strict'] or final[-1]['missing'] or final[-1]['unexpected'] or engine.lm.dep_q!=16:
        raise ValueError('Strict full native model load required')
    install(engine);engine.lm.requires_grad_(False)
    if list(engine.lm.delays)!=DELAYS or engine.frame_size!=1920 or engine.frame_rate!=12.5:
        raise ValueError('Original native delay/stream clock required')
    chunks=[]
    for row in proof['reference_sources']:
        p=Path(row['wav']);wave,rate=sf.read(p,dtype='float32')
        if sha(p)!=row['source_sha256'] or rate!=24000 or wave.ndim!=1 or not np.isfinite(wave).all():
            raise ValueError('Exact original same-speaker reference PCM required')
        chunks.append(wave)
    reference=np.concatenate(chunks);padding=(-len(reference))%1920;reference=np.pad(reference,(0,padding))
    sf.write(out/'derived-reference.wav',reference,24000,subtype='FLOAT')
    engine.reset()
    with torch.no_grad():voice=engine.mimi.encode(torch.as_tensor(reference[None,None],device='cuda')).detach().cpu()[0]
    if voice.dtype!=torch.int64 or voice.shape!=(8,len(reference)//1920):
        raise ValueError('Complete native eight-codebook reference required')
    torch.save(dict(codes=voice),out/'voice.pt')
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,engine.spm);initial=engine.lm._get_initial_token().detach().cpu()
    before=tensor_hashes(engine.lm.named_parameters())
    report=dict(schema=SCHEMA,source=a.source,pid=os.getpid(),capture_code_sha256=sha(Path(__file__)),preflight=proof,voice_sha256=sha(out/'voice.pt'),
        prefix_sha256=digest(prefix),initial_token_sha256=digest(initial),reference_tail_padding_samples=padding,
        reference_wav_sha256=sha(out/'derived-reference.wav'),native_graph_policy=GRAPH_POLICY,
        native_protocol='bootstrap-preserve-provided-v1',execution=snapshot(engine.lm),strict_model_loads=loads,
        cases=[],complete48=False,passed=False,optimizer_updates=0,training_approved=False,
        teacher_text_and_audio_forced=True,latency_or_voice_quality_qualified=False)
    for source,entry in zip(rows,cache['rows']):
        ensure_guard();guarded(a.lease,60)
        folder=out/entry['id'];folder.mkdir()
        clean=torch.load(a.cache/'codes'/entry['file'],weights_only=True,map_location='cpu')['codes'][0]
        wave,rate=sf.read(a.data/source['path'],dtype='float32',always_2d=True)
        if rate!=24000 or wave.shape!=(clean.shape[1]*1920,2):raise ValueError('Every original source frame required')
        saved,calls,decoded,execution=trace(engine,voice,clean,wave[:,1],entry['id'])
        codes,mask=assemble(clean,prefix,saved['streamed_user_codes']);offline,_=assemble(clean,prefix)
        saved.update(codes=codes,mask=mask,targets=codes[0],offline_inputs=delayed(offline,_delay_sequence,initial))
        ownership=validate_trace(clean,prefix,saved,calls,_delay_sequence,initial)
        torch.save(saved,folder/'inputs.pt');sf.write(folder/'teacher-forced-native.wav',decoded,24000,subtype='FLOAT')
        report['cases'].append(dict(id=entry['id'],fold='TRAIN',calls=calls,ownership=ownership,
            inputs_sha256=sha(folder/'inputs.pt'),source_cache_sha256=entry['sha256'],
            teacher_forced_wav_sha256=sha(folder/'teacher-forced-native.wav'),native_execution=execution))
        (out/'engine-inputs.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(completed=len(report['cases']),required=48,optimizer_updates=0)),flush=True)
    if len(report['cases'])!=48 or before!=tensor_hashes(engine.lm.named_parameters()):
        raise ValueError('Complete48 and unchanged model required')
    report.update(complete48=True,passed=True,all_model_parameters_unchanged=True,independent_input_reconstruction_pending=True)
    (out/'engine-inputs.json').write_text(json.dumps(report,indent=2)+'\n')


def verify(a):
    """Fresh-process CPU reconstruction; no model, gradient or optimizer approval."""
    import numpy as np
    import soundfile as sf
    import torch
    from evals.native_control_engine import assemble,validate_trace
    from evals.context_native_audio49_data import digest
    rows,cache,current=preflight(a)
    out=a.root/'capture48';report=json.loads((out/'engine-inputs.json').read_text())
    saved=dict(report['preflight']);saved.pop('source')
    current=dict(current);current.pop('source')
    if (report['schema']!=SCHEMA or report['pid']==os.getpid() or saved!=current
            or report['capture_code_sha256']!=sha(Path(__file__)) or not report['complete48'] or not report['passed']
            or report['optimizer_updates']!=0 or report['training_approved'] is not False
            or report['all_model_parameters_unchanged'] is not True
            or len(report['cases'])!=48 or [r['id'] for r in report['cases']]!=[r['id'] for r in rows]
            or sha(out/'voice.pt')!=report['voice_sha256']):
        raise ValueError('Complete actual separate-process native capture required')
    os.environ['CS_BASE_DIR']=str(PP);sys.path.insert(0,str(VENDOR))
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    import sentencepiece
    spm=sentencepiece.SentencePieceProcessor(str(PP/'tokenizer_spm_32k_3.model'))
    voice=torch.load(out/'voice.pt',map_location='cpu',weights_only=True)['codes']
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
    initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
    if digest(prefix)!=report['prefix_sha256'] or digest(initial)!=report['initial_token_sha256']:
        raise ValueError('Reference/prefix/bootstrap identity differs')
    for row,encoded in zip(report['cases'],cache['rows']):
        folder=out/row['id']
        if (row['fold']!='TRAIN' or row['source_cache_sha256']!=encoded['sha256']
                or sha(folder/'inputs.pt')!=row['inputs_sha256']
                or sha(folder/'teacher-forced-native.wav')!=row['teacher_forced_wav_sha256']
                or row['native_execution']['cuda_main_graph_captured'] is not True):
            raise ValueError('Actual source/capture artifact identity differs')
        clean=torch.load(a.cache/'codes'/encoded['file'],map_location='cpu',weights_only=True)['codes'][0]
        tensors=torch.load(folder/'inputs.pt',map_location='cpu',weights_only=True)
        if validate_trace(clean,prefix,tensors,row['calls'],_delay_sequence,initial)!=row['ownership']:
            raise ValueError('Independent all-frame target/clock reconstruction differs')
        codes,mask=assemble(clean,prefix,tensors['streamed_user_codes'])
        if not torch.equal(tensors['codes'],codes) or not torch.equal(tensors['mask'],mask):
            raise ValueError('Independent teacher ownership changed')
        wave,rate=sf.read(folder/'teacher-forced-native.wav',dtype='float32')
        if rate!=24000 or wave.shape!=(clean.shape[1]*1920,) or not np.isfinite(wave).all():
            raise ValueError('Incomplete native teacher waveform')
    result=dict(passed=True,complete48=True,source=a.source,capture_source=report['source'],
        capture_pid=report['pid'],verification_pid=os.getpid(),actual_CPU_only=True,
        report_sha256=sha(out/'engine-inputs.json'),all_original_targets_and_recorded_inputs_reconstructed=True,
        optimizer_updates=0,training_approved=False,native_gradient_restore_and_behavior_still_required=True)
    path=out/'independent-inputs.json'
    if path.exists():raise ValueError('Preserve existing fresh verification; never overwrite')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def main(a):
    if a.verify_only:verify(a)
    elif a.preflight_only:
        _,_,proof=preflight(a);a.root.mkdir(parents=True,exist_ok=False)
        (a.root/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
        print(json.dumps(proof),flush=True)
    else:capture(a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','data','cache','sources'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source',required=True);p.add_argument('--lease');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--verify-only',action='store_true');main(p.parse_args())
