"""Complete native encoding of the preserved48 source exports; no weight updates.

Reuses the original pinned Mimi/interleaver and200ms measured-word check.
This is not a native recorded-call, gradient or training-state approval.
"""
import argparse,json,subprocess,os,sys
from pathlib import Path
from data.personaplex.source_export import sha
from evals import native_control_cache as native
from evals.recovery_encode import native_word_timing

EXPORT_SHA='74e7e406443830da4765eb4a5ad1df45b04a59f4e9b321aeb1e89801872ba9b2'


def inputs(data):
    data=Path(data);manifest=data/'manifest.json'
    if sha(manifest)!=EXPORT_SHA:raise ValueError('Exact complete source export required')
    d=json.loads(manifest.read_text());rows=d['rows']
    if not d['complete_all48'] or len(rows)!=48 or len({r['id'] for r in rows})!=48 or any(r['fold']!='TRAIN' or not r['original_channel_samples_exact'] for r in rows):raise ValueError('All48 original TRAIN scenes required')
    for r in rows:
        p=data/r['path']
        if not p.resolve().is_relative_to(data.resolve()) or p.is_symlink() or sha(p)!=r['wav_sha256'] or sha(p.with_suffix('.json'))!=r['labels_sha256']:raise ValueError('Exact original source export files required')
    return d,rows


def text_rows(a,rows,tokenizer,builder):
    import soundfile as sf
    from data.venus.retention_orthography import words
    plan={r['id']:r for r in json.loads(Path('evals/fixtures/venus-retention-request-source-proposal-v2-20261107.json').read_text())['scenes']}
    result=[]
    for r in rows:
        p=a.data/r['path'];info=sf.info(p);side=json.loads(p.with_suffix('.json').read_text());frames=info.frames//1920
        if info.samplerate!=24000 or info.channels!=2 or info.frames%1920 or abs(info.duration-r['duration'])>1e-9:raise ValueError('Complete original stereo native frames required')
        text=builder.prepare_item(side['alignments'],native.frame_duration(frames))
        if tuple(text.shape)!=(1,1,frames) or (text==4).any():raise ValueError('No lost frames or spurious native action tokens')
        timing=native_word_timing(text[0,0].tolist(),side['alignments'],tokenizer.encode,12.5)
        decoded=tokenizer.decode([int(t) for t in text[0,0] if int(t) not in (-1,0,1,2,3)])
        if words(decoded)!=words(plan[r['id']]['teacher_program']['answer']):raise ValueError('Every original assistant word must survive native tokenization')
        result.append(dict(id=r['id'],frames=frames,text_codes=text,native_word_timing=timing))
    return result


def main(a):
    import torch
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source:raise ValueError('Exact committed codec producer required')
    _,rows=inputs(a.data);pins=native.source_pins(a.codec_source,a.interleaver_source,a.model)
    _,tokenizer,builder=native.text_builder(a.codec_source,a.interleaver_source,a.model,'cpu')
    texts=text_rows(a,rows,tokenizer,builder)
    if a.preflight_only:
        a.root.mkdir(parents=True,exist_ok=False)
        torch.save(texts,a.root/'text48.pt')
        r=dict(source=a.source,source_export_sha256=EXPORT_SHA,pins=pins,complete48=True,all_words_exact=True,
            maximum_first_subword_offset_ms=max(t['native_word_timing']['max_first_word_offset_ms'] for t in texts),
            text48_sha256=sha(a.root/'text48.pt'),optimizer_updates=0,native_capture_and_training_approved=False)
        (a.root/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);return
    from portable.venus.native_app import ensure_guard
    from training.venus.isambard_native import guarded
    import soundfile as sf
    ensure_guard();guarded(a.lease,180)
    pre=json.loads((a.root/'preflight.json').read_text())
    if pre['source']!=a.source or pre['pins']!=pins or not pre['complete48'] or sha(a.root/'text48.pt')!=pre['text48_sha256']:raise ValueError('Exact completed source/text preflight required')
    driver=subprocess.check_output([sys.executable,'-B','-m','evals.cuda_linker_path','--out',str(a.root/'driver')],text=True).strip()
    os.environ['TRITON_LIBCUDA_PATH']=driver
    codes_root=a.root/'codes';codes_root.mkdir(exist_ok=False)
    mimi,tokenizer,builder=native.runtime(a.codec_source,a.interleaver_source,a.model);records=[]
    for row,expected in zip(rows,texts):
        ensure_guard();guarded(a.lease,90);p=a.data/row['path'];audio,sr=sf.read(p,dtype='float32')
        codes,timing=native.encode_item(audio,json.loads(p.with_suffix('.json').read_text()),mimi,builder,tokenizer)
        if sr!=24000 or not torch.equal(codes[:,:1],expected['text_codes']):raise ValueError('Native encoded text differs from complete CPU preflight')
        dest=codes_root/(row['id']+'.pt');torch.save(dict(codes=codes,fold='TRAIN',source_record=row),dest)
        records.append(dict(id=row['id'],fold='TRAIN',file=dest.name,sha256=sha(dest),frames=codes.shape[-1],native_word_timing=timing))
        (a.root/'progress.json').write_text(json.dumps(dict(complete=len(records),required=48,source=a.source))+'\n');print(json.dumps(dict(encoded=len(records),required=48)),flush=True)
    report=dict(source=a.source,pins=pins,source_export_sha256=EXPORT_SHA,complete48=len(records)==48,actual_CUDA=True,
        independent_source_and_measured_word_approvals_preserved=True,native_recorded_user_capture_pending=True,
        native_gradient_and_state_restore_pending=True,training_approved=False,optimizer_updates=0,external_API_calls=0,rows=records)
    (a.root/'cache.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','data','codec-source','interleaver-source','model'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source',required=True);p.add_argument('--lease');p.add_argument('--preflight-only',action='store_true');main(p.parse_args())
