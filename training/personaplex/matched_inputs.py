"""Bind the complete approved323-row native corpus without changing old gates.

CPU preparation only. Reuse the original strict source/ownership validators;
new bases/capacities still need their own native gradient and restore proofs.
Frozen native520 anchor outputs remain separate original protected artifacts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from evals.moshi_model_download import sha256 as sha
from training.personaplex import matched_schedule

BASE=Path('/projects/u6ada/duplex-live')
OLD=Path('/projects/u6si/duplex-live')
MIG=Path('/workspace/migration')
PUBLIC=BASE/'public-models/personaplex-fdaf-v1'
ASSETS=OLD/'runs/context-spanning-assets-7454205/manifest.json'
PARENT=OLD/'runs/context-training-proof-7510331'


def component_teacher_caches(audits):
    """Older compact audits omit teacher metadata; read their pinned manifests.

    Some action-only components have no teacher. Their original validators still
    run below; omitting a nonexistent teacher field never omits a component.
    """
    from evals.context_component_kind import manifest_name
    caches=set()
    for row in audits:
        folder=Path(row['component']);path=folder/manifest_name(folder,mixed=True)
        if sha(path)!=row['manifest_sha256']:raise ValueError('Original component manifest changed')
        teacher=json.loads(path.read_text()).get('teacher')
        if isinstance(teacher,dict) and 'cache' in teacher:
            cache=teacher['cache']
            if not isinstance(cache,str) or not cache.startswith(str(OLD/'runs')+'/') or '..' in Path(cache).parts:
                raise ValueError('Unexpected original teacher cache path')
            caches.add(cache)
    return caches


def component_inputs(spm,voice,examples,audits):
    import torch
    from evals.context_component_kind import manifest_name
    from evals import context_control_rollin,context_request_prefix,context_action_component,context_early_validate,context_future_validate
    validators={'control-rollin.json':context_control_rollin.validate,
        'request-prefix.json':context_request_prefix.validate,'action-future.json':context_action_component.validate,
        'early-future.json':context_early_validate.validate,'actual-future.json':context_future_validate.validate}
    rows=[]
    for expected in audits:
        folder=Path(expected['component']);name=manifest_name(folder,mixed=True)
        manifest=json.loads((folder/name).read_text());history=Path(manifest['actual_history'])
        original=json.loads((history/'student-history.json').read_text())
        # Original validators derive COMPLETE future clocks, not just the sparse
        # raw capture mask. Reusing raw .pt masks would silently lose silence.
        audit,component=validators[name](folder,Path(original['source']['conversion']),ASSETS,examples,spm,voice)
        if audit!=expected:raise ValueError('Original independent component approval changed')
        raw=torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)['delayed_inputs']
        live=[c for c in original['calls'] if c['mode']=='live']
        stop=live[manifest['future_first_real_frame']]['start']
        x=torch.cat([raw[:,:,:stop],component['delayed_inputs']],dim=2)
        m=torch.zeros(x.shape[-1],dtype=torch.bool);y=torch.zeros(x.shape[-1],dtype=torch.int64)
        m[stop:]=component['text_mask'];y[stop:]=component['text_targets']
        if m[:stop].any() or int(m.sum())!=audit['text_targets']:
            raise ValueError('Original full future mask or past exclusion changed')
        boundary=(manifest.get('native_response_boundary_version') in (2,3)
            and (manifest.get('target_kind')=='ordinary_answer' or (manifest.get('native_response_boundary_version')==3
                and manifest.get('target_kind')=='clarification_question'
                and manifest.get('question_retiming_before_project_verified') is True))
            and manifest.get('actual_boundary_cache_verified') is True)
        rows.append(dict(id=str(len(rows))+':'+str(folder),kind=manifest.get('target_kind','private_result'),
            fold='TRAIN',inputs=x,mask=m,targets=y,boundary=boundary,original_audit=audit))
    if len(rows)!=28:raise ValueError('All28 original actual-history components required')
    return rows


def engine_rows(cache,engine,reasoning=False):
    """Use the old complete native input validators; never their old gradient gate."""
    import torch
    if reasoning:
        from evals.native_reasoning_engine import verify
        original=json.loads((cache/'cache.json').read_text())
        args=SimpleNamespace(inputs=MIG/'native-reasoning32-timing-v11-inputs-v1.json',
            version=MIG/'native-reasoning32-timing-v11-reapproval-v1',
            semantic=MIG/'native-reasoning32-derived-v10-semantic-v1',
            data=MIG/'native-reasoning32-converted-v11',model=Path(original['model']),
            codec_source=Path(original['codec_source']),interleaver_source=Path(original['interleaver_source']),
            cache=cache,assets=ASSETS,proof=PARENT,out=engine)
        report=verify(args)
    else:
        from evals.native_control_engine import verify
        report=verify(cache,engine)
    result=[]
    for entry in report['cases']:
        p=engine/entry['id']/entry['file']
        saved=torch.load(p,weights_only=True,map_location='cpu')
        result.append(dict(id=entry['id'],kind=entry['family'],fold='TRAIN',
            inputs=saved['inputs'],mask=saved['mask'],targets=saved['targets'],
            codes=saved['codes'][None],audio_mask=saved['audio_mask'][None] if reasoning else saved['mask'][None],
            calls=entry['calls'],source_sha256=sha(p)))
    if len(result)!=(32 if reasoning else 40):raise ValueError('Complete native control/reasoning cohort required')
    return result


def ordinary_rows(root,data,cache,spm):
    import torch
    from training.personaplex.native_capture48 import CODEC_SHA,SCHEMA
    from training.personaplex.codec_cache import inputs
    from evals.native_control_engine import validate_trace
    from evals.context_native_audio49_data import digest
    from contextspan.moshi.models.lm import _delay_sequence
    from contextspan.model.sequence_convention import persona_prefix
    from evals.lychee_context import BASE_INSTRUCTIONS
    out=root/'capture48';report=json.loads((out/'engine-inputs.json').read_text())
    independent=json.loads((out/'independent-inputs.json').read_text());_,sources=inputs(data)
    if (report['schema']!=SCHEMA or not report['passed'] or not report['complete48']
            or not report['all_model_parameters_unchanged'] or report['optimizer_updates']!=0
            or report['training_approved'] is not False or not independent['passed'] or not independent['complete48']
            or independent['capture_pid']==independent['verification_pid']
            or independent['report_sha256']!=sha(out/'engine-inputs.json') or sha(cache/'cache.json')!=CODEC_SHA
            or report['capture_code_sha256']!=sha(Path(__file__).with_name('native_capture48.py'))):
        raise ValueError('Actual complete48 and separate CPU reconstruction required')
    encoded=json.loads((cache/'cache.json').read_text())['rows']
    if [r['id'] for r in report['cases']]!=[r['id'] for r in sources] or len(encoded)!=48:
        raise ValueError('Every original ordinary48 must remain')
    voice=torch.load(out/'voice.pt',weights_only=True,map_location='cpu')['codes']
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
    initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
    if sha(out/'voice.pt')!=report['voice_sha256'] or digest(prefix)!=report['prefix_sha256']:
        raise ValueError('Original same-speaker reference changed')
    rows=[]
    for row,source in zip(report['cases'],encoded):
        p=out/row['id']/'inputs.pt';original=cache/'codes'/source['file']
        if sha(p)!=row['inputs_sha256'] or sha(original)!=row['source_cache_sha256']:
            raise ValueError('Original ordinary source/capture tensor changed')
        saved=torch.load(p,weights_only=True,map_location='cpu')
        clean=torch.load(original,weights_only=True,map_location='cpu')['codes'][0]
        if validate_trace(clean,prefix,saved,row['calls'],_delay_sequence,initial)!=row['ownership']:
            raise ValueError('Original complete48 teacher ownership changed')
        rows.append(dict(id=row['id'],kind='ordinary',fold='TRAIN',inputs=saved['inputs'],
            mask=saved['mask'],targets=saved['targets'],codes=saved['codes'][None],
            audio_mask=saved['mask'][None],calls=row['calls'],source_sha256=sha(p)))
    return rows


def prepare(a):
    import torch
    import sentencepiece
    from evals.native_retained_prepared import load
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.context_training_data import load_training
    from evals.context_warm_ordinary_retention import select as words, IDS
    from evals.context_warm_ordinary_clock import select as clocks,CLOCK_IDS
    from evals.context_native_stream_training import bindings
    from evals.context_native_audio49_data import digest
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source:
        raise ValueError('Exact Git data producer required')
    spm=sentencepiece.SentencePieceProcessor(str(PUBLIC/'tokenizer_spm_32k_3.model'))
    voice_file=PARENT/'voice.pt';voice=torch.load(voice_file,weights_only=True,map_location='cpu')['codes']
    data,prepared,policy=load(MIG/'native-retained49-prepared-data-v1',
        MIG/'native-original-runs-local-stage-v1.json',ASSETS,voice_file,spm)
    identity,parent=verify_checkpoint(PARENT)
    cache_paths={r['cache'] for r in parent['retention_examples']}
    cache_paths.update(component_teacher_caches(data['original28_actual_components']))
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
    examples,_=load_training([Path(p) for p in sorted(cache_paths)],spm,prefix,
        text_padding='legacy-epad0',request_policy='native-prelead-epad')
    groups={'actual':component_inputs(spm,voice,examples,data['original28_actual_components'])}
    initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
    refs=[]
    for record in parent['retention_examples']:
        example=next(e for e in examples if e['record']['id']==record['id'])
        if example['record']!=record:raise ValueError('Original four retention components changed')
        codes=example['codes'][None];x=torch.cat([initial,_delay_sequence([0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1],codes,initial)],dim=2)[:,:,:-1]
        refs.append(dict(id=record['id'],kind=record['kind'],fold='TRAIN',inputs=x,mask=example['text_mask'],record=record))
    groups['reference']=refs;old=[];new=[];audio=[]
    for row in prepared:
        record=row['record']['teacher'];batch=(row['delayed_inputs'],row['text_mask'][0],row['codes'][0,0],record)
        (old if row['record']['group']=='old25' else new).append(batch)
        audio.append(dict(id=record['id'],kind=record['kind'],fold='TRAIN',codes=row['codes'],audio_mask=row['audio_mask']))
    for group,items in (('old25',old),('new24',new)):
        groups[group]=[dict(id=r['id'],kind=r['kind'],fold='TRAIN',inputs=x,mask=m,targets=y,record=r) for x,m,y,r in items]
    groups['audio49']=audio
    stream=bindings(SimpleNamespace(actual_native_teacher_preflight=OLD/'runs/native-old25-native-math-7545975',
        native_v2_preflight=OLD/'runs/native-v2-native-math-7546608'),old,new)
    groups['stream49']=[dict(id=r['id'],kind=r['kind'],fold='TRAIN',inputs=x,mask=m,targets=y,calls=calls) for (x,m,y,r),calls in stream]
    actual=[(r['inputs'],r['mask'],r['targets']) for r in groups['actual']]
    anchor_root=OLD/'runs/native-audio49-training-preflight-7581639'
    for group,selected,ids,stem in [('word',words(old,new),IDS,'warm-ordinary'),
        ('clock',clocks(old,new,actual,data['original28_actual_components']),CLOCK_IDS,'warm-ordinary-clock')]:
        file=anchor_root/(stem+'-references.pt');audit=json.loads((anchor_root/(stem+'-reference-audit.json')).read_text())
        if sha(file)!=audit['cache_sha256'] or not audit['reference_targets_verified'] or audit['optimizer_steps_during_construction']!=0:
            raise ValueError('Original native520 protected anchors changed')
        stored=torch.load(file,weights_only=True,map_location='cpu');entries=[]
        for (x,m,y,record),ident in zip(selected,ids):
            active=m & (y!=3) if group=='word' else m
            h,logp=stored[ident+'.hidden'],stored[ident+'.logp']
            if h.shape!=(1,int(active.sum()),4096) or logp.shape!=(int(active.sum()),32000):
                raise ValueError('Full protected anchor ownership changed')
            entries.append(dict(id=ident,kind='simple',fold='TRAIN',inputs=x,mask=active,
                anchor_file=str(file),anchor_sha256=audit['cache_sha256'],anchor_tensor_prefix=ident,
                original_anchor_policy=audit['policy']))
        groups[group]=entries
    groups['control40']=engine_rows(MIG/'native-control40-preflight-v2/codes',a.control_engine)
    groups['reasoning32']=engine_rows(MIG/'native-reasoning32-codes-v11',MIG/'native-reasoning32-engine-v11',reasoning=True)
    groups['ordinary48']=ordinary_rows(a.capture,BASE/'native-personaplex-source48-v1',BASE/'native-personaplex-codec48-v2',spm)
    matched_schedule.validate(groups)
    schedules=[matched_schedule.epoch(groups,n) for n in range(4)]
    a.out.mkdir(parents=True,exist_ok=False)
    artifact=a.out/'inputs.pt';torch.save(groups,artifact)
    fingerprints={g:[dict(id=r['id'],tensors={k:digest(v) for k,v in r.items() if isinstance(v,torch.Tensor)}) for r in rows] for g,rows in groups.items()}
    report=dict(source=a.source,complete323=True,groups={k:len(v) for k,v in groups.items()},
        input_sha256=sha(artifact),input_tensor_fingerprints=fingerprints,prepared49_policy=policy,
        original_parent=identity,protected_word_clock_anchors_original=True,
        new_model_outputs_gradients_or_approval_decisions_cached=False,complete_four_epoch_schedules=schedules,
        native_gradient_restore_and_behavior_required=True,optimizer_updates=0,training_approved=False)
    (a.out/'inputs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete323=True,groups=report['groups'],input_sha256=report['input_sha256'],training_approved=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','capture','control-engine'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',required=True);a=p.parse_args()
    assets=json.loads(ASSETS.read_text());os.environ.update(CS_BASE_DIR=str(PUBLIC),CS_WEIGHTS_DIR=assets['path'],CS_VOICES_DIR=str(Path(assets['path'])/'voices'))
    sys.path.insert(0,assets['source']);prepare(a)
