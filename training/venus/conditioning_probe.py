"""Fixed-history native speech-conditioning measurement, with ZERO updates.

A numerical difference is not proof of audible damage. No thresholds here adopt
or reject a model. Run only on approved TRAIN/DEV caches, after the live server
has stopped; the GPU runtime/guard remain the caller's responsibility.
"""
import torch
from .lora import disabled,selected_forward


def post_token_example(llm,example,device):
    if example['fold'] not in ('TRAIN','DEV'):
        raise ValueError('Conditioning diagnosis excludes held-out data')
    mask=~example['decision_mask'].bool()
    positions=example['positions'][mask]+1
    ids=example['targets'][mask]
    if not len(ids) or int(positions.max())>=len(example['embeddings']):
        raise ValueError('Complete spoken post-token positions are required')
    ends={int(s['start']+s['length']-1) for s in example['segments'] if s['length']==1}
    if any(int(p) not in ends for p in positions):
        raise ValueError('Expected a native single-token feed at every post-token state')
    with torch.no_grad():
        expected=llm.model.embed_tokens(ids.to(device)).detach().cpu()
    if not torch.equal(expected,example['embeddings'][positions]):
        raise ValueError('Post-token inputs differ from the exact fixed target embeddings')
    result=dict(example,positions=positions)
    return result,ids


def tts_condition(tts,ids,hidden):
    """Match the acquired duplex _convert_results_to_tts_input before audio BOS."""
    device=tts.emb_text.weight.device
    dtype=next(tts.projector_semantic.parameters()).dtype
    projected=tts.projector_semantic(hidden.to(device=device,dtype=dtype))
    semantic=torch.nn.functional.normalize(projected,p=2,dim=-1)
    return tts.emb_text(ids.to(device))+semantic,semantic


def compare_fixed_history(model,example,device):
    post,ids=post_token_example(model.llm,example,device)
    with torch.no_grad():
        with disabled(model.llm):
            _,base=selected_forward(model.llm,post,device)
            base_condition,base_semantic=tts_condition(model.tts,ids,base)
        _,adapted=selected_forward(model.llm,post,device)
        adapted_condition,adapted_semantic=tts_condition(model.tts,ids,adapted)
        delta=(adapted_condition.float()-base_condition.float())
        cosine=torch.nn.functional.cosine_similarity(base_semantic.float(),adapted_semantic.float(),dim=-1)
    if not torch.isfinite(delta).all() or not torch.isfinite(cosine).all():
        raise ValueError('Nonfinite native TTS conditioning')
    return dict(id=example['id'],fold=example['fold'],family=example['family'],spoken_tokens=len(ids),
        exact_target_embedding_alignment=True,fixed_token_history=True,fixed_native_segment_schedule=True,
        post_token_states=True,hidden_mse=float((adapted-base).square().mean()),
        condition_mse=float(delta.square().mean()),condition_max_abs=float(delta.abs().max()),
        semantic_cosine_mean=float(cosine.mean()),semantic_cosine_min=float(cosine.min()),
        optimizer_updates=0,audible_quality_measured=False,model_adoption=False)


def diagnostic_rows(manifest):
    """Select before reading losses/outputs; one TRAIN and DEV per family."""
    families=('ordinary','delegate','quiet','cancel','pause','pending')
    rows=[]
    for fold in ('TRAIN','DEV'):
        for family in families:
            matches=sorted((r for r in manifest['scenes'] if r['fold']==fold and r['family']==family),key=lambda r:r['id'])
            if not matches:raise ValueError('Missing diagnostic family: '+fold+'/'+family)
            rows.append(matches[0])
    return rows


def main(a):
    import hashlib,json,os,subprocess
    from pathlib import Path
    from .native_cache import load_native
    from .inference import load_candidate
    from .train import assert_deadline,read_example
    from portable.venus.native_app import ensure_guard
    ensure_guard();assert_deadline(a.lease,360)
    # Never load a second model beside a live conversation or another GPU job.
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    if any(line.strip() and int(line.strip())!=os.getpid() for line in running.splitlines()):
        raise RuntimeError('Another GPU process is active; conditioning probe must run sequentially')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text())
    if digest(cache/'manifest.json')!=a.manifest_sha256 or digest(a.checkpoint)!=a.checkpoint_sha256:
        raise ValueError('Exact approved cache/checkpoint hashes required')
    approval=json.loads(Path(a.approval).read_text())
    if not approval['approved'] or approval['manifest_sha256']!=manifest['source_manifest_sha256'] or manifest['proof_only']:
        raise ValueError('Complete approved native corpus required')
    if len(manifest['scenes'])!=268 or manifest['prefix_producer']!='8fea2b1b1720c6744390006cef435b6a80dddfd4':
        raise ValueError('Use the existing exact complete native prefix cache; no recaching')
    chosen=diagnostic_rows(manifest);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'selection.json').write_text(json.dumps(dict(rows=chosen,selection='first sorted ID per TRAIN/DEV and all six families, before outputs',adoption_evaluation=False),indent=2)+'\n')
    model,_=load_native(a.model);profile=load_candidate(model.llm,a.checkpoint)
    if profile['optimizer_updates']!=244:raise ValueError('This diagnosis is pinned to actual244')
    results=[]
    for row in chosen:
        ensure_guard();assert_deadline(a.lease,180)
        result=compare_fixed_history(model,read_example(cache,row),'cuda');results.append(result)
        print(json.dumps(result),flush=True)
    report=dict(complete=len(results)==12,source=a.source,checkpoint_sha256=a.checkpoint_sha256,
        manifest_sha256=a.manifest_sha256,rows=results,optimizer_updates=0,
        numerical_conditioning_only=True,audible_damage_established=False,adopted=False)
    (out/'conditioning.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model','cache','manifest-sha256','checkpoint','checkpoint-sha256','approval','lease','source','output'):
        parser.add_argument('--'+key,required=True)
    main(parser.parse_args())
