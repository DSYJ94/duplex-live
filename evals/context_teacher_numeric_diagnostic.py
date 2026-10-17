"""Isolate numerical reduction differences on one approved native teacher.

This never emits an optimizer approval or changes the production execution path.
"""
import json


def run(lm,spm,voice,examples,args):
    import torch
    from torch.nn import functional as F
    from contextspan.model.sequence_convention import persona_prefix
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_native_teacher_data import load
    from evals.context_training_parity import precise_native
    from evals.personaplex_adapters import install_adapters
    from evals.context_execution import snapshot
    manifest,rows=load(args.teacher_native_math_audit,args.teacher_engine_input_audit,args.assets,
        args.out/'voice.pt',examples,persona_prefix(BASE_INSTRUCTIONS,voice,spm),spm)
    row=next(r for r in rows if r['teacher']['id']==args.teacher_native_math_diagnostic)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):install_adapters(lm,last_layers=4)
    lm.eval().requires_grad_(False);inputs=row['inputs'].cuda();records=[]
    original=F.scaled_dot_product_attention;linear=F.linear;wide={}
    def double_attention(q,k,v,*a,**kw):
        return original(q.double(),k.double(),v.double(),*a,**kw).to(q.dtype)
    def double_linear(x,weight,bias=None):
        key=id(weight)
        if key not in wide:wide[key]=(weight,weight.double())
        if wide[key][0] is not weight:raise RuntimeError('Diagnostic weight identity changed')
        return linear(x.double(),wide[key][1],None if bias is None else bias.double()).to(x.dtype)
    with torch.no_grad(),precise_native(lm):
        execution=snapshot(lm)
        modes=(('original-fp32','original-fp32-repeat','fp64-attention-reductions-only')
            if args.teacher_native_math_diagnostic_mode=='attention' else
            ('fp64-linear-reductions-only','fp64-linear-and-attention-reductions'))
        for mode in modes:
            F.scaled_dot_product_attention=double_attention if 'attention' in mode else original
            F.linear=double_linear if 'linear' in mode else linear
            try:
                pieces=[]
                with lm.streaming(1):
                    for call in row['calls']:
                        f,n=call['start'],call['frames'];h,z=lm.forward_codes(inputs[:,:,f:f+n])
                        pieces.append((h.clone(),z.clone()))
                sh=torch.cat([v[0] for v in pieces],1);sz=torch.cat([v[1] for v in pieces],2)
                bh,bz=lm.forward_codes(inputs);comparisons={}
                for name,a,b in [('hidden',sh,bh),('text_logits',sz,bz)]:
                    delta=(a-b).abs();index=int(delta.flatten().argmax())
                    comparisons[name]=dict(maximum_absolute_error=float(delta.max()),
                        mean_absolute_error=float(delta.mean()),maximum_flat_index=index,
                        maximum_coordinate=list(torch.unravel_index(torch.tensor(index),delta.shape)),
                        reference_absolute_maximum=float(b.abs().max()))
                    comparisons[name]['maximum_coordinate']=[int(x) for x in comparisons[name]['maximum_coordinate']]
                mask=row['mask'].cuda();targets=row['targets'].cuda();s=sz[0,0,mask];b=bz[0,0,mask]
                y=targets[mask];record=dict(mode=mode,comparisons=comparisons,
                    eligible_top1_disagreements=int((s.argmax(-1)!=b.argmax(-1)).sum()),
                    eligible_targets=int(mask.sum()),streamed_ce=float(F.cross_entropy(s,y)),
                    batched_ce=float(F.cross_entropy(b,y)))
                records.append(record);print(json.dumps(record),flush=True)
            finally:
                F.scaled_dot_product_attention=original;F.linear=linear
    result=dict(schema='dl-context-teacher-numeric-diagnostic-v1',teacher=row['teacher'],
        native_data=manifest,execution=execution,comparisons=records,optimizer_steps=0,
        training_approved=False,quality_qualified=False,
        scope='one teacher; FP64 reductions are diagnostic only; original 0.001 preflight unchanged')
    (args.out/'teacher-numeric-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
