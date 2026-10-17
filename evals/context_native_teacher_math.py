"""Zero-update streamed/batched numerical audit on every approved native teacher."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def run(lm,spm,voice,examples,args):
    import torch
    from contextspan.model.sequence_convention import persona_prefix
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_native_teacher_data import load
    from evals.context_training_parity import precise_native
    from evals.personaplex_adapters import install_adapters
    from evals.context_retention_teacher import loss
    manifest,rows=load(args.teacher_native_math_audit,args.teacher_engine_input_audit,args.assets,
        args.out/'voice.pt',examples,persona_prefix(BASE_INSTRUCTIONS,voice,spm),spm)
    data_file=args.out/'native-teacher-data.json';data_file.write_text(json.dumps(manifest,indent=2)+'\n')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):install_adapters(lm,last_layers=4)
    lm.eval().requires_grad_(False);records=[]
    with torch.no_grad(),precise_native(lm):
        for row in rows:
            inputs=row['inputs'].cuda();streamed=[]
            with lm.streaming(1):
                for call in row['calls']:
                    f,n=call['start'],call['frames'];h,z=lm.forward_codes(inputs[:,:,f:f+n])
                    streamed.append((h.detach().clone(),z.detach().clone()))
            sh=torch.cat([v[0] for v in streamed],dim=1);sz=torch.cat([v[1] for v in streamed],dim=2)
            bh,bz=lm.forward_codes(inputs);comparisons={}
            for name,a,b in [('hidden',sh,bh),('text_logits',sz,bz)]:
                if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise RuntimeError('Invalid native teacher numerical output')
                comparisons[name]=dict(maximum_absolute_error=float((a-b).abs().max()),
                    mean_absolute_error=float((a-b).abs().mean()))
            changed=inputs.clone();changed[:,0,-1]=(changed[:,0,-1]+1)%32000
            _,cz=lm.forward_codes(changed)
            causal=float((cz[:,:,:-1]-bz[:,:,:-1]).abs().max())
            y=row['targets'].cuda();mask=row['mask'].cuda();logits=bz[0,0]
            spoken=mask&(y>=5)
            record=dict(teacher=row['teacher'],comparisons=comparisons,causal_prefix_maximum_error=causal,
                teacher_text_ce=float(loss(logits,y,mask)),
                spoken_text_ce=float(torch.nn.functional.cross_entropy(logits[spoken],y[spoken])),
                text_targets=int(mask.sum()),spoken_targets=int(spoken.sum()),
                passed=causal==0. and all(v['maximum_absolute_error']<=.001 for v in comparisons.values()))
            records.append(record);print(json.dumps(record),flush=True)
    result=dict(schema='dl-context-native-teacher-math-audit-v1',passed=all(r['passed'] for r in records),
        data_sha256=sha256(data_file),data_loader_code_sha256=sha256(Path(__file__).with_name('context_native_teacher_data.py')),
        math_code_sha256=sha256(Path(__file__)),cases=records,threshold=.001,
        dtype='FP32 MATH, zero adapters, no graphs/compilation; integer inputs already delayed by native Engine',
        optimizer_steps=0,training_approved=False,quality_qualified=False)
    (args.out/'native-teacher-math-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('Native teacher numerical audit failed')
    return result
