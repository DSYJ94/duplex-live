"""Zero-update streamed/batched numerical audit on every approved native teacher."""
import json
import hashlib
import re
import subprocess
from contextlib import nullcontext
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
    baseline=None
    if args.teacher_native_fp32_preflight:
        folder=args.teacher_native_fp32_preflight;p=folder/'native-teacher-math-audit.json'
        b=json.loads(p.read_text());commit=args.teacher_native_fp32_source_commit
        if not re.fullmatch('[0-9a-f]{40}',commit):raise RuntimeError('Missing original FP32 source identity')
        content=subprocess.check_output(['git','show',commit+':evals/context_native_teacher_math.py'])
        if (b['schema']!='dl-context-native-teacher-math-audit-v1' or b['optimizer_steps']!=0
                or b['training_approved'] is not False or b['threshold']!=.001
                or b['math_code_sha256']!=hashlib.sha256(content).hexdigest()
                or b['data_loader_code_sha256']!=manifest['data_loader_code_sha256']
                or sha256(folder/'native-teacher-data.json')!=b['data_sha256']
                or json.loads((folder/'native-teacher-data.json').read_text())!=manifest
                or [c['teacher'] for c in b['cases']]!=[r['teacher'] for r in rows]
                or any(c['causal_prefix_maximum_error']!=0. for c in b['cases'])):
            raise RuntimeError('Reference audit must retain the complete matching original FP32 result')
        baseline=dict(report_sha256=sha256(p),source_commit=commit,original_math_code_sha256=b['math_code_sha256'],
            passed=b['passed'],cases=b['cases'])
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):install_adapters(lm,last_layers=4)
    lm.eval().requires_grad_(False);records=[]
    from evals.context_teacher_numeric_reference import linear_reference
    with torch.no_grad(),precise_native(lm),linear_reference() if baseline else nullcontext():
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
    if baseline:
        result.update(schema='dl-context-native-teacher-math-audit-v2',fp32_baseline=baseline,
            reference_precision='fp32-weights-activations-fp64-linear-v1',
            reference_code_sha256=sha256(Path(__file__).with_name('context_teacher_numeric_reference.py')),
            dtype='FP32 weights and activations, FP64 linear arithmetic; FP32 MATH attention; zero adapters; numerical reference only',
            production_and_training_execution_unchanged=True)
    (args.out/'native-teacher-math-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('Native teacher numerical audit failed')
    return result
