"""Wait for the original failed whole40 audit, then diagnose only its failed case.

One diagnostic cannot approve a corpus. No optimizer, sampler, production dtype,
threshold, input, target or native call boundary is changed. The only new arithmetic
is FP64 attention inside the independent reference, alongside the old FP64 linear.
"""
import argparse
from datetime import datetime,timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import time

from data.synthetic.native_control_render import sha256
from portable.context.native_preflight_takeover import process


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','base','proof','engine','original-numerical','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--original-pid',type=int,required=True)
    p.add_argument('--original-birth',type=int,required=True);p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    root=Path(__file__).resolve().parents[1];commit=os.environ['DL_TASK_COMMIT']
    def clean():
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=commit
                or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):
            raise ValueError('Exact clean immutable diagnostic source required')
    clean()
    if deadline<=time.time()+60:raise ValueError('Future bounded diagnostic deadline required')
    def timeout(*_):raise TimeoutError('Bounded diagnostic deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    a.out.mkdir(parents=True,exist_ok=False)
    state=dict(schema='dl-native-control40-attention-reference-diagnostic-v1',pid=os.getpid(),
               phase='waiting_for_original_complete_audit',case=a.case,original_pid=a.original_pid,
               original_birth=a.original_birth,deadline_utc=a.deadline_utc,
               optimizer_updates=0,training_approved=False,behavior_qualified=False,
               complete40_retested=False,diagnostic_only=True)
    state['execution_source_commit']=commit
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        tmp=a.out/'diagnostic.partial';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.out/'diagnostic.json')
    save()
    try:
        while time.time()<deadline:
            old=process(a.original_pid)
            if old is not None and old['birth']!=a.original_birth:raise ValueError('Original audit PID was reused')
            if old is None or old['state']=='Z':break
            save();time.sleep(5)
        else:raise TimeoutError('Original audit still live at diagnostic deadline; do not duplicate GPU work')
        clean()
        original=json.loads((a.original_numerical/'numerical.json').read_text())
        if (original['schema']!='dl-native-control40-numerical-v1' or not original['complete40']
                or original['passed'] is not False or not original['all_model_parameters_unchanged']
                or original['threshold']!=.001 or original['optimizer_updates']!=0
                or original['source_commit']!='7769e7cc06f80e609db3f8ac2611977cd0d7cb66'
                or original['code_sha256']!=sha256(Path(__file__).with_name('native_control_numerical.py'))
                or original['reference_code_sha256']!=sha256(Path(__file__).with_name('native_control_reference.py'))):
            raise ValueError('Preserve original complete failed whole40 audit before a diagnostic')
        failed=next(c for c in original['cases'] if c['id']==a.case)
        if failed['passed']:raise ValueError('Only an actually failed case may be diagnosed')
        for _ in range(20):
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if not any(x.strip().isdigit() for x in gpu.splitlines()):break
            time.sleep(.5)
        else:raise RuntimeError('GPU still occupied; no concurrent numerical job allowed')
        state.update(phase='loading_unchanged_native520',original_numerical_sha256=sha256(a.original_numerical/'numerical.json'),
                     original_failed_case=failed,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        save()
        from evals.native_control_numerical import load_model,differences
        from evals.context_training_parity import precise_native
        from evals.native_control_reference import linear_reference
        from evals.native_control_attention_reference import attention_reference
        from evals.moshi_training_smoke import tensor_hashes
        from evals.context_execution import snapshot
        import torch
        lm,parent,adapter=load_model(a)
        if parent!=original['parent']:raise ValueError('Diagnostic changed original parent identity')
        manifest=json.loads((a.engine/'engine-inputs.json').read_text())
        if sha256(a.engine/'engine-inputs.json')!=original['engine_report_sha256']:raise ValueError('Original all40 Engine inputs changed')
        record=next(r for r in manifest['cases'] if r['id']==a.case)
        saved=torch.load(a.engine/a.case/record['file'],weights_only=True,map_location='cpu')
        inputs=saved['inputs'].cuda();before=tensor_hashes(lm.named_parameters());execution=snapshot(lm)
        state['phase']='fp64_attention_reference_comparison';save()
        with torch.no_grad(),precise_native(lm),linear_reference() as linear,attention_reference() as attention:
            bh,bz=lm.forward_codes(inputs);hidden=[];logits=[]
            with lm.streaming(1):
                for call in record['calls']:
                    i,n=call['start'],call['frames'];h,z=lm.forward_codes(inputs[:,:,i:i+n])
                    hidden.append(h.detach().clone());logits.append(z.detach().clone())
            sh=torch.cat(hidden,dim=1);sz=torch.cat(logits,dim=2)
            comparisons=dict(hidden=differences(bh,sh),text_logits=differences(bz,sz))
            error=(bz-sz).abs();flat=int(error.flatten().argmax());location=list(torch.unravel_index(torch.tensor(flat,device=error.device),error.shape))
            state.update(comparisons=comparisons,maximum_logit_error_location=[int(x) for x in location],
                         reference_linear=dict(linear),reference_attention=dict(attention),
                         within_original_threshold=all(x['maximum_absolute_error']<=.001 for x in comparisons.values()))
        state.update(phase='complete_diagnostic_only',all_model_parameters_unchanged=before==tensor_hashes(lm.named_parameters()),
                     original_execution=execution,execution_policy_unchanged=execution==snapshot(lm),
                     reference_module_sha256=sha256(Path(__file__).with_name('native_control_attention_reference.py')))
        save()
    except BaseException as error:
        state.update(phase='failed',error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
