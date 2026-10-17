"""Check batched versus streamed temporal computation on actual corrective histories."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def audit(lm, folders, components, audits, out):
    import torch
    from evals.context_training_parity import precise_native
    from evals.personaplex_adapters import install_adapters
    records = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        install_adapters(lm,last_layers=4)
    lm.eval().requires_grad_(False)
    with torch.no_grad(), precise_native(lm):
        for folder, component, approval in zip(folders,components,audits):
            early = (folder/'early-future.json').exists()
            from evals.context_component_kind import manifest_name
            manifest = json.loads((folder/manifest_name(folder,mixed=True)).read_text())
            history = Path(manifest['actual_history'])
            record = json.loads((history/'student-history.json').read_text())
            raw = torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)['delayed_inputs']
            live = [c for c in record['calls'] if c['mode']=='live']
            stop = live[manifest['future_first_real_frame']]['start']
            calls = [c for c in record['calls'] if c['start'] < stop]
            if calls[-1]['start']+calls[-1]['frames'] != stop:
                raise RuntimeError('Actual prefix cut splits a native forward')
            # These tensors already are consumed temporal inputs, never delay again.
            sequence = torch.cat([raw[:,:,:stop],component['delayed_inputs']],dim=2).cuda()
            if sequence.shape[-1] >= 1024: raise RuntimeError('Bounded audit exceeds tested context')
            streamed = []
            with lm.streaming(1):
                for call in calls:
                    lm.forward_codes(sequence[:,:,call['start']:call['start']+call['frames']])
                future_calls = manifest['calls'] if early else [
                    {'start':n,'frames':1} for n in range(component['hidden'].shape[1])]
                for call in future_calls:
                    n,count = call['start'],call['frames']
                    h,z = lm.forward_codes(sequence[:,:,stop+n:stop+n+count])
                    streamed.append((h.detach().clone(),z.detach().clone()))
            sh = torch.cat([v[0] for v in streamed],dim=1)
            sz = torch.cat([v[1] for v in streamed],dim=2)
            bh,bz = lm.forward_codes(sequence)
            bh,bz = bh[:,stop:],bz[:,:,stop:]
            comparisons = {}
            for name,a,b in [('hidden',sh,bh),('text_logits',sz,bz)]:
                if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise RuntimeError('Invalid actual temporal audit output')
                comparisons[name] = {'maximum_absolute_error':float((a-b).abs().max()),
                    'mean_absolute_error':float((a-b).abs().mean())}
            # Change only the final future input: no earlier logits may change.
            changed = sequence.clone(); changed[:,0,-1] = (changed[:,0,-1]+1)%32000
            _,cz = lm.forward_codes(changed)
            causal_error = float((cz[:,:,:-1]-lm.forward_codes(sequence)[1][:,:,:-1]).abs().max())
            mask = component['text_mask'].cuda(); target = component['text_targets'].cuda()
            logits = bz[0,0]
            nll = torch.nn.functional.cross_entropy(logits[mask],target[mask])
            numerical = max(c['maximum_absolute_error'] for c in comparisons.values()) <= .001
            nonfact = 'diagnostic_frame' in approval
            diagnostic_frame,diagnostic_target = ((approval['diagnostic_frame'],approval['diagnostic_target']) if nonfact
                else (approval['first_fact_frame'],approval['first_fact_target']))
            probability_name=('request_target_probability_fp32' if approval.get('diagnostic_kind')=='backend_request'
                else 'first_content_target_probability_fp32' if nonfact else 'first_fact_target_probability_fp32')
            probability = float(logits[diagnostic_frame].softmax(-1)[diagnostic_target])
            item = {**approval, 'prefix_conditioning_frames':stop, 'past_loss_targets':0,
                'already_delayed_inputs_used_without_reapplying_delay':True,
                'comparisons':comparisons,'causal_prefix_maximum_error':causal_error,
                'native_bf16_feature_difference_diagnostic_only':float((sh.cpu()-component['hidden'].float()).abs().max()),
                'zero_adapter_future_text_nll_fp32':float(nll),
                probability_name:probability,
                'passed':numerical and causal_error == 0.}
            records.append(item)
            print(json.dumps(item),flush=True)
    result = {'schema':'dl-context-actual-temporal-audit-v1','dtype':'float32, MATH attention, no graphs/compile',
        'threshold':.001,'components':records,'data_sha256':sha256(out/'data.json'),
        'passed':all(r['passed'] for r in records), 'optimizer_steps':0,
        'free_running_or_audio_quality_qualified':False}
    (out/'actual-temporal-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']: raise RuntimeError('Actual-history streamed/batched temporal audit failed')
    return result
