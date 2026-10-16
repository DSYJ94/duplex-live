"""Bounded resumable actual-history adaptation with independent TRAIN retention."""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from evals.moshi_model_download import sha256


def run(lm, spm, voice, examples, components, audits, args):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from safetensors.torch import save_file, load_file
    from contextspan.moshi.models.lm import _delay_sequence
    from contextspan.moshi.utils.compile import no_compile, no_cuda_graph
    from evals.personaplex_adapters import install_adapters, adapter_parameters, checkpoint_language_layers
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_spanning_download import SOURCE_COMMIT
    from evals.context_training_state import save as save_state, restore as restore_state
    pre = args.actual_temporal_preflight
    gate = json.loads((pre/'actual-temporal-audit.json').read_text())
    if (gate['schema'] != 'dl-context-actual-temporal-audit-v1' or gate['passed'] is not True
            or gate['data_sha256'] != sha256(args.out/'data.json') or gate['optimizer_steps'] != 0
            or len(gate['components']) != len(audits)):
        raise RuntimeError('Actual temporal fit requires the passing matching zero-update audit')
    for checked, audit in zip(gate['components'],audits):
        if (any(checked.get(k) != v for k,v in audit.items()) or checked['passed'] is not True
                or checked['causal_prefix_maximum_error'] != 0.
                or set(checked['comparisons']) != {'hidden','text_logits'}
                or any(not 0. <= v['maximum_absolute_error'] <= .001 for v in checked['comparisons'].values())):
            raise RuntimeError('Actual temporal audit changed or failed')
    if not torch.equal(voice,torch.load(pre/'voice.pt',map_location='cpu',weights_only=True)['codes']):
        raise RuntimeError('Temporal preflight used different voice codes')
    embedded = args.out/'approved-temporal-audit.json'
    if args.reload:
        if sha256(embedded) != sha256(pre/'actual-temporal-audit.json'):
            raise RuntimeError('Changed packaged temporal audit')
    else:
        embedded.write_bytes((pre/'actual-temporal-audit.json').read_bytes())
    batches = []
    for folder, component in zip(args.actual_head_data,components):
        name = 'early-future.json' if args.actual_early_data else 'actual-future.json'
        m = json.loads((folder/name).read_text()); h = Path(m['actual_history'])
        record = json.loads((h/'student-history.json').read_text())
        raw = torch.load(h/'student-history.pt',map_location='cpu',weights_only=True)['delayed_inputs']
        live = [c for c in record['calls'] if c['mode']=='live']
        stop = live[m['future_first_real_frame']]['start']
        inputs = torch.cat([raw[:,:,:stop],component['delayed_inputs']],dim=2).cuda()
        mask = torch.zeros(inputs.shape[-1],dtype=torch.bool,device='cuda')
        mask[stop:] = component['text_mask'].cuda()
        target = torch.zeros(inputs.shape[-1],dtype=torch.int64,device='cuda')
        target[stop:] = component['text_targets'].cuda()
        if mask[:stop].any() or int(mask.sum()) != int(component['text_mask'].sum()):
            raise RuntimeError('Actual past accidentally acquired labels')
        batches.append((inputs,mask,target))
    retention = []
    for kind in ('private','simple','clarify','cancel'):
        e = max((e for e in examples if e['record']['kind']==kind),key=lambda e:e['codes'].shape[-1])
        codes = e['codes'][None].cuda(); initial = lm._get_initial_token()
        inputs = torch.cat([initial,_delay_sequence(lm.delays,codes,initial)],dim=2)[:,:,:-1]
        retention.append((inputs,e['text_mask'].cuda(),e['record']))
    def forward(inputs): return lm.forward_codes(inputs)
    def digests():
        result = {}
        with torch.no_grad():
            for index,(audit,(inputs,mask,_)) in enumerate(zip(audits,batches)):
                z = forward(inputs)[1][0,0,mask].detach()
                if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite valid temporal logits')
                result[str(index)+':'+audit['scenario']] = hashlib.sha256(z.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        return result
    def metrics():
        results = []
        with torch.no_grad():
            for audit,(inputs,mask,target) in zip(audits,batches):
                z = forward(inputs)[1][0,0]; pos = int(torch.where(mask)[0][0])
                y = target[mask]; z = z[mask].float()
                weights = torch.where(y==3,.5,1.)
                ce = F.cross_entropy(z,y,reduction='none')
                results.append({'scenario':audit['scenario'],'nll':float((ce*weights).sum()/weights.sum()),
                    'top1':float((z.argmax(-1)==y).float().mean()),
                    'non_pad_nll':float(ce[y!=3].mean()),
                    'silence_targets':int((y==3).sum()),
                    'silence_top1':float((z[y==3].argmax(-1)==3).float().mean()) if (y==3).any() else None,
                    'first_supervised_token_target':int(target[pos]),
                    'first_supervised_token_probability':float(z[0].softmax(-1)[target[pos]])})
        return results
    def state_metadata(frozen):
        return {'source_commit':SOURCE_COMMIT,
            'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'assets_manifest_sha256':sha256(args.assets),'data_sha256':sha256(args.out/'data.json'),
            'native_protocol':{'name':'bootstrap-preserve-provided-v1',
                'code_sha256':sha256(Path(__file__).with_name('context_native_layout.py')),
                'attention_backend':'math'},
            'voice_sha256':sha256(args.out/'voice.pt'),'frozen_hashes':frozen,
            'retention_examples':[r[2] for r in retention],
            'recipe':{'rank':8,'scaling':2.,'last_layers':4,'lr':1e-5,'weight_decay':0.,
                'gradient_clip':1.,'retention_kl_weight':1.,'retention_hidden_mse_weight':10.,
                'text_pad3_loss_weight':.5,
                'data_order':'actual step modulo component count; retention step modulo four',
                'actual_component_count':len(batches),'past_loss_targets':0,'audio_loss_targets':0,
                'execution':'BF16 batch MATH, no compilation or CUDA graphs'}}
    with no_compile(), no_cuda_graph(), sdpa_kernel(SDPBackend.MATH):
        baseline = digests(); baseline_metrics = metrics()
        frozen = tensor_hashes(lm.named_parameters())
        references = []
        with torch.no_grad():
            for inputs,mask,_ in retention:
                h,z = forward(inputs)
                references.append((h[:,mask].detach().clone(),z[0,0,mask].float().softmax(-1).detach()))
        rng = torch.cuda.get_rng_state().clone()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(17); torch.cuda.manual_seed_all(17)
            modules = install_adapters(lm,last_layers=4)
        if not torch.equal(rng,torch.cuda.get_rng_state()) or digests() != baseline:
            raise RuntimeError('Zero temporal adapters changed native logits or RNG')
        learned = adapter_parameters(lm)
        if {n for n,p in lm.named_parameters() if p.requires_grad} != set(learned):
            raise RuntimeError('Unexpected temporal trainable parameters')
        if args.reload:
            report = json.loads((args.out/'report.json').read_text())
            if (report['schema'] != 'dl-context-actual-adapter-proof-v1' or report['frozen_hashes'] != frozen
                    or report['retention_examples'] != [r[2] for r in retention]
                    or report['data_sha256'] != sha256(args.out/'data.json')
                    or report['voice_sha256'] != sha256(args.out/'voice.pt')
                    or report['assets_manifest_sha256'] != sha256(args.assets)
                    or report['actual_temporal_audit_sha256'] != sha256(pre/'actual-temporal-audit.json')
                    or report['checkpoint_sha256'] != sha256(args.out/'adapters.safetensors')):
                raise RuntimeError('Changed actual adapter proof')
            state = load_file(args.out/'adapters.safetensors',device='cuda')
            if set(state) != set(learned): raise RuntimeError('Incomplete actual adapter keys')
            with torch.no_grad():
                for n,p in learned.items():p.copy_(state[n])
            resume_verified = None
            if report.get('training_state_sha256'):
                if report['training_state_sha256'] != sha256(args.out/'training-state.pt'):
                    raise RuntimeError('Changed portable training state')
                before = tensor_hashes(learned.items())
                opt = torch.optim.AdamW(list(learned.values()),lr=1e-5,weight_decay=0.)
                snapshot = restore_state(args.out/'training-state.pt',learned,opt,state_metadata(frozen))
                resume_verified = (snapshot['completed_steps']==report['optimizer_steps']
                    and before==tensor_hashes(learned.items()))
                if not resume_verified:raise RuntimeError('Portable state differs from saved adapter')
            actual = digests(); passed = actual == report['expected_logits']
            result = {'schema':'dl-context-actual-adapter-reload-v1','fresh_process':True,
                'all_valid_logits_exact':passed,'actual':actual,'checkpoint_sha256':report['checkpoint_sha256'],
                'data_sha256':report['data_sha256'],'optimizer_rng_state_restored':resume_verified,
                'training_state_sha256':report.get('training_state_sha256'),'passed':passed}
            (args.out/'reload-report.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
            if not passed: raise RuntimeError('Fresh actual adapter logits changed')
            return
        optimizer = torch.optim.AdamW(list(learned.values()),lr=1e-5,weight_decay=0.)
        initial_step = 0
        if args.resume_training:
            if args.actual_fit_updates>8:
                parent = json.loads((args.resume_training.parent/'report.json').read_text())
                loaded = json.loads((args.resume_training.parent/'reload-report.json').read_text())
                if (not parent['passed'] or not loaded['passed'] or not loaded['optimizer_rng_state_restored']
                        or parent['training_state_sha256'] != sha256(args.resume_training)
                        or loaded['training_state_sha256'] != parent['training_state_sha256']):
                    raise RuntimeError('Longer resumed adaptation lacks a verified portable preflight')
            snapshot = restore_state(args.resume_training,learned,optimizer,state_metadata(frozen))
            initial_step = snapshot['completed_steps']
        steps = []; seen_a = seen_b = False; start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for local_step in range(args.actual_fit_updates):
            step = initial_step+local_step
            optimizer.zero_grad(set_to_none=True)
            inputs,mask,target = batches[step%len(batches)]
            with checkpoint_language_layers(lm):
                logits = forward(inputs)[1][0,0,mask].float()
                y = target[mask]; weights = torch.where(y==3,.5,1.)
                ce = (F.cross_entropy(logits,y,reduction='none')*weights).sum()/weights.sum()
                ce.backward()
                ri = step%len(retention); inputs,mask,_ = retention[ri]
                h,z = forward(inputs); base_h,prob = references[ri]
                kl = F.kl_div(z[0,0,mask].float().log_softmax(-1),prob,reduction='batchmean')
                hidden = (h[:,mask].float()-base_h.float()).square().mean()
                regularizer = kl+10.*hidden; regularizer.backward()
            if not all(torch.isfinite(v) for v in (ce,kl,hidden)):
                raise RuntimeError('Nonfinite actual or retention loss')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):
                raise RuntimeError('Missing/nonfinite actual temporal gradient')
            a = sum(int(torch.count_nonzero(p.grad)) for n,p in learned.items() if '.lora_A.' in n)
            b = sum(int(torch.count_nonzero(p.grad)) for n,p in learned.items() if '.lora_B.' in n)
            seen_a |= a>0; seen_b |= b>0
            norm = float(torch.nn.utils.clip_grad_norm_(list(learned.values()),1.))
            if not math.isfinite(norm): raise RuntimeError('Nonfinite actual gradient norm')
            optimizer.step()
            save_state(args.out/'training-state.pt',learned,optimizer,step+1,state_metadata(frozen))
            row = {'step':step+1,'scenario':audits[step%len(batches)]['scenario'],
                'future_text_ce':float(ce.detach()),'retention_kind':retention[ri][2]['kind'],
                'retention_kl':float(kl.detach()),'retention_hidden_mse':float(hidden.detach()),
                'nonzero_A_gradients':a,'nonzero_B_gradients':b,'gradient_norm_before_clip':norm}
            steps.append(row); print(json.dumps(row),flush=True)
        if not seen_a or not seen_b: raise RuntimeError('Dead actual temporal adapter')
        if frozen != tensor_hashes((n,p) for n,p in lm.named_parameters() if not p.requires_grad):
            raise RuntimeError('Frozen speech/base weights changed')
        expected = digests()
        if expected == baseline: raise RuntimeError('Actual temporal updates changed no valid logits')
        save_file({n:p.detach().cpu().contiguous() for n,p in learned.items()},args.out/'adapters.safetensors')
        report = {'schema':'dl-context-actual-adapter-proof-v1','source_commit':SOURCE_COMMIT,
            'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'assets_manifest_sha256':sha256(args.assets),'data_sha256':sha256(args.out/'data.json'),
            'voice_sha256':sha256(args.out/'voice.pt'),'voice_reference_sha256':sha256(args.voice_reference),
            'checkpoint_sha256':sha256(args.out/'adapters.safetensors'),
            'actual_temporal_audit_sha256':sha256(pre/'actual-temporal-audit.json'),
            'actual_temporal_preflight':str(pre),'all_actual_temporal_audits_passed':True,
            'adapter_modules':modules,'last_layers':4,'rank':8,'scaling':2.,
            'trainable_parameters':sum(p.numel() for p in learned.values()),'optimizer_steps':initial_step+len(steps),
            'updates_this_run':len(steps),'initial_optimizer_steps':initial_step,
            'resumed_from_training_state':str(args.resume_training) if args.resume_training else None,
            'training_state_sha256':sha256(args.out/'training-state.pt'),
            'past_loss_targets':0,'audio_loss_targets':0,'all_component_audits_passed':True,
            'native_protocol':state_metadata(frozen)['native_protocol'],
            'zero_adapter_native_batch_logits_exact':True,'adapter_construction_rng_preserved':True,
            'frozen_weights_unchanged':True,'frozen_hashes':frozen,'base_audio_parameters_frozen':True,
            'retention_examples':[r[2] for r in retention],
            'retention_objective':'KL on unchanged base text distribution plus 10x hidden MSE, approved independent TRAIN histories',
            'baseline_metrics':baseline_metrics,'final_metrics':metrics(),'expected_logits':expected,'steps':steps,
            'execution':'native BF16 batch MATH attention, no compilation/graphs; actual FP32 causal preflight required',
            'training_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'quality_qualified':False,'free_running_result_use_proven':False,
            'audio_behavior_preservation_proven':False,'passed':True}
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('passed','checkpoint_sha256','final_metrics','peak_allocated_bytes')}),flush=True)
