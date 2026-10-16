"""Bounded text-head fit on verified actual-history future features, not task success."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
from evals.moshi_model_download import sha256


def run(lm, spm, voice, examples, args):
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file, save_file
    from evals.context_future_validate import validate
    from evals.personaplex_adapters import install_adapters, adapter_parameters
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_spanning_download import SOURCE_COMMIT
    audits, components = [], []
    for folder in args.actual_head_data:
        audit, component = validate(folder, args.actual_source_data, args.assets, examples, spm, voice)
        audits.append(audit); components.append(component)
    if len({a['source_history_sha256'] for a in audits}) != len(audits):
        raise RuntimeError('Duplicate actual-history component')
    data = {'schema':'dl-context-result-head-data-v1', 'components':audits,
        'all_component_audits_passed':True, 'past_loss_targets':0, 'audio_loss_targets':0,
        'feature_fit_is_not_result_use':True, 'later_features_contain_earlier_teacher_audio':True}
    data_path = args.out/'data.json'
    if args.reload:
        if json.loads(data_path.read_text()) != data: raise RuntimeError('Changed head proof data')
    else: data_path.write_text(json.dumps(data,indent=2)+'\n')
    # Already normalized native features. No second delay, transformer, or out_norm.
    x = torch.cat([c['hidden'][0,c['text_mask']] for c in components]).cuda()
    y = torch.cat([c['text_targets'][c['text_mask']] for c in components]).cuda()
    fact_offsets, cursor = [], 0
    for audit, component in zip(audits, components):
        fact_offsets.append(cursor+int(component['text_mask'][:audit['first_fact_frame']].sum()))
        cursor += int(component['text_mask'].sum())
    facts = torch.tensor(fact_offsets,device='cuda')
    def canonical():
        h = lm.text_linear
        base = F.linear(x.float(),h.weight.float(),None if h.bias is None else h.bias.float())
        if hasattr(h,'lora_A'):
            base = base+F.linear(F.linear(x.float(),h.lora_A.weight),h.lora_B.weight)*h.scaling
        return base
    def outputs():
        with torch.no_grad(): return {'canonical_fp32':canonical(), 'native_bf16':lm.text_linear(x)}
    def digest(values):
        return {k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for k,v in values.items()}
    def metrics(values):
        result = {}
        for name,z in values.items():
            if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite head logits')
            per = F.cross_entropy(z.float(),y,reduction='none')
            result[name] = {'nll':float(per.mean()), 'top1':float((z.argmax(-1)==y).float().mean()),
                'first_fact_nll':float(per[facts].mean()),
                'first_fact_top1':float((z[facts].argmax(-1)==y[facts]).float().mean()),
                'first_fact_predictions':z[facts].argmax(-1).tolist(), 'first_fact_targets':y[facts].tolist()}
        return result
    initial = outputs(); baseline = metrics(initial); zero_digest = digest(initial)
    rng_before = torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        modules = install_adapters(lm,last_layers=4)
    if not torch.equal(rng_before,torch.cuda.get_rng_state()):
        raise RuntimeError('Head adapter construction changed sampling RNG')
    all_adapters = adapter_parameters(lm)
    learned = {n:p for n,p in all_adapters.items() if n.startswith('text_linear.')}
    for name,p in all_adapters.items(): p.requires_grad_(name in learned)
    if len(learned) != 2 or {n for n,p in lm.named_parameters() if p.requires_grad} != set(learned):
        raise RuntimeError('Unexpected head-only trainable parameters')
    frozen = tensor_hashes((n,p) for n,p in lm.named_parameters() if not p.requires_grad)
    if digest(outputs()) != zero_digest: raise RuntimeError('Zero head adapters changed feature logits')
    if args.reload:
        report = json.loads((args.out/'report.json').read_text())
        if (report['schema'] != 'dl-context-result-head-proof-v1' or report['frozen_hashes'] != frozen
                or report['data_sha256'] != sha256(data_path)
                or report['voice_sha256'] != sha256(args.out/'voice.pt')
                or report['assets_manifest_sha256'] != sha256(args.assets)
                or report['checkpoint_sha256'] != sha256(args.out/'adapters.safetensors')):
            raise RuntimeError('Changed head proof artifacts')
        state = load_file(args.out/'adapters.safetensors',device='cuda')
        if set(state) != set(all_adapters): raise RuntimeError('Incomplete adapter keys')
        with torch.no_grad():
            for name,p in all_adapters.items():
                if name not in learned and not torch.equal(p,state[name]):
                    raise RuntimeError('Checkpoint changed frozen non-head adapter')
                p.copy_(state[name])
        actual = digest(outputs()); passed = actual == report['expected_logits']
        result = {'schema':'dl-context-result-head-reload-v1', 'fresh_process':True,
            'all_feature_logits_exact':passed, 'actual':actual, 'checkpoint_sha256':report['checkpoint_sha256'],
            'data_sha256':report['data_sha256'], 'passed':passed}
        (args.out/'reload-report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not passed: raise RuntimeError('Fresh-process head logits changed')
        return
    optimizer = torch.optim.AdamW(list(learned.values()),lr=2e-5,weight_decay=0.)
    steps = []; start = time.monotonic(); seen_a = seen_b = False
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        z = canonical(); loss = F.cross_entropy(z,y)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite head loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):
            raise RuntimeError('Missing or nonfinite head gradient')
        a = int(torch.count_nonzero(lm.text_linear.lora_A.weight.grad))
        b = int(torch.count_nonzero(lm.text_linear.lora_B.weight.grad))
        seen_a |= a>0; seen_b |= b>0
        norm = float(torch.nn.utils.clip_grad_norm_(list(learned.values()),1.))
        optimizer.step()
        row = {'step':step+1,'loss':float(loss.detach()),'nonzero_A_gradients':a,
            'nonzero_B_gradients':b,'gradient_norm_before_clip':norm}
        steps.append(row); print(json.dumps(row),flush=True)
    if not seen_a or not seen_b: raise RuntimeError('Dead head adapter')
    if frozen != tensor_hashes((n,p) for n,p in lm.named_parameters() if not p.requires_grad):
        raise RuntimeError('Frozen native or adapter weights changed')
    final = outputs(); expected = digest(final)
    if any(expected[k] == zero_digest[k] for k in expected):
        raise RuntimeError('Head updates did not affect both feature execution paths')
    save_file({n:p.detach().cpu().contiguous() for n,p in all_adapters.items()},args.out/'adapters.safetensors')
    report = {'schema':'dl-context-result-head-proof-v1', 'source_commit':SOURCE_COMMIT,
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'assets_manifest_sha256':sha256(args.assets), 'voice_reference_sha256':sha256(args.voice_reference),
        'voice_sha256':sha256(args.out/'voice.pt'), 'data_sha256':sha256(data_path),
        'checkpoint_sha256':sha256(args.out/'adapters.safetensors'), 'adapter_modules':modules,
        'last_layers':4,'rank':8,'scaling':2.,'trainable_names':list(learned),
        'trainable_parameters':sum(p.numel() for p in learned.values()), 'optimizer_steps':len(steps),
        'zero_adapter_feature_logits_exact':True,'adapter_construction_rng_preserved':True,
        'frozen_weights_unchanged':True,'frozen_hashes':frozen,'only_text_head_trained':True,
        'past_loss_targets':0,'audio_loss_targets':0,'base_audio_parameters_frozen':True,
        'all_component_audits_passed':True, 'train_components':len(components), 'text_targets':len(y),
        'expected_logits':expected,'baseline_metrics':baseline,'final_metrics':metrics(final),
        'steps':steps,'training_seconds':time.monotonic()-start,'quality_qualified':False,
        'free_running_result_use_proven':False,'audio_behavior_preservation_proven':False,
        'fit_limit':'Later frozen features contain previous teacher audio; native unforced evaluation required',
        'passed':True}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','checkpoint_sha256','baseline_metrics','final_metrics')}),flush=True)
