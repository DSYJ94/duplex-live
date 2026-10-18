"""One bounded text-controller adapter pilot, never native duplex qualification.

Teacher quality and held-out families are verified before allocating the model.
The base is frozen; target-position features are cached without gradients. Only
an explicit output correction trains. Full held-out worker generations run both
before and after, with wrong routes, unanswered requests and lost speech kept.
"""
import argparse
import json
from pathlib import Path
import random
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','assets','kernel-assets','out','baseline'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--function-name-only',action='store_true',help='Match training targets and restored inference to the deployed function-name correction boundary')
    args=ap.parse_args()
    from training.control.verified_data import verified_data
    rows,heldout,data_manifest=verified_data(args.data)
    if len(rows)!=216 or len(heldout)!=72:raise ValueError('Pilot requires the complete predeclared full corpus')
    args.out.mkdir(parents=True,exist_ok=False)
    from evals.moshi_model_download import sha256
    settings=dict(seed=42424242,rank=8,scale=1.,epochs=3,learning_rate=.0005,weight_decay=.01,batch_size=1,
        train_rows=len(rows),heldout_rows=len(heldout),full_worker_generation_required=True,loss='unweighted mean next-token loss on final native action and EOS only',
        data_manifest_sha256=sha256(args.data/'manifest.json'),data_quality_sha256=data_manifest['quality_sha256'],
        base_assets_sha256=sha256(args.assets),fixed_before_training=True,training_step_selection='fixed final step; no held-out checkpoint selection')
    if args.function_name_only:
        settings.update(loss='mean next-token loss only where the actual function-name inference hook applies correction',
            inference_scope='function-name-only',argument_loss_targets=0,content_loss_targets=0,
            boundary_code_sha256=sha256(Path(__file__).with_name('function_boundary.py')))
    (args.out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    import torch
    from training.control.head_adapter import Delta,ResidualHead,save_delta,load_delta,parameter_hashes
    from training.control.curriculum import evaluate_scene
    torch.manual_seed(settings['seed']);torch.cuda.manual_seed_all(settings['seed']);rng=random.Random(settings['seed'])
    model=executor.model;tokenizer=executor.tokenizer;source=dict(model.named_parameters());before=parameter_hashes(source)
    (args.out/'source-parameters-before.json').write_text(json.dumps(before,indent=2)+'\n')
    base=model.lm_head;delta=Delta(base.in_features,base.out_features,settings['rank'],settings['scale']).to(base.weight.device)
    head=ResidualHead(base,delta);model.lm_head=head
    if {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in delta.parameters()}:raise ValueError('Optimizer partition includes a source parameter')
    from training.control.evaluate import verify_baseline,evaluate_suite
    baseline,baseline_proof=verify_baseline(args.baseline,heldout,args.assets)
    (args.out/'baseline-source.json').write_text(json.dumps(baseline_proof,indent=2)+'\n')
    features=[]
    for row in rows:
        prompt=tokenizer.apply_chat_template(row['student_messages'],tools=row['student_tools'],tokenize=False,add_generation_prompt=True,enable_thinking=False)+row['student_assistant_prefill']
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        if prefix!=tokenizer.encode(prompt):raise ValueError('Student tokenizer adds an unaccounted prompt token')
        if not row['target_native_text'].startswith(row['student_assistant_prefill']):raise ValueError('Teacher action does not match native student prefix')
        suffix=row['target_native_text'][len(row['student_assistant_prefill']):]
        target=tokenizer.encode(suffix,add_special_tokens=False)+[tokenizer.eos_token_id];all_ids=prefix+target
        if len(all_ids)>4096 or not prefix or not target:raise ValueError('Training example exceeds explicit text bounds')
        ids=torch.tensor([all_ids[:-1]],device=base.weight.device)
        with torch.no_grad():
            hidden=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)[0][0,len(prefix)-1:].detach().cpu().contiguous()
        if hidden.shape!=(len(target),base.in_features):raise ValueError('Teacher-forcing target alignment differs')
        positions=list(range(len(target)))
        if args.function_name_only:
            from training.control.function_boundary import target_positions
            positions=target_positions(target,lambda ids:tokenizer.decode(ids,skip_special_tokens=False))
        features.append(dict(id=row['id'],hidden=hidden[positions].contiguous(),target=torch.tensor(target,dtype=torch.long)[positions],prompt_tokens=len(prefix),native_target_tokens=len(target),selected_positions=positions))
    (args.out/'feature-inventory.json').write_text(json.dumps([dict(id=f['id'],prompt_tokens=f['prompt_tokens'],target_tokens=len(f['target']),native_target_tokens=f['native_target_tokens'],selected_positions=f['selected_positions']) for f in features],indent=2)+'\n')
    optimizer=torch.optim.AdamW(delta.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay']);steps=0
    with (args.out/'training.jsonl').open('w') as log:
        for epoch in range(settings['epochs']):
            order=list(range(len(features)));rng.shuffle(order)
            for index in order:
                f=features[index];hidden=f['hidden'].to(base.weight.device);target=f['target'].to(base.weight.device)
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(head(hidden).float(),target)
                if not torch.isfinite(loss):raise ValueError('Nonfinite controller training loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in delta.parameters()):raise ValueError('Invalid controller adapter gradients')
                if any(p.grad is not None for p in source.values()):raise ValueError('A published source parameter received gradients')
                torch.nn.utils.clip_grad_norm_(delta.parameters(),1.);optimizer.step();steps+=1
                entry=dict(step=steps,epoch=epoch,id=f['id'],loss=float(loss.detach()),target_tokens=len(target));log.write(json.dumps(entry)+'\n');log.flush()
    if steps!=len(rows)*settings['epochs']:raise ValueError('Incomplete planned training updates')
    provenance=dict(data_manifest_sha256=settings['data_manifest_sha256'],data_quality_sha256=settings['data_quality_sha256'],training_rows=len(rows),optimizer_steps=steps,
        source_producer_commit=data_manifest['producer_commit'],settings_sha256=sha256(args.out/'settings.json'),training_code_sha256=sha256(Path(__file__)))
    metadata=save_delta(args.out/'candidate-adapter',delta,settings['base_assets_sha256'],purpose='trained-development-candidate',provenance=provenance)
    def cpu_tree(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu_tree(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
        return value
    state=dict(optimizer=cpu_tree(optimizer.state_dict()),torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all(),
        python_rng_state=rng.getstate(),completed_steps=steps,completed_epochs=settings['epochs'],settings_sha256=sha256(args.out/'settings.json'),data_manifest_sha256=settings['data_manifest_sha256'])
    state_path=args.out/'candidate-adapter/training-state.pt';torch.save(state,state_path)
    (args.out/'candidate-adapter/training-state.json').write_text(json.dumps(dict(schema='dl-controller-head-training-state-v1',training_state_sha256=sha256(state_path),
        adapter_weights_sha256=metadata['weights_sha256'],settings_sha256=state['settings_sha256'],data_manifest_sha256=state['data_manifest_sha256'],
        completed_steps=steps,completed_epochs=settings['epochs'],torch_version=torch.__version__,optimizer='AdamW',base_assets_sha256=settings['base_assets_sha256']),indent=2)+'\n')
    restored,_=load_delta(args.out/'candidate-adapter',settings['base_assets_sha256'],base.weight.device)
    saved=torch.load(state_path,map_location='cpu',weights_only=True)
    resumed_optimizer=torch.optim.AdamW(restored.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay']);resumed_optimizer.load_state_dict(saved['optimizer'])
    resumed=cpu_tree(resumed_optimizer.state_dict())
    if resumed['param_groups']!=saved['optimizer']['param_groups'] or set(resumed['state'])!=set(saved['optimizer']['state']):raise ValueError('Optimizer checkpoint inventory differs')
    for identity,values in saved['optimizer']['state'].items():
        if set(values)!=set(resumed['state'][identity]):raise ValueError('Optimizer checkpoint fields differ')
        for name,value in values.items():
            if isinstance(value,torch.Tensor):
                if not torch.equal(value,resumed['state'][identity][name]):raise ValueError('Optimizer checkpoint tensor differs')
            elif value!=resumed['state'][identity][name]:raise ValueError('Optimizer checkpoint value differs')
    torch.set_rng_state(saved['torch_rng_state']);torch.cuda.set_rng_state_all(saved['cuda_rng_states']);rng.setstate(saved['python_rng_state'])
    if not torch.equal(torch.get_rng_state(),saved['torch_rng_state']) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng_states'])):raise ValueError('RNG checkpoint restore differs')
    with torch.no_grad():
        fixture=features[0]['hidden'].to(base.weight.device)
        if not torch.equal(head(fixture),ResidualHead(base,restored)(fixture)):raise ValueError('Strict controller checkpoint restore differs')
    # Evaluate exactly what was saved/restored, never an unsaved in-memory arm.
    if args.function_name_only:
        from training.control.scoped_head import FunctionNameHead,FunctionNameGenerator
        model.lm_head=FunctionNameHead(base,restored)
        executor.generator=FunctionNameGenerator(executor.generator,model.lm_head)
    else:model.lm_head=ResidualHead(base,restored)
    adapted=evaluate_suite(executor,heldout,args.out/'restored-adapted')
    after=parameter_hashes(source)
    if before!=after:raise ValueError('Published source weights changed')
    report=dict(schema='dl-controller-head-training-pilot-v1',component_execution_completed=True,settings=settings,load=executor.proof,
        optimizer_steps=steps,source_parameter_count=len(source),all_source_parameter_hashes_unchanged=True,baseline_source=baseline_proof,source_inventory_sha256=sha256(args.out/'source-parameters-before.json'),
        strict_checkpoint_restore_logits_exact=True,optimizer_state_restore_exact=True,rng_state_restore_exact=True,training_state_sha256=sha256(state_path),baseline=baseline,adapted=adapted,adapter_metadata=metadata,
        heldout_rows_used_for_optimizer=0,heldout_checkpoint_selection_used=False,synthetic_text_only=True,speech_qualified=False,native_duplex_qualified=False,
        hybrid_system_qualified=False,production_adopted=False,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(completed=True,baseline=baseline,adapted=adapted,optimizer_steps=steps,production_adopted=False)),flush=True)


if __name__=='__main__':main()
