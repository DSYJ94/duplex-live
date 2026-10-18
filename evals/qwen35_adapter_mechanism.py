"""GPU mechanics only: random numerical fixture, never a trained model candidate."""
import argparse
import json
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from evals.qwen35_controller import load_model
    model,tokenizer,load=load_model(args.assets,kernel_assets=args.kernel_assets)
    import torch
    from training.control.head_adapter import Delta,ResidualHead,save_delta,load_delta,parameter_hashes
    from evals.moshi_model_download import sha256
    torch.manual_seed(42424242);torch.cuda.manual_seed_all(42424242)
    source=dict(model.named_parameters());assert all(not p.requires_grad for p in source.values())
    before=parameter_hashes(source);(args.out/'source-parameters-before.json').write_text(json.dumps(before,indent=2)+'\n')
    base=model.lm_head;delta=Delta(base.in_features,base.out_features).to(base.weight.device);head=ResidualHead(base,delta)
    hidden=torch.randn(2,base.in_features,device=base.weight.device,dtype=base.weight.dtype)
    with torch.no_grad():assert torch.equal(base(hidden),head(hidden))
    model.lm_head=head
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in delta.parameters()}
    # A real published-model forward proves integration. No fixture text enters TRAIN.
    ids=tokenizer('Say ready.',return_tensors='pt').to(base.weight.device)
    with torch.no_grad():
        actual=model(**ids,use_cache=False,logits_to_keep=1).logits
        model.lm_head=base;original=model(**ids,use_cache=False,logits_to_keep=1).logits;model.lm_head=head
        assert torch.equal(actual,original)
    labels=torch.tensor([17,41],device=base.weight.device);optimizer=torch.optim.AdamW(delta.parameters(),lr=.001,weight_decay=0.)
    grads=[];losses=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(head(hidden).float(),labels);loss.backward()
        if not torch.isfinite(loss):raise ValueError('Nonfinite mechanism loss')
        row={k:float(v.grad.detach().abs().max()) for k,v in delta.named_parameters()}
        if any(not torch.isfinite(v.grad).all() for v in delta.parameters()):raise ValueError('Nonfinite adapter gradient')
        grads.append(row);losses.append(float(loss));optimizer.step()
    assert grads[0]['up']>0 and grads[1]['down']>0 and all(p.grad is None for p in source.values())
    metadata=save_delta(args.out/'fixture-adapter',delta,sha256(args.assets),purpose='mechanism-test-only',provenance=dict(random_numerical_fixture=True,conversation_training_rows=0,fixture_optimizer_steps=2))
    try:load_delta(args.out/'fixture-adapter',sha256(args.assets),base.weight.device)
    except ValueError:fixture_rejected=True
    else:raise AssertionError('Fixture incorrectly accepted as a model candidate')
    restored,_=load_delta(args.out/'fixture-adapter',sha256(args.assets),base.weight.device,allow_mechanism_fixture=True)
    with torch.no_grad():assert torch.equal(head(hidden),ResidualHead(base,restored)(hidden))
    after=parameter_hashes(source);assert before==after
    report=dict(schema='dl-controller-head-mechanism-v1',all_mechanism_checks_passed=True,load=load,
        source_parameter_count=len(source),all_source_parameter_hashes_unchanged=True,source_parameter_inventory_sha256=sha256(args.out/'source-parameters-before.json'),
        zero_delta_real_model_forward_exact=True,zero_delta_output_exact=True,gradient_maxima=grads,fixture_losses=losses,
        adapter_parameters=sum(p.numel() for p in delta.parameters()),strict_save_restore_logits_exact=True,fixture_rejected_as_candidate=fixture_rejected,
        fixture_optimizer_steps=2,conversation_optimizer_steps=0,training_data_used=False,fixture_only=True,production_adopted=False,duplex_qualified=False,
        adapter_metadata=metadata,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(mechanics_passed=True,source_weights_unchanged=True,fixture_only=True)),flush=True)


if __name__=='__main__':main()
