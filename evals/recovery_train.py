"""Bounded native LoRA adaptation on verified fresh, causally conditioned data.

Validation uses teacher forcing and is diagnostic. Final qualification requires
separate generated-speech, delegation and listening evaluations.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_data import validate_cache


def load_base(manifest):
    import torch
    from moshi.models import LMModel,loaders
    rag = Path(manifest["rag"]["path"])
    for name in ("config.json","model.safetensors"):
        if sha256(rag / name) != manifest["rag"]["files"][name]["sha256"]:
            raise RuntimeError("Pinned pretrained RAG weights/config changed")
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo=None,moshi_weights=rag / "model.safetensors",
        mimi_weights=rag / "tokenizer-e351c8d8-checkpoint125.safetensors",
        tokenizer=rag / "tokenizer_spm_32k_3.model",config_path=rag / "config.json")
    original = LMModel.load_state_dict
    audit = []
    def strict(module,*args,**kwargs):
        result = original(module,*args,**kwargs)
        extras = [k for k in result.unexpected_keys if not k.startswith("condition_provider.conditioners.reference_with_time.")]
        if result.missing_keys or extras:
            raise RuntimeError("Incomplete base speech checkpoint")
        audit.append(len(result.unexpected_keys))
        return result
    LMModel.load_state_dict = strict
    try:
        model = info.get_moshi(device="cuda",dtype=torch.bfloat16,skip_conditioners=["reference_with_time"])
    finally:
        LMModel.load_state_dict = original
    if len(audit) != 1:
        raise RuntimeError("Missing base checkpoint load audit")
    return model.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=32)
    parser.add_argument("--rag-weight",type=float,default=1.)
    parser.add_argument("--result-weight",type=float,default=1.)
    parser.add_argument("--reload",action="store_true")
    args = parser.parse_args()
    if not 4 <= args.steps <= 128:
        parser.error("Bounded fresh pilot requires 4–128 optimizer steps")
    if not 1 <= args.rag_weight <= 16 or not 1 <= args.result_weight <= 8:
        parser.error("Control/result weights exceed the bounded development range")
    cache = validate_cache(args.cache)
    manifest = json.loads(args.manifest.read_text())
    model = load_base(manifest)
    if args.reload:
        from evals.moshi_rag_adapter_pilot import verify_reload
        verify_reload(model,args.out)
        return
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from moshi.models.loaders import get_lora_moshi
    from moshi.utils.compile import no_compile
    from moshi.inference_utils.utils import get_condition_tensors
    from evals.moshi_rag_adapter_pilot import adapter_parameters,precise_sample
    from evals.moshi_training_smoke import tensor_hashes
    from evals.moshi_timed_reference import forward_with_reference
    sys.path.insert(0,"/projects/u6si/duplex-live/vendor/moshi-finetune-2acc879fe7c48f885a18f6cc9548bccb2674d87b")
    from finetune.mixed_precision import prepare_mixed_precision,upcast_mixed_precision,downcast_mixed_precision

    args.out.mkdir(parents=True,exist_ok=False)
    train = [r for r in cache["examples"] if r["split"] == "train"]
    validation = [r for r in cache["examples"] if r["split"] == "validation"]
    if len(train) < 16 or len(validation) < 4:
        raise RuntimeError("Insufficient verified fresh folds")
    def sample(row):
        state = torch.load(args.cache / row["file"],map_location="cpu",weights_only=True)
        return state["codes"].cuda(),state["reference"].cuda()
    def loss_and_metrics(row):
        codes,reference = sample(row)
        conditions = get_condition_tensors("moshi",model,1,1.,first_speaker="user")
        with no_compile():
            result = forward_with_reference(model,codes,reference,conditions)
        text_target = codes[:,:1][result.text_mask]
        text_logits = result.text_logits[result.text_mask].float()
        nll = F.cross_entropy(text_logits,text_target,reduction="none")
        weights = torch.where(text_target == 3,.5,1.)
        weights[text_target == 4] *= args.rag_weight
        result_words = torch.zeros_like(text_target,dtype=torch.bool)
        if row["kind"] in ("private","clarify","cancel"):
            sidecar = json.loads((Path(cache["conversion"]) / row["scenario_id"] / "conversation.json").read_text())
            last = [e for e in sidecar["timeline"]["events"] if e["ch"] == "assistant" and e["kind"] == "utterance"][-1]
            times = torch.arange(codes.shape[-1],device=codes.device)*.08
            result_window = ((times >= last["actual_start_ms"]/1000)&(times < last["actual_end_ms"]/1000))[None,None]
            result_words = result_window.expand_as(result.text_mask)[result.text_mask] & (text_target >= 5)
            weights[result_words] *= args.result_weight
        text_loss = (nll*weights).sum()/weights.sum()
        targets = codes[:,model.audio_offset:model.audio_offset+model.dep_q]
        audio_losses = []
        for q in range(model.dep_q):
            mask = result.mask[:,q]
            audio_losses.append(F.cross_entropy(result.logits[:,q][mask].float(),targets[:,q][mask]))
        # Native first-codebook emphasis, normalized by the total weight.
        speech_loss = (100*audio_losses[0]+sum(audio_losses[1:]))/(100+model.dep_q-1)
        word_mask = text_target >= 5
        rag_mask = text_target == 4
        predicted = text_logits.argmax(-1)
        metrics = {"id":row["scenario_id"],"text":float(text_loss.detach()),"speech":float(speech_loss.detach()),
            "word_nll":float(nll[word_mask].mean().detach()) if word_mask.any() else None,
            "rag_nll":float(nll[rag_mask].mean().detach()) if rag_mask.any() else None,
            "result_word_nll":float(nll[result_words].mean().detach()) if result_words.any() else None,
            "rag_target_count":int(rag_mask.sum()),"rag_predicted_count":int((predicted==4).sum()),
            "rag_true_positive":int(((predicted==4)&rag_mask).sum())}
        return text_loss+speech_loss,metrics
    def evaluate(label):
        model.eval()
        with torch.inference_mode():
            metrics = [loss_and_metrics(row)[1] for row in validation]
        (args.out / f"validation-{label}.json").write_text(json.dumps(metrics,indent=2)+"\n")
        model.train()
        return metrics

    fixed_row = next(r for r in train if r["reference_rows"] > 0)
    fixed_codes,fixed_reference = sample(fixed_row)
    # A short parity probe includes the actual result arrival, not just silence.
    active = (fixed_reference[0].abs().sum(-1)>0).nonzero().flatten()
    cut = min(fixed_codes.shape[-1],int(active[-1])+32)
    fixed_codes,fixed_reference = fixed_codes[...,:cut],fixed_reference[:,:cut]
    baseline = precise_sample(model,fixed_codes,fixed_reference)
    torch.manual_seed(424242)
    random.seed(424242)
    model = get_lora_moshi(model,None,lora_rank=8,lora_scaling=2.,device="cuda",fuse_lora=False)
    model.requires_grad_(False)
    adapters = adapter_parameters(model)
    with torch.no_grad():
        for name,parameter in adapters.items():
            if ".lora_B." in name:
                parameter.zero_()
            parameter.requires_grad_(not name.startswith("condition_provider."))
    zero = precise_sample(model,fixed_codes,fixed_reference)
    zero_error = {name:float(torch.nan_to_num((baseline[name]-zero[name]).abs()).max()) for name in zero}
    if any(v > .001 for v in zero_error.values()):
        raise RuntimeError("Adapter initialization changed the pretrained model")
    del baseline,zero
    trainable = {n:p for n,p in adapters.items() if p.requires_grad}
    frozen = tensor_hashes((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    initial = tensor_hashes(adapters.items())
    prepare_mixed_precision(trainable.values(),param_dtype=torch.bfloat16,optim_dtype=torch.float32)
    optimizer = torch.optim.AdamW(list(trainable.values()),lr=1e-5,weight_decay=.01)
    model.transformer.checkpointing = model.depformer.checkpointing = True
    report = {"mode":"fresh verified conditioned LoRA pilot; teacher-forced validation is diagnostic",
        "commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "model":manifest["rag"],"cache":str(args.cache),"cache_sha256":sha256(args.cache / "cache.json"),
        "rank":8,"scaling":2.,"trainable_count":sum(p.numel() for p in trainable.values()),
        "rag_weight":args.rag_weight,"result_word_weight":args.result_weight,
        "zero_adapter_logit_errors":zero_error,"validation_before":evaluate("before"),"losses":[],
        "test_fold_used_for_training_or_selection":False,"human_listening":"pending","quality_qualified":False}
    order = []
    for step in range(args.steps):
        if not order:
            order = random.sample(train,len(train))
        row = order.pop()
        optimizer.zero_grad(set_to_none=True)
        # Checkpoint recomputation must use the same compile setting as forward.
        # Ending no_compile before backward changes the saved tensor graph.
        with no_compile():
            loss,metrics = loss_and_metrics(row)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite fresh training loss")
            loss.backward()
        gradients = [p.grad for p in trainable.values() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients) or not any(g.count_nonzero() for g in gradients):
            raise RuntimeError("Invalid adapter gradients")
        upcast_mixed_precision(trainable.values(),optim_dtype=torch.float32)
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        downcast_mixed_precision(trainable.values(),param_dtype=torch.bfloat16)
        metrics.update(step=step+1,gradient_norm=float(norm))
        report["losses"].append(metrics)
        print(json.dumps(metrics),flush=True)
        if (step+1)%8 == 0:
            save_file({n:p.detach().cpu().contiguous() for n,p in adapters.items()},args.out / f"step-{step+1:04d}.safetensors")
    if frozen != tensor_hashes((n,p) for n,p in model.named_parameters() if not p.requires_grad):
        raise RuntimeError("Frozen pretrained/conditioner weights changed")
    report["validation_after"] = evaluate("after")
    report["adapter_keys"] = list(adapters)
    report["adapter_hashes"] = tensor_hashes(adapters.items())
    report["changed_adapter_tensors"] = sum(initial[n] != digest for n,digest in report["adapter_hashes"].items())
    report["frozen_weights_unchanged"] = True
    save_file({n:p.detach().cpu().contiguous() for n,p in adapters.items()},args.out / "lora.safetensors")
    torch.save({"optimizer":optimizer.state_dict(),"master_parameters":{n:p._mp_param.detach().cpu() for n,p in trainable.items()},
        "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),"python_rng":random.getstate(),"steps":args.steps},args.out / "optimizer.pt")
    torch.save({"codes":fixed_codes.cpu(),"reference":fixed_reference.cpu(),**precise_sample(model,fixed_codes,fixed_reference)},
        args.out / "fixed-reference.pt")
    report.update(saved_adapter_sha256=sha256(args.out / "lora.safetensors"),passed=True,optimizer_steps=args.steps,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.out / "report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
