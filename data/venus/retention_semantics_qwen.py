"""Independent public32B reviewer; same16 controls and all96 semantic gate."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from .retention_semantics import PROMPT,collect_review,teachers,verdict,calibration,calibration_v2
from .reviewer_model import REPO,REVISION,verify_ready,sha


def final_content(token_ids,tokenizer):
    # Qwen's documented closing-think token. Never store hidden reasoning.
    ids=list(token_ids)
    if 151668 not in ids:return '',dict(closing_think=False,thinking_tokens=None)
    end=len(ids)-ids[::-1].index(151668)
    return tokenizer.decode(ids[end:],skip_special_tokens=True).strip(),dict(closing_think=True,thinking_tokens=end)


def main(a):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from portable.venus.native_app import ensure_guard
    from training.venus.isambard_native import guarded
    ensure_guard();guarded(a.lease,900);ready=verify_ready(a.model,a.download_receipt)
    rows,manifest_sha=teachers(a.timelines,getattr(a,'alignment',None))
    reused={};lineage=None
    if getattr(a,'repair_origin',None):
        from .retention_review_lineage import origin
        rows,reused,lineage=origin(a.repair_origin,a.repair_origin_sha,rows,manifest_sha)
    if a.output.exists():raise ValueError('Preserve original failed reviews')
    a.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    llm=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
    llm.requires_grad_(False);versions={n:p._version for n,p in llm.named_parameters()}
    def judge(row):
        if row['id'] in reused:
            saved=reused[row['id']]
            (a.output/(row['id']+'.json')).write_bytes((a.repair_origin/(row['id']+'.json')).read_bytes())
            return saved
        ensure_guard();guarded(a.lease,180);set_seed(42)
        messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(row['payload'],separators=(',',':')))]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        encoded=tokenizer(prompt,return_tensors='pt').to('cuda')
        if encoded['input_ids'].shape[-1]>6000:raise ValueError('Never truncate teacher context')
        started=time.monotonic()
        with torch.inference_mode():
            out=llm.generate(**encoded,max_new_tokens=2048,do_sample=True,temperature=.6,top_p=.95,top_k=20,
                pad_token_id=tokenizer.eos_token_id)
        ids=out[0,encoded['input_ids'].shape[-1]:].tolist();raw,thinking=final_content(ids,tokenizer)
        result=dict(id=row['id'],payload_sha256=hashlib.sha256(json.dumps(row['payload'],sort_keys=True).encode()).hexdigest(),
            raw_response=raw,parsed=verdict(raw),input_tokens=int(encoded['input_ids'].shape[-1]),
            output_tokens=len(ids),seconds=time.monotonic()-started,**thinking)
        (a.output/(row['id']+'.json')).write_text(json.dumps(result,indent=2)+'\n');return result
    report=collect_review(a,rows,manifest_sha,judge,dict(repo=REPO,revision=REVISION,
        download_receipt_sha256=sha(a.download_receipt),independent_of_native_foundation_and_GPT_author=True,
        thinking=True,seed=42,temperature=.6,top_p=.95,top_k=20,maximum_new_tokens=2048),
        controls_factory=calibration_v2 if a.calibration_version=='v2' else calibration,
        calibration_version=a.calibration_version)
    if any(p._version!=versions[n] for n,p in llm.named_parameters()):raise ValueError('Reviewer weights changed')
    report.update(frozen_model_weights_unchanged=True,public_weights_all_hashes=ready['all_hashes'])
    if getattr(a,'alignment',None):report['measured_alignment_sha256']=sha(a.alignment)
    if lineage:report['review_lineage']=lineage
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all96_semantic_pass']:raise RuntimeError('Complete calibrated review failed; preserve all verdicts')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--alignment',type=Path);p.add_argument('--repair-origin',type=Path);p.add_argument('--repair-origin-sha')
    p.add_argument('--calibration-version',choices=('v1','v2'),default='v1')
    for name in ('model','download-receipt','timelines','output','lease'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
