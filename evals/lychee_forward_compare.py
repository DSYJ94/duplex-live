"""Compare released native vLLM prefills with the released HF training graph.

Shared exact input embeddings isolate the transformer/branch/head graph. This
does not validate audio feature construction, incremental KV reuse or training.
Raw errors and ranking changes are retained; no tolerance is fitted to results.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from evals.lychee_model_download import SOURCE_COMMIT
from evals.moshi_model_download import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','native-run','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    import torch
    from evals.lychee_environment_check import configure_bf16
    configure_bf16()
    assets=json.loads(args.assets.read_text());source=Path(assets['source'])
    if (assets['source_commit']!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Native source changed')
    model_path=Path(assets['models']['duplex']['path'])
    for name,record in assets['models']['duplex']['files'].items():
        if sha256(model_path/name)!=record['sha256']:raise RuntimeError('Pinned checkpoint changed')
    report=json.loads((args.native_run/'report.json').read_text())
    captures=[Path(p) for p in report['forward_captures']]
    if len(captures)!=3 or not report['strict_loading']['model']['all_parameter_shards_loaded']:
        raise RuntimeError('Three complete native prefills with strict checkpoint loading are required')
    args.out.mkdir(exist_ok=False)
    sys.path.insert(0,str(source))
    from Training_package.Code.Models.LycheeFD import LycheeFD
    model,loading=LycheeFD.from_pretrained(model_path,torch_dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('HF training graph did not load the entire checkpoint: '+repr(loading))
    model.eval();rows=[]
    with torch.inference_mode():
        for path in captures:
            saved=torch.load(path,map_location='cpu',weights_only=True)
            inputs=saved['embeddings'].to(device='cuda',dtype=torch.bfloat16).unsqueeze(0)
            ids=saved['input_ids'].to('cuda')
            if inputs.shape[1]!=ids.shape[1] or not torch.equal(saved['positions'],torch.arange(ids.shape[1])):
                raise RuntimeError('Capture is not an exact full prefill')
            output=model(input_ids=ids,inputs_embeds=inputs,apply_internal_merge=False,
                use_cache=False,return_dict=True)
            # Native merge consumes the next known text token at each historical
            # position and the actual newly sampled text at the final position.
            shifted=torch.cat([ids[:,1:],torch.tensor([[saved['next_text']]],device='cuda')],dim=1)
            merge=model.merge_model(inputs_embeds=output.stoken_hidden_states+model.get_input_embeddings()(shifted),
                use_cache=False,return_dict=True)
            logits={'text':output.logits[0,-1].float().cpu(),
                'control':output.control_logits[0,-1].float().cpu(),
                'stoken':model.lm_head(merge.last_hidden_state[:,-1:])[0,0].float().cpu()}
            for head,actual in logits.items():
                expected=saved[head+'_logits']
                if actual.shape!=expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                    raise RuntimeError('Invalid head logits in native/training comparison')
                diff=actual-expected;p=expected.log_softmax(-1);q=actual.log_softmax(-1)
                top_native=torch.topk(expected,20).indices.tolist();top_hf=torch.topk(actual,20).indices.tolist()
                rows.append({'capture':path.name,'capture_sha256':sha256(path),'positions':ids.shape[1],
                    'speech_in_history':saved['speech_in_history'],'head':head,
                    'max_abs_logit_error':float(diff.abs().max()),'rms_logit_error':float(diff.square().mean().sqrt()),
                    'kl_native_to_hf':float((p.exp()*(p-q)).sum()),
                    'top1_equal':top_native[0]==top_hf[0],
                    'top20_overlap':len(set(top_native)&set(top_hf)),
                    'native_top20':top_native,'hf_top20':top_hf})
            del output,merge
    result={'schema':'dl-lychee-native-hf-graph-comparison-v1','source_commit':SOURCE_COMMIT,
        'native_report_sha256':sha256(args.native_run/'report.json'),'hf_loading':loading,
        'shared_exact_input_embeddings':True,'audio_construction_compared':False,
        'incremental_kv_compared':False,'microphone_acceptance':False,'training_approved':False,
        'all_raw_head_top1_equal':all(r['top1_equal'] for r in rows),
        'quality_qualified':False,'comparisons':rows}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
