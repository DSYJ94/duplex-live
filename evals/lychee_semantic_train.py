"""Bounded native semantic adapter proof, not a model-quality qualification."""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.lychee_model_download import SOURCE_COMMIT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','data','codec-data','codec-audit','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--source-data',type=Path,nargs='+',required=True)
    ap.add_argument('--history-roots',type=Path,nargs='+',required=True)
    ap.add_argument('--steps',type=int,default=8)
    ap.add_argument('--stage',choices=('proof','pilot'),default='proof')
    ap.add_argument('--reference-proof',type=Path)
    ap.add_argument('--attention-backend',choices=('default','no-cudnn','math'),default='no-cudnn')
    ap.add_argument('--compare-attention-backends',action='store_true')
    args=ap.parse_args()
    reference=None
    if args.stage=='proof':
        if not 1<=args.steps<=8:raise ValueError('Implementation proof permits at most eight updates')
    else:
        if args.steps not in (64,128) or not args.reference_proof or args.attention_backend=='default':
            raise ValueError('Bounded pilot requires 64/128 updates, a passing implementation proof and a finite attention backend')
        reference=json.loads(args.reference_proof.read_text())
        if (not reference.get('passed') or reference['frozen_before_sha256']!=reference['frozen_after_sha256']
                or any(reference['zero_adapter_errors'].values()) or any(reference['post_update_raw_branch_errors'].values())
                or any(reference['fresh_adapter_reload_errors'].values())):
            raise RuntimeError('Bounded pilot requires a passing native implementation proof')
    import torch
    def configure_attention(mode):
        torch.backends.cuda.enable_cudnn_sdp(mode=='default')
        torch.backends.cuda.enable_flash_sdp(mode!='math')
        torch.backends.cuda.enable_mem_efficient_sdp(mode!='math')
        torch.backends.cuda.enable_math_sdp(True)
    configure_attention(args.attention_backend)
    from torch.nn import functional as F
    from evals.lychee_environment_check import configure_bf16
    configure_bf16()
    from evals.lychee_semantic_adapter import attach,detach,state_dict,load_state,frozen_hash
    from evals.recovery_data import validate_conversion
    data=json.loads((args.data/'semantic-data.json').read_text())
    if data['schema']!='dl-lychee-native-semantic-data-v1' or data['training_approved'] is not True:
        raise RuntimeError('Causal native semantic data approval required')
    sources=data.get('source_ingestions',[{'path':str(args.source_data[0]),'sha256':data['source_ingestion_sha256']}])
    if {r['path'] for r in sources}!={str(p) for p in args.source_data}:
        raise RuntimeError('Semantic training needs every approved source conversion')
    for source in sources:
        path=Path(source['path']);validate_conversion(path)
        if sha256(path/'ingestion.json')!=source['sha256']:raise RuntimeError('Source approval changed')
    if (sha256(args.codec_data/'manifest.json')!=data['codec_manifest_sha256']
            or sha256(args.codec_audit/'checks.json')!=data['codec_checks_sha256']):
        raise RuntimeError('Source approvals changed after native target assembly')
    codec_checks=json.loads((args.codec_audit/'checks.json').read_text())
    if sha256(args.codec_audit/'report.json')!=codec_checks['speech_report_sha256']:
        raise RuntimeError('Independent native speech audit changed')
    gold_reports={}
    for row in json.loads((args.codec_data/'manifest.json').read_text())['examples']:
        path=Path(row['path']);report=json.loads((path/'report.json').read_text())
        if (sha256(path/'report.json')!=row['report_sha256']
                or sha256(path/'model.wav')!=report['audio_sha256'] or sha256(path/'codes.pt')!=report['codes_sha256']):
            raise RuntimeError('Native codec source artifact changed')
        gold_reports[row['report_sha256']]=report
    histories={}
    for folder in args.history_roots:
        row=json.loads((folder/'actual-idle-state.json').read_text())
        if row['source_split']!='train' or sha256(folder/row['file'])!=row['sha256']:
            raise RuntimeError('Actual student history provenance changed')
        histories[row['sha256']]=row
    examples=[]
    for row in data['examples']:
        if row['source_split']!='train' or sha256(args.data/row['file'])!=row['sha256']:
            raise RuntimeError('Semantic data fold/hash changed')
        if row['student_history_sha256'] not in histories or row['gold_report_sha256'] not in gold_reports:
            raise RuntimeError('Semantic example lost its actual student/gold source evidence')
        item=torch.load(args.data/row['file'],map_location='cpu',weights_only=True)
        if item['embeddings'].shape[0]!=len(item['input_ids']) or len(item['labels'])!=len(item['input_ids']):
            raise RuntimeError('Native semantic input/label alignment changed')
        targets=(item['labels']!=-100).nonzero().reshape(-1)
        if not len(targets) or int(targets.min())<1 or not torch.equal(item['labels'][targets],item['input_ids'][targets]):
            raise RuntimeError('Invalid shifted future-word targets')
        examples.append(item)
    if not examples:raise RuntimeError('No approved semantic proof examples')
    assets=json.loads(args.assets.read_text());source=Path(assets['source'])
    if (assets['source_commit']!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Pinned training source changed')
    root=Path(assets['models']['duplex']['path'])
    for name,record in assets['models']['duplex']['files'].items():
        if sha256(root/name)!=record['sha256']:raise RuntimeError('Pinned base checkpoint changed')
    from transformers import AutoTokenizer
    from evals.lychee_loss_scope import audit as audit_loss_scope
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=True)
    loss_scope=[audit_loss_scope(item,row,tokenizer) for item,row in zip(examples,data['examples'])]
    print(json.dumps({'loss_ownership_audit':loss_scope,'user_loss_mask_would_change_targets':False}),flush=True)
    args.out.mkdir(exist_ok=False);sys.path.insert(0,str(source))
    from Training_package.Code.Models.LycheeFD import LycheeFD
    torch.manual_seed(424242)
    model,loading=LycheeFD.from_pretrained(root,torch_dtype=torch.bfloat16,device_map={'':0},
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('Native training checkpoint load is incomplete')
    model.eval()
    def inputs(item):
        return item['embeddings'].to(device='cuda',dtype=torch.bfloat16).unsqueeze(0)
    def positions(item):
        return (item['labels']!=-100).nonzero().reshape(-1).to('cuda')
    def heads(item):
        with torch.inference_mode():
            main=model.model(inputs_embeds=inputs(item),use_cache=False,output_hidden_states=True,return_dict=True)
            control=model.control_model(inputs_embeds=main.hidden_states[model.control_model_index],use_cache=False,return_dict=True)
            speech=model.stoken_model(inputs_embeds=main.hidden_states[model.stoken_model_index],use_cache=False,return_dict=True)
            ids=item['input_ids'].to('cuda').unsqueeze(0)
            shifted=model.get_input_embeddings()(ids)
            shifted=torch.cat([shifted[:,1:],torch.zeros_like(shifted[:,:1])],dim=1)
            merge=model.merge_model(inputs_embeds=speech.last_hidden_state+shifted,use_cache=False,return_dict=True)
            at=positions(item)-1
            return {'text':model.lm_head(main.last_hidden_state[:,at]).float().cpu(),
                'speech_hidden':merge.last_hidden_state.cpu(),'control_hidden':control.last_hidden_state.cpu()}
    def loss(item):
        at=positions(item)
        hidden=model.model(inputs_embeds=inputs(item),use_cache=False,return_dict=True).last_hidden_state
        logits=model.lm_head(hidden[:,at-1]).float().reshape(-1,model.config.text_config.vocab_size)
        return F.cross_entropy(logits,item['labels'][at.cpu()].to('cuda'))
    original=heads(examples[0]);cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
    modules=attach(model,8)
    if not torch.equal(cpu,torch.get_rng_state()) or not torch.equal(gpu,torch.cuda.get_rng_state()):
        raise RuntimeError('Adapter construction changed sampling RNG')
    zero=heads(examples[0]);zero_errors={k:float((original[k]-zero[k]).abs().max()) for k in original}
    if any(zero_errors.values()):raise RuntimeError('Zero semantic adapter changed native reference outputs')
    kernel_comparison=[]
    if args.compare_attention_backends:
        for mode in ('default','no-cudnn','math'):
            configure_attention(mode);model.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                measured=loss(examples[0]);measured.backward()
            failures=[]
            for name,module in modules.items():
                for key in ('a','b'):
                    grad=getattr(module,key).grad
                    if grad is None or not torch.isfinite(grad).all():
                        failures.append({'tensor':name+'.'+key,'missing':grad is None,
                            'nan':int(torch.isnan(grad).sum()) if grad is not None else None,
                            'inf':int(torch.isinf(grad).sum()) if grad is not None else None})
            kernel_comparison.append({'backend':mode,'loss':float(measured.detach()),
                'gradient_failures':failures,'optimizer_step_performed':False,
                'actual_attention_ops':[{'name':e.key,'calls':e.count} for e in profile.key_averages()
                    if 'scaled_dot_product' in e.key]})
        model.zero_grad(set_to_none=True);configure_attention(args.attention_backend)
        hardware={'node':os.environ.get('SLURMD_NODENAME',platform.node()),'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        (args.out/'kernel-comparison.json').write_text(json.dumps({'same_model_and_gpu':True,
            'hardware':hardware,'comparisons':kernel_comparison},indent=2)+'\n')
        print(json.dumps({'same_gpu_kernel_comparisons':kernel_comparison,'hardware':hardware}),flush=True)
    frozen_before=frozen_hash(model)
    if reference and frozen_before!=reference['frozen_after_sha256']:
        raise RuntimeError('Pilot base parameters differ from passing implementation proof')
    parameters=[p for module in modules.values() for p in (module.a,module.b)]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
    seen=set();losses=[]
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        value=loss(examples[step%len(examples)])
        if not torch.isfinite(value):raise RuntimeError('Nonfinite native semantic loss')
        value.backward()
        failures=[]
        for name,module in modules.items():
            for key in ('a','b'):
                grad=getattr(module,key).grad
                if grad is None or not torch.isfinite(grad).all():
                    failures.append({'gradient_failure':name+'.'+key,'step':step+1,'finite_loss':float(value.detach()),
                        'gradient_missing':grad is None,'gradient_nan':int(torch.isnan(grad).sum()) if grad is not None else None,
                        'gradient_inf':int(torch.isinf(grad).sum()) if grad is not None else None})
                    continue
                if bool(grad.abs().sum()>0):seen.add(name+'.'+key)
        if failures:
            print(json.dumps({'attention_backend':args.attention_backend,'gradient_failures':failures}),flush=True)
            raise RuntimeError('Missing/nonfinite semantic gradients; see exact tensor report')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('A frozen native parameter received gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not math.isfinite(float(norm)) or float(norm)<=0:raise RuntimeError('Invalid semantic gradient norm')
        optimizer.step();losses.append(float(value.detach()))
        print(json.dumps({'step':step+1,'loss':float(value.detach()),'gradient_norm':float(norm)}),flush=True)
    if args.steps>1 and len(seen)!=len(parameters):raise RuntimeError('Some intended semantic tensors never received nonzero gradients')
    frozen_after=frozen_hash(model)
    if frozen_before!=frozen_after:raise RuntimeError('Frozen pretrained parameters changed')
    trained=heads(examples[0])
    branch_errors={k:float((zero[k]-trained[k]).abs().max()) for k in ('speech_hidden','control_hidden')}
    if any(branch_errors.values()):raise RuntimeError('Post-branch semantic adapters changed raw speech/control outputs')
    text_change=float((zero['text']-trained['text']).abs().max())
    if text_change<=0:raise RuntimeError('Semantic updates had no native text effect')
    checkpoint={'schema':'dl-lychee-semantic-adapter-v1','rank':8,'layers':[24,25,26,27],
        'scale':1.,'state':state_dict(modules),'frozen_sha256':frozen_after,
        'base_assets_manifest_sha256':sha256(args.assets),'semantic_data_sha256':sha256(args.data/'semantic-data.json'),
        'steps':args.steps,'quality_qualified':False}
    checkpoint['training_attention_backend']=args.attention_backend
    checkpoint['training_stage']=args.stage
    checkpoint['reference_implementation_proof_sha256']=sha256(args.reference_proof) if reference else None
    torch.save(checkpoint,args.out/'adapter.pt')
    detach(model,modules);del optimizer,parameters,modules
    fresh=attach(model,8)
    saved=torch.load(args.out/'adapter.pt',map_location='cpu',weights_only=True)
    load_state(fresh,saved['state'])
    reloaded=heads(examples[0]);reload_errors={k:float((trained[k]-reloaded[k]).abs().max()) for k in trained}
    if any(reload_errors.values()):raise RuntimeError('Fresh semantic adapter reload changed native outputs')
    result={'schema':'dl-lychee-semantic-proof-v1','passed':True,'steps':args.steps,'approved_train_examples':len(examples),
        'trainable_tensors':len(seen),'trainable_parameters':sum(p.numel() for module in fresh.values() for p in (module.a,module.b)),
        'zero_adapter_errors':zero_errors,'frozen_before_sha256':frozen_before,'frozen_after_sha256':frozen_after,
        'post_update_raw_branch_errors':branch_errors,'max_text_logit_change':text_change,
        'fresh_adapter_reload_errors':reload_errors,'losses':losses,'checkpoint_sha256':sha256(args.out/'adapter.pt'),
        'data_sha256':sha256(args.data/'semantic-data.json'),'strict_hf_loading':loading,
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'native_merged_deployment_tested':False,'task_success_tested':False,'quality_qualified':False,
        'training_attention_backend':args.attention_backend,
        'training_stage':args.stage,'reference_implementation_proof_sha256':sha256(args.reference_proof) if reference else None,
        'same_gpu_attention_comparisons':kernel_comparison,
        'loss_ownership_audit':loss_scope,'waveform_aec_tested':False,
        'scope':'native semantic implementation proof; raw speech/control compared on one identical teacher-forced input'}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
