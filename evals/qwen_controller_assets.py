"""Pinned text-only controller assets and CPU imports, not a duplex model pass.

A small pretrained language model may supply local ordinary knowledge and tool
selection; its intelligence, timing and interaction must be measured separately.
No adaptation, inference, package installation or shared environment changes here.
"""
import argparse
import hashlib
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dl',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    pinfile=Path(__file__).with_name('assets')/'qwen4b-controller-pins.json';pins=json.loads(pinfile.read_text())
    if pins['repo']!='Qwen/Qwen3-4B-Instruct-2507' or pins['license']!='apache-2.0' or pins['gated']:
        raise RuntimeError('Unexpected text-controller model scope')
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import torch,transformers,tokenizers,huggingface_hub
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,Qwen3ForCausalLM
    from transformers.modeling_utils import no_init_weights
    assets=[];folder=None;inventory={}
    for row in pins['files']:
        file=Path(hf_hub_download(pins['repo'],row['file'],revision=pins['revision'],cache_dir=str(args.dl/'models/hf-cache/hub')))
        if file.stat().st_size!=row['bytes']:raise RuntimeError('Changed controller asset size')
        digest=sha256(file)
        if row['lfs_sha256']:
            if digest!=row['lfs_sha256']:raise RuntimeError('Changed public controller LFS object')
        else:
            data=file.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\x00'+data).hexdigest()
            if blob!=row['git_blob']:raise RuntimeError('Changed controller Git object')
        if folder is None:folder=file.parent
        if file.parent!=folder:raise RuntimeError('Controller files span different HF snapshots')
        if file.suffix=='.safetensors':
            with safe_open(file,framework='pt',device='cpu') as tensors:
                for name in tensors.keys():
                    if name in inventory:raise RuntimeError('Duplicated controller tensor')
                    value=tensors.get_slice(name)
                    inventory[name]=dict(file=row['file'],shape=value.get_shape(),dtype=value.get_dtype())
        assets.append(dict(file=row['file'],path=str(file),bytes=row['bytes'],sha256=digest))
    index=json.loads((folder/'model.safetensors.index.json').read_text())
    if set(index['weight_map'])!=set(inventory) or any(inventory[k]['file']!=v for k,v in index['weight_map'].items()):
        raise RuntimeError('Controller shard inventory does not match its index')
    config=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    if config.model_type!='qwen3' or config.num_hidden_layers!=36 or config.hidden_size!=2560:
        raise RuntimeError('Unexpected local text-model architecture')
    prompt=tokenizer.apply_chat_template([dict(role='user',content='State how many sides a triangle has.')],tokenize=False,add_generation_prompt=True)
    encoded=tokenizer(prompt)['input_ids']
    if not encoded or max(encoded)>=config.vocab_size:raise RuntimeError('Controller template/tokenizer mismatch')
    report=dict(schema='dl-qwen4b-controller-assets-v1',passed=True,repo=pins['repo'],revision=pins['revision'],
        license=pins['license'],folder=str(folder),assets=assets,tensors=inventory,tensor_count=len(inventory),
        pins_sha256=sha256(pinfile),code_sha256=sha256(Path(__file__)),
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,huggingface_hub=huggingface_hub.__version__),
        actual_cpu_model_class_import=Qwen3ForCausalLM.__name__,no_init_weights_imported=True,template_tokens=len(encoded),
        model_inference_executed=False,gpu_used=False,training_approved=False,duplex_system_qualified=False,native_speech_model_qualified=False)
    (args.out/'assets.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
