"""Strict released custom-vLLM Lychee incremental token benchmark.

No vocoder is included: token speed is not streamed PCM or microphone acceptance.
"""
import argparse
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.lychee_model_download import SOURCE_COMMIT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('assets','batch','runtime','preflight','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--warm-input',action='store_true')
    ap.add_argument('--worker-context',action='store_true')
    ap.add_argument('--static-context-diagnostic',action='store_true')
    ap.add_argument('--result-response',choices=('model','application-event-diagnostic'),default='model')
    ap.add_argument('--result-delivery',choices=('immediate','idle-quiet-diagnostic'),default='immediate')
    ap.add_argument('--data',type=Path)
    ap.add_argument('--scenario')
    ap.add_argument('--private-value')
    args=ap.parse_args()
    if args.worker_context and (not args.data or not args.scenario):
        ap.error('Actual worker context requires approved development data/scenario')
    if args.result_response!='model' and (not args.worker_context or args.static_context_diagnostic):
        ap.error('Application response event requires actual late-worker diagnostic, never static context')
    if args.result_delivery!='immediate' and (not args.worker_context or args.static_context_diagnostic):
        ap.error('Delayed delivery is an actual-worker application scheduling diagnostic only')
    if args.static_context_diagnostic and not args.worker_context:
        ap.error('Static reference is a disclosed private-context diagnostic only')
    if args.private_value and not args.worker_context:
        ap.error('Counterfactual private value is only an actual-worker diagnostic world')
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from evals.lychee_environment_check import configure_bf16
    configure_bf16()
    from data.synthetic.quality import checked_rows
    from evals.lychee_vllm_loading import strict_loading
    import vllm
    if not Path(vllm.__file__).resolve().is_relative_to(args.runtime.resolve()):
        raise RuntimeError('A different vLLM package was imported')
    kernels=json.loads((args.preflight/'kernel-report.json').read_text())
    imported=json.loads((args.preflight/'python-import.json').read_text())
    if (not kernels['passed'] or not imported['passed'] or sha256(args.runtime/'vllm/_C.abi3.so')!=kernels['library_sha256']):
        raise RuntimeError('Actual native kernel/import proof is missing or changed')
    if not args.worker_context:
        checked_rows(args.batch,[{'scenario_id':'dense_dialog-001'}],role='Lychee released backend benchmark')
    assets=json.loads(args.assets.read_text());source=Path(assets['source'])
    if (assets['source_commit']!=SOURCE_COMMIT or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Accepted Lychee source changed')
    model=assets['models']['duplex'];model_path=Path(model['path'])
    for name,record in model['files'].items():
        if sha256(model_path/name)!=record['sha256']:
            raise RuntimeError('Pinned Lychee tensor/config file changed: '+name)
    sys.path[:0]=[str(source),str(source/'third_party/Step-Audio2')]
    # The released runtime imports an absent alias; select the exact config
    # implementation whose full checkpoint names/shapes were already verified.
    from Training_package.Code.Models.LycheeFD import LycheeFDConfig, LycheeFD
    alias=types.ModuleType('lychee_fd.models.lychee_duplex')
    alias.LycheeDuplexConfig=LycheeFDConfig
    alias.LycheeDuplexForCausalLM=LycheeFD  # No HF decoder is instantiated by this benchmark.
    sys.modules[alias.__name__]=alias
    from lychee_fd.runtime.vllm_generation import VLLMGenerationFramework
    args.out.mkdir(exist_ok=False)
    environment={'schema':'dl-lychee-vllm-probe-v1','source_commit':SOURCE_COMMIT,
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'torch':torch.__version__,'vllm':vllm.__version__,'vllm_file':vllm.__file__,
        'kernel_proof_sha256':sha256(args.preflight/'kernel-report.json'),
        'assets_manifest_sha256':sha256(args.assets),'native_kernel_library_sha256':kernels['library_sha256'],
        'enforce_eager':True,'align_audio_input':True,'live_audio_or_delegation_qualified':False,
        'released_control_chunk_slots':10,'control_slot_seconds':.04,'input_window_seconds':.4,
        'native_training_first_subword_offset_bar_ms':200,'response_start_latency_tested':False}
    (args.out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    torch.manual_seed(424242)
    with strict_loading(model_path) as loading:
        framework=VLLMGenerationFramework('FD',str(model_path),device='cuda',torch_dtype=torch.bfloat16,
            align_audio_input=True,gpu_memory_utilization=.65,max_model_len=8192,enforce_eager=True)
    (args.out/'loading.json').write_text(json.dumps(loading,indent=2)+'\n')
    if loading['model'] is None or loading['audio'] is None:
        raise RuntimeError('Released backend did not run strict full model/audio loading in this process')
    if args.warm_input:
        from evals.lychee_warmup import warm_input
        environment['input_warmup']=warm_input(framework)
        (args.out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    if args.worker_context:
        from evals.lychee_worker import run
        from evals.lychee_vllm_loading import audited_side_tokens
        with audited_side_tokens(framework.vllm_engine) as audit:
            run(framework,args,loading,audit)
        return
    pcm,rate=sf.read(args.batch/'dense_dialog-001/stem_user.wav',dtype='float32')
    if pcm.ndim!=1 or len(pcm)<60*rate or not np.isfinite(pcm).all():
        raise RuntimeError('Invalid approved user input')
    divisor=math.gcd(rate,16000)
    audio=resample_poly(pcm[:60*rate],16000//divisor,rate//divisor).astype(np.float32)
    audio=np.pad(audio,(0,10*16000));sf.write(args.out/'user_in.wav',audio,16000)
    from evals.lychee_paced import run_incremental
    from evals.lychee_vllm_loading import audited_side_tokens
    with torch.inference_mode(),audited_side_tokens(framework.vllm_engine) as side_audit:
        generated,timing=run_incremental(framework,audio,args.out)
    (args.out/'native-trace.json').write_text(json.dumps(generated['trace'])+'\n')
    (args.out/'events.json').write_text(json.dumps(generated['events'],indent=2)+'\n')
    report={'text':' '.join(e['text'] for e in generated['events']),'paced_tokens':timing,
        'strict_loading':loading,'native_side_token_audit':side_audit,
        'input_warmup':environment.get('input_warmup'),'live_pcm_tested':False,'microphone_acceptance':False,
        'quality_qualified':False,'learned_delegation_tested':False}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
