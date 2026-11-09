"""Read-only aliases keep original artifact receipts byte-identical."""
from pathlib import Path

BASE=Path('/projects/u6ada/duplex-live')


def legacy_mounts():
    old='/projects/u6si/duplex-live'
    return [(str(BASE),str(BASE),''),(str(BASE.resolve()),str(BASE.resolve()),''),
        (str(BASE/'legacy-native-v1'),old,'ro'),
        (str(BASE/'legacy-native-v1/migration'),'/workspace/migration','ro'),
        (str(BASE/'vendor/context-spanning-79df-v1'),old+'/vendor/context-spanning-79df302fa2820cae10c61bb2b7f302c52edb6ed2','ro'),
        (str(BASE/'public-models/context-spanning-e62a-v1'),old+'/models/hf-cache/hub/models--mindlogicinc--context-spanning-7b/snapshots/e62a4494a295ff8cee345451cf40aae85719a9f5','ro'),
        (str(BASE/'public-models/personaplex-fdaf-v1'),old+'/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309','ro'),
        (str(BASE/'vendor/moshi-rag-8c6df-v1'),'/workspace/control40-vendor-moshi-rag','ro'),
        (str(BASE/'vendor/moshi-finetune-2acc-v1'),'/workspace/control40-vendor-moshi-finetune','ro')]


def command(work,*,gpu=None,environment=None):
    # Login CPU namespace needs the shared system /dev. Compute nodes use the
    # established contained GPU profile. No custom /dev or global host changes.
    result=['apptainer','exec','--cleanenv']
    result+=['--no-home','--no-mount','cwd'] if gpu is None else ['--containall','--nv']
    for source,target,mode in legacy_mounts():
        result+=['--bind',source+':'+target+(':'+mode if mode else '')]
    # The cache's original snapshot verifier intentionally uses cwd/runs.
    # These aliases preserve its exact bytes/paths and all7797 source checks.
    snapshot=BASE/'native-control40-workset-v1'
    result+=['--bind',str(snapshot)+':/workspace/control40-stage-e658c20:ro',
        '--bind',str(snapshot/'runs')+':'+str(Path(work)/'runs')+':ro']
    legacy=BASE/'source-original-input-verifier-v1'
    result+=['--bind',str(legacy)+':'+str(legacy)+':ro',
        '--bind',str(snapshot/'runs')+':'+str(legacy/'runs')+':ro']
    result+=['--bind',str(BASE/'native-reasoning32-verifier-inputs-v1')+':/tmp/duplex-reasoning32-verifier-v1:ro',
        '--bind',str(BASE/'native-reasoning32-timing-v11-semantic-v1')+':/workspace/migration/native-reasoning32-timing-v11-semantic-v1:ro']
    result+=['--pwd',str(work)]
    env=dict(CUDA_VISIBLE_DEVICES='-1' if gpu is None else str(gpu),
        PYTHONPATH=str(BASE/'artifacts/original-verifier-python-v1')+':'+str(BASE/'artifacts/personaplex-runtime-v1/pylibs')+':'+str(work),
        HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    if environment:
        if 'CUDA_VISIBLE_DEVICES' in environment:raise ValueError('Explicit selected GPU ownership cannot be overridden')
        env.update(environment)
    for name,value in env.items():result+=['--env',name+'='+str(value)]
    result+=[str(BASE/'artifacts/native-v75/runtime/pytorch-new.sif'),
        str(BASE/'envs/venus-arm64-v1/bin/python'),'-B','-u','-m']
    return result
