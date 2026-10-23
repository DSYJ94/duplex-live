"""Run the unchanged bounded trainer with audited immutable input hash reuse."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evals.immutable_file_hashes import verified_reads

PROFILE='full-first-read-plus-file-identity-every-reuse-v1'
CODE=('evals/immutable_file_hashes.py','evals/moshi_model_download.py',
      'portable/context/native_training_cached.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_receipt(path,kind,commit):
    r=json.loads(Path(path).read_text());root=Path(__file__).resolve().parents[2]
    expected={name:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=root)).hexdigest() for name in CODE}
    if (r['schema']!='dl-native-immutable-input-hash-reuse-v1' or r['passed'] is not True
            or r['stage_kind']!=kind or r['source_commit']!=commit or r['code_sha256']!=expected
            or r['cache']['mode']!=PROFILE or r['cache']['unique_hashed_files']<1
            or r['cache']['all_cached_identities_rechecked'] is not True
            or r['cache']['validation_results_or_model_outputs_cached'] is not False):
        raise ValueError('Complete matching input-hash verification receipt required')
    return r


def main():
    argv=sys.argv[1:]
    if '--control40-cache' not in argv:raise ValueError('This wrapper is specific to one bounded retained control40 pilot')
    def option(name):return Path(argv[argv.index('--'+name)+1])
    out=option('out');source=Path(__file__).resolve().parents[2]
    commit=os.environ['DL_TASK_COMMIT']
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()):
        raise ValueError('Clean exact committed execution source required')
    roots=[Path('/projects/u6si/duplex-live/runs'),source/'runs']
    if any(out.resolve().is_relative_to(p.resolve()) for p in roots):
        raise ValueError('Generated outputs must be outside immutable input roots')
    kind='reload' if '--reload' in argv else 'readiness' if '--native-audio49-preflight-only' in argv else 'training'
    ready=option('native-audio49-preflight') if kind!='readiness' else None
    if ready is not None:verify_receipt(ready/'immutable-input-readiness.json','readiness',commit)
    if kind=='reload':verify_receipt(out/'immutable-input-training.json','training',commit)
    passed=False;reads=None
    try:
        with verified_reads(roots) as reads:
            from evals.context_training_proof import main as train
            train()
        passed=True
    finally:
        if out.is_dir() and reads is not None:
            r=dict(schema='dl-native-immutable-input-hash-reuse-v1',passed=passed,stage_kind=kind,
                   source_commit=commit,cache=reads.receipt(),
                   code_sha256={name:sha(source/name) for name in CODE},
                   model_loss_dtype_sampler_gradients_or_optimizer_changed=False)
            (out/('immutable-input-'+kind+'.json')).write_text(json.dumps(r,indent=2)+'\n')
    if kind=='training':
        shutil.copyfile(ready/'immutable-input-readiness.json',out/'approved-immutable-input-readiness.json')
        report=out/'report.json';r=json.loads(report.read_text())
        r['immutable_input_hashes']=dict(profile=PROFILE,
            training_receipt_sha256=sha(out/'immutable-input-training.json'),
            readiness_receipt_sha256=sha(out/'approved-immutable-input-readiness.json'))
        report.write_text(json.dumps(r,indent=2)+'\n')
    elif kind=='reload':
        report=out/'reload-report.json';r=json.loads(report.read_text())
        r['immutable_input_hashes_receipt_sha256']=sha(out/'immutable-input-reload.json')
        report.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
