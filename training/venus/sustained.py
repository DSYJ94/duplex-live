"""Reuse immutable native proofs; never represent them as new measurements."""
import json
from pathlib import Path
import shutil
from .large_inputs import sha

AUTH=Path('docs/NATIVE-VENUS-SUSTAINED-TRAINING-AUTHORIZATION-2026-11-05.json')
ORIGINAL=Path('/workspace/migration/venus-large-capacity-v65/native-v73')


def authority():
    from .large_guard import AGGREGATE_AUTH,aggregate_authority
    aggregate_authority();a=json.loads(AUTH.read_text())
    if (a['status']!='APPROVED' or sha(a['proposal_path'])!=a['proposal_sha256']
            or a['prior_authorization_sha256']!=sha(AGGREGATE_AUTH)
            or (a['maximum_phase_USD'],a['maximum_project_USD'],a['maximum_aggregate_GPU_seconds'],a['prior_GPU_upper_seconds'],a['additional_GPU_seconds_maximum'],a['shutdown_reserve_seconds'],a['maximum_GPU_intervals'],a['storage_deadline_unix'],a['total_new_updates'])!=(175,350,73600,8700,64800,100,12,1794052800.,120)
            or a['vm_id']!='7521667153519853425' or a['disk_id']!='2297341858915206463'):
        raise ValueError('Exact approved sustained continuation required')
    return a


def validate(root,source):
    a=authority();root=Path(root);m=json.loads((root/'reuse-manifest.json').read_text())
    if (m['controller_source']!=source or m['original_source']!=a['original_source']
            or m['authorization_sha256']!=sha(AUTH) or m['original_root']!=str(ORIGINAL)):
        raise ValueError('Separate original proof and current execution provenance required')
    for n,h in a['proof_files'].items():
        if sha(root/'original-proofs'/n)!=h:raise ValueError('Original proof bytes changed')
    for n,h in a['math_files'].items():
        if sha(Path('training/venus')/n)!=h:raise ValueError('Native math changed; proof cannot be reused')
    for n in ('admission.json','control-runtime.json','workload-geometry.json'):
        if sha(root/n)!=a['proof_files'][n]:raise ValueError('Original evidence cannot be relabelled')
    old=json.loads((root/'admission.json').read_text())
    fresh=json.loads((root/'original-proofs/fresh-process-restore.json').read_text())
    if (old['source']!=a['original_source'] or old['checkpoint_sha256']!=a['init_sha256']
            or not old['actual_CUDA'] or not old['full304_hashes_and_masks_verified']
            or not old['exact_initial_native_logits'] or old['new_optimizer_updates']!=0
            or not old['measured']['all_Adam_moments_materialized']
            or old['measured']['trainable_parameters']!=698351616
            or not fresh['passed'] or not fresh['rng_exact'] or not fresh['all_Adam_and_OPAL_state_exact']
            or not fresh['native_logits_exact'] or fresh['checkpoint_sha256']!=a['init_sha256']):
        raise ValueError('Complete actual original native admission/restore required')
    return m


def stage(output,source):
    a=authority();out=Path(output);out.mkdir(exist_ok=False)
    for n,h in a['proof_files'].items():
        if sha(ORIGINAL/n)!=h:raise ValueError('Original native proof hash differs')
        target=out/'original-proofs'/n;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ORIGINAL/n,target)
    for n in ('admission.json','control-runtime.json','workload-geometry.json'):
        shutil.copyfile(ORIGINAL/n,out/n)
    if sha(ORIGINAL/'expanded-init.pt')!=a['init_sha256']:raise ValueError('Original checkpoint differs')
    (out/'expanded-init.pt').symlink_to(ORIGINAL/'expanded-init.pt')
    m=dict(schema='native-proof-reuse-v1',controller_source=source,original_source=a['original_source'],original_root=str(ORIGINAL),authorization_sha256=sha(AUTH),new_optimizer_updates=0,original_full64_baseline_reused=True)
    (out/'reuse-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    validate(out,source)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);a=p.parse_args();stage(a.output,a.source)
