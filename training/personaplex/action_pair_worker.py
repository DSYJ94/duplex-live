"""One fixed paired fit, followed by independent native restore/evaluation.

Never changes the backbone or original cache. Each fitted arm is a new lineage;
no automatic retry, saved-update replay, forced dispatch or product adoption.
"""
import argparse
import gc
import json
import os
import sys
import uuid
from pathlib import Path
import torch
from .expanded_guard import PAIR_AUTH, sha
from .inference_guard import FrameGuard
from .action_inference import identity
from .action_fit import fit, validate, new_runtime, evaluate
from .action_restore import restore_fit
from .native_process import process_identity

BASE = Path('/projects/u6ada/duplex-live')
CACHE = BASE/'native-feature-restore-20261117-v1/cache'
CACHE_BACKUP = Path('/home/u6ada/dsyj94.u6ada/duplex-native-feature-backups-v1/job-8535017/cache')
RESTORED = BASE/'native-restore-only-20261117-v1'


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load_cache(bound, scope):
    terminal = json.loads((RESTORED/'verify/terminal.json').read_text())
    host = json.loads((RESTORED/'job-8537504/host-terminal.json').read_text())
    if (not host['success'] or not host['fresh_native_restore_passed']
            or terminal['source'] != scope['restore_source'] or not terminal['all_features_exact']
            or not terminal['all_Adam_RNG_OPAL_restored_exact'] or terminal['optimizer_updates'] != 0):
        raise ValueError('Actual complete native cache restore required')
    for key in ('cache_backup', 'verify_backup'):
        receipt = host[key]
        destination = Path(receipt['destination'])
        if (not receipt['passed'] or not receipt['every_file_sha256_verified']
                or sha(destination/'archive-inventory.json') != receipt['inventory_sha256']):
            raise ValueError('Independent original cache and restore backups required')
    for root in (CACHE, CACHE_BACKUP):
        if sha(root/'features.pt') != scope['features_sha256']:
            raise ValueError('Original and independent feature bytes changed')
    cache = torch.load(CACHE/'features.pt', map_location='cpu', weights_only=False)
    if (cache['source'] != scope['cache_source'] or cache['identity'] != bound
            or len(cache['rows']) != 197 or cache['population']['positive'] != 89
            or cache['population']['negative'] != 75959 or validate(cache) != 4096):
        raise ValueError('Exact full197 original TRAIN feature population required')
    return cache


def run(a):
    guard = FrameGuard(a.lease)
    lease = guard(900)
    scope = json.loads(PAIR_AUTH.read_text())
    if (lease.get('paired_authorization_sha256') != sha(PAIR_AUTH)
            or lease['source'] != a.source or torch.cuda.device_count() != 1):
        raise ValueError('New separately admitted paired action allocation required')
    if 'paired_recovery_sha256'in lease and a.mode!='bias':
        raise ValueError('Recovery cannot repeat the completed readout arm')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if sha(a.plan) != lease['action_plan_sha256']:
        raise ValueError('Admitted phase plan changed')
    plan = json.loads(a.plan.read_text())
    selection = plan['selection']
    bound = identity(selection)
    if bound['checkpoint_sha256'] != scope['parent_checkpoint_sha256']:
        raise ValueError('Exact preserved196 checkpoint required')
    cache = load_cache(bound, scope)
    folder = a.root/a.mode/a.stage
    folder.mkdir(parents=True, exist_ok=False)
    execution = uuid.uuid4().hex
    provenance = dict(source=a.source, job_id=lease['job_id'], execution=execution,
                      process_identity=process_identity(), features_sha256=scope['features_sha256'])
    if a.stage == 'fit':
        result = fit(cache, folder/'state', mode=a.mode, device='cuda',
                     opal_source=BASE/'opal-upstream-v1', guard=guard,
                     identity=bound, provenance=provenance)
        result.update(checkpoint_sha256=sha(result['checkpoint']), provenance=provenance)
        write(folder/'terminal.json', result)
        return
    receipt = json.loads((a.root/a.mode/'fit/terminal.json').read_text())
    state_path = Path(receipt['checkpoint'])
    if (state_path != a.root/a.mode/'fit/state/step-0128.pt'
            or sha(state_path) != receipt['checkpoint_sha256']):
        raise ValueError('Exact backed final128 fit required')
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    if state['metadata']['mode'] != a.mode:
        raise ValueError('No swapping fitted arms')
    prior = state['metadata']['provenance']
    if (prior != receipt['provenance'] or prior['source'] != a.source
            or prior['job_id'] != lease['job_id'] or prior['features_sha256'] != scope['features_sha256']
            or prior['execution'] == execution or prior['process_identity'] == process_identity()):
        raise ValueError('Pinned original fit and distinct native restore process required')
    from .expanded_capture import PP, VENDOR, WEIGHTS, INSTRUCTIONS, REFERENCE_SHA
    from .foundation_context import preflight
    from .expanded_evaluation import make_engine
    from evals.context_execution import math_attention
    os.environ.update(CS_BASE_DIR=str(PP), CS_WEIGHTS_DIR=str(WEIGHTS), CS_VOICES_DIR=str(WEIGHTS/'voices'))
    sys.path.insert(0, str(VENDOR))
    math_attention()
    if len(preflight(Path.cwd())) != 32:
        raise ValueError('Full native public/runtime preflight required')
    a.capacity = 'expanded105M'
    a.base = 'personaplex'
    a.trainer_source = selection['trainer_source']
    engine = make_engine(a, selection, readout_state=state)
    runtime, recipe = new_runtime(engine.lm.text_linear.delta, a.mode,
                                 opal_source=BASE/'opal-upstream-v1', out=folder/'restore-opal')
    try:
        if recipe != state['metadata']['recipe']:
            raise ValueError('Original paired recipe required')
        restore = restore_fit(state, engine.lm.text_linear.delta, runtime, bound)
        rows = [dict(r, **{k:r[k].cuda() for k in ('features','RET','other_logsumexp','positive')}) for r in cache['rows']]
        if evaluate(engine.lm.text_linear.delta, rows) != state['metrics']:
            raise ValueError('All fitted TRAIN probabilities differ after native restore')
        del rows
    finally:
        runtime.close()
    engine.lm.requires_grad_(False)
    write(folder/'native-restore.json', dict(restore, provenance=provenance, actual_CUDA=True,
                                           original_fit_sha256=receipt['checkpoint_sha256']))
    from . import matched_evaluation as evaluation
    capture = Path(plan['parent_root'])/'capture/shard-00'
    reference = json.loads((capture/'capture.json').read_text())
    if reference['reference_source_sha256'] != REFERENCE_SHA or reference['instructions'] != INSTRUCTIONS or sha(capture/'voice.pt') != reference['voice_sha256']:
        raise ValueError('Exact original RP voice required')
    voice = torch.load(capture/'voice.pt', map_location='cpu', weights_only=True)['codes']
    original_root = a.root
    cohorts = []
    for cohort, count, key in (('control32',32,'control32_sha256'), ('dev64',64,'dev64_sha256')):
        path = a.inputs/('control-cases.json' if count == 32 else 'dev-cases.json')
        if sha(path) != plan[key]:
            raise ValueError('Exact unchanged complete native evaluation inputs required')
        rows = json.loads(path.read_text())
        if len(rows) != count:
            raise ValueError('Whole native cohort required')
        a.root = folder/cohort
        a.root.mkdir()
        evaluation.collect(a, rows, selection, factory=lambda *_:engine, persona=INSTRUCTIONS,
                           voice_codes=voice, voice_sha256=reference['voice_sha256'],
                           profile_extra=dict(action_mode=a.mode, fitted_updates=128,
                                              action_fit_sha256=receipt['checkpoint_sha256']),
                           guard=lambda _lease, needed=0:guard(needed))
        cohorts.append((a.root, rows, count))
    engine.lm_gen._stop_streaming()
    del engine, cache
    gc.collect()
    torch.cuda.empty_cache()
    for root, rows, count in cohorts:
        a.root = root
        evaluation.audit(a, rows, required=count, require_spoken_content=True,
                         guard=lambda _lease, needed=0:guard(needed))
    a.root = original_root
    write(folder/'terminal.json', dict(passed=True, actual_CUDA=True, native_restore=restore,
                                      mode=a.mode, outputs=96, optimizer_updates=0,
                                      source=a.source, provenance=provenance,
                                      connected_backend_tested=False, native_policy_accepted=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root','lease','plan','inputs'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--mode', choices=('readout','bias'), required=True)
    p.add_argument('stage', choices=('fit','evaluate'))
    run(p.parse_args())
