"""Recreate exact legacy input paths after complete CPU destination verification.

No model, training, provider action or original-file rewrite. Published vendor
sources are Git-cloned from verified preserved bundles into separate checkouts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from evals.native_gcp_destination_verify import INVENTORY_SHA256
from evals.native_gcp_transfer_inventory import load, safe
from evals.native_saved_stage_recovery import verified_inputs

RUNTIME = Path('/opt/reasoning32-runtime-root-v12')
ORIGINAL = Path('/opt/duplex-reasoning32-da389a7-v15')
ALIASES = (Path('/projects/u6si/duplex-live'), Path('/lus/lfs1aip2/projects/u6si/duplex-live'))
CONTEXT = '79df302fa2820cae10c61bb2b7f302c52edb6ed2'
RAG = '8c6dfc101b7871baa428424bcdc583b74fb561d9'
FINETUNE = '2acc879fe7c48f885a18f6cc9548bccb2674d87b'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def regular_link(source, target, expected=None):
    if source.is_symlink() or not source.is_file():
        raise ValueError('Only a preserved regular file can be materialized')
    if expected and (source.stat().st_size != expected['bytes'] or digest(source) != expected['sha256']):
        raise ValueError('Pinned materialization input changed')
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (target.parent, *target.parent.parents)):
        raise ValueError('Materialization destination cannot traverse symlinks')
    os.link(source, target, follow_symlinks=False)


def regular_copy_tree(source, target):
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Physical preserved directory required')
    target.mkdir(parents=True, exist_ok=False)
    count = 0
    for item in sorted(source.rglob('*')):
        if item.is_symlink():
            raise ValueError('Temporary verifier inputs cannot contain symlinks')
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError('Only regular temporary verifier inputs allowed')
        destination = target/item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with item.open('rb') as reader, destination.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=1024*1024)
        if digest(destination) != digest(item):
            raise ValueError('Temporary verifier copy changed')
        count += 1
    return count


def git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()


def clone_bundle(bundle, target, commit, sparse=None):
    if target.exists() or target.is_symlink():
        raise ValueError('New separate vendor checkout required')
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', str(bundle), str(target)], check=True)
    if sparse:
        git('-C', target, 'sparse-checkout', 'set', *sparse)
    git('-C', target, 'checkout', '--detach', commit)
    if git('-C', target, 'rev-parse', 'HEAD') != commit or git('-C', target, 'status', '--porcelain'):
        raise ValueError('Exact clean preserved vendor source required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root != Path('/workspace') or os.uname().sysname != 'Linux':
        raise ValueError('This exact GCP restoration requires Linux /workspace')
    report = json.loads(args.destination_report.read_text())
    if (report['inventory_sha256'] != INVENTORY_SHA256 or report['passed'] is not True
            or report['all_catalog_entries_verified'] is not True or report['failures']
            or report['verified_regular_files'] != 40132 or report['verified_bytes'] != 81154452891
            or report['verified_archival_links'] != 8 or report['physical_root'] != str(root)):
        raise ValueError('Actual full destination receipt required before reconstruction')
    if args.out.exists():
        raise ValueError('New CPU reconstruction receipt required')
    inventory = Path(__file__).resolve().parents[2]/'docs/GCP-NATIVE-TRANSFER-INVENTORY-2026-10-24.json.gz'
    catalog, _ = load(inventory, INVENTORY_SHA256)
    archive = root/'native-preservation-v3'
    if git('-C', ORIGINAL, 'rev-parse', 'HEAD') != 'da389a7031a5504495397d1575952194ffa209f1':
        raise ValueError('Original DA checkout required')
    if RUNTIME.is_symlink() or RUNTIME.exists() and any(RUNTIME.iterdir()):
        raise ValueError('New or operator-created empty runtime directory required')
    RUNTIME.mkdir(exist_ok=True)
    mirrored = 0
    for row in catalog['files']:
        prefix = 'native-preservation-v3/'
        if not row['path'].startswith(prefix):
            continue
        relative = safe(row['path'][len(prefix):])
        if Path(relative).parts[0] in {'models', 'runs', 'tmp', 'logs', 'worktrees'}:
            regular_link(root/row['path'], RUNTIME/relative)
            mirrored += 1
    bundles = archive/'runs/local-preservation-vendor-sources-20261019'
    clone_bundle(bundles/('context-spanning-'+CONTEXT+'.bundle'),
                 RUNTIME/('vendor/context-spanning-'+CONTEXT), CONTEXT, ('contextspan', 'docs'))
    clone_bundle(bundles/('moshi-rag-'+RAG+'.bundle'), root/'control40-vendor-moshi-rag', RAG)
    clone_bundle(bundles/('moshi-finetune-'+FINETUNE+'.bundle'), root/'control40-vendor-moshi-finetune', FINETUNE)
    source_runs = ORIGINAL/'runs'
    if source_runs.is_symlink() or source_runs.exists() and any(source_runs.iterdir()):
        raise ValueError('Original source runs directory must be empty and physical')
    source_runs.mkdir(exist_ok=True)
    # Fixed parent directories are created/chowned by the approved SSH operator.
    for alias in ALIASES:
        alias.symlink_to(RUNTIME, target_is_directory=True)
    (root/'duplex-live').symlink_to(RUNTIME, target_is_directory=True)
    # /tmp may be cleared when CPU stops and the same disk boots on the GPU.
    # Keep regular copies durable, then bind this directory to its exact /tmp
    # path inside each execution container (never replace it with a symlink).
    durable_temporary = root/'gcp-verifier-inputs-v1'
    durable_temporary.mkdir(exist_ok=False)
    logical_temporary = Path('/tmp/duplex-reasoning32-verifier-v1')
    temporary = []
    for group in catalog['temporary_materializations']:
        relative = Path(group['destination']).relative_to(logical_temporary)
        target = durable_temporary/safe(str(relative))
        count = regular_copy_tree(root/safe(group['source_prefix']), target)
        temporary.append(dict(container_destination=group['destination'], host_source=str(target), regular_files=count))
    spec = catalog['snapshot_materialization']
    snapshot = root/safe(spec['destination_prefix'])
    original = root/safe(spec['inventory_artifact'])
    if digest(original) != spec['inventory_sha256']:
        raise ValueError('Original closed82 snapshot inventory changed')
    rows = json.loads(original.read_text())['files']
    if len(rows) != 82:
        raise ValueError('Complete closed snapshot required')
    for row in rows:
        name = safe(row['path'])
        regular_link(root/safe(spec['source_prefix'])/name, snapshot/name, row)
    checked = verified_inputs(snapshot, root/safe(spec['original_observation']))
    command = checked['controller']['planned_commands']['bounded8_training']
    # Only check file presence. NEVER execute the old eight-update command.
    missing = [word for word in command if word.startswith('/')
               and word != command[command.index('--out')+1] and not Path(word).exists()]
    if missing:
        raise ValueError('Original command still has missing physical inputs: '+repr(missing))
    receipt = dict(schema='dl-gcp-native-exact-inputs-v1', source_commit=git('rev-parse', 'HEAD'),
        destination_report_sha256=digest(args.destination_report), mirrored_regular_files=mirrored,
        temporary_inputs=temporary, snapshot_regular_files=82, original_saved8_metadata_verified=True,
        original_command_inputs_present=True, new_optimizer_updates=0,
        historical_code_source_only=True, model_loaded=False, native_behavior_evaluated=False,
        runtime_root=str(RUNTIME), original_da=str(ORIGINAL))
    receipt['required_temporary_bind_mount'] = dict(source=str(durable_temporary),
        destination=str(logical_temporary), readonly=True, symlink_root_allowed=False)
    with args.out.open('x') as writer:
        json.dump(receipt, writer, indent=2); writer.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
