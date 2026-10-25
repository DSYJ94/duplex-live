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
TOKENIZER_SHA = '78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d'
TOKENIZER_BYTES = 552778
OLD_MODEL = 'models/hf-cache/hub/models--kyutai--moshika-rag-pytorch-bf16/snapshots/7135a6e3c46abb66c2cd95cb04cbfcbe8376f83d'
PERSONA_MODEL = 'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'


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


def vendor_checkouts(root):
    return ((RUNTIME/('vendor/context-spanning-'+CONTEXT), CONTEXT, 'context-spanning'),
            (root/'control40-vendor-moshi-rag', RAG, 'moshi-rag'),
            (root/'control40-vendor-moshi-finetune', FINETUNE, 'moshi-finetune'),
            (RUNTIME/('vendor/moshi-finetune-'+FINETUNE), FINETUNE, 'moshi-finetune'))


def verify_vendors(root):
    for target, commit, _ in vendor_checkouts(root):
        old25 = target == RUNTIME/('vendor/moshi-finetune-'+FINETUNE)
        if old25:
            if not target.is_symlink() or target.readlink() != root/'control40-vendor-moshi-finetune':
                raise ValueError('Exact shared old25 vendor checkout alias required')
        elif target.is_symlink():
            raise ValueError('Physical canonical vendor checkout required')
        if (not target.is_dir()
                or git('-C', target, 'rev-parse', 'HEAD') != commit
                or git('-C', target, 'status', '--porcelain')):
            raise ValueError('Exact clean reconstructed vendor checkout required')


def historical_tokenizer(create=False):
    """Old25 needs the identical shared vocabulary, not a substituted backbone."""
    model = json.loads((RUNTIME/'runs/native-old25-codes-7543730/cache.json').read_text())['model']
    name = 'tokenizer_spm_32k_3.model'
    expected = dict(bytes=TOKENIZER_BYTES, sha256=TOKENIZER_SHA)
    if (model['repo'] != 'kyutai/moshika-rag-pytorch-bf16'
            or model['path'] != '/projects/u6si/duplex-live/'+OLD_MODEL
            or model['files'][name] != expected):
        raise ValueError('Exact original old25 vocabulary reference required')
    source, target = RUNTIME/PERSONA_MODEL/name, RUNTIME/OLD_MODEL/name
    if source.is_symlink() or source.stat().st_size != TOKENIZER_BYTES or digest(source) != TOKENIZER_SHA:
        raise ValueError('Preserved shared vocabulary changed')
    if create:
        regular_link(source, target, expected)
    if (target.is_symlink() or not os.path.samefile(source, target)
            or digest(target) != TOKENIZER_SHA):
        raise ValueError('Exact regular historical vocabulary alias required')
    return dict(path=str(target), bytes=TOKENIZER_BYTES, sha256=TOKENIZER_SHA,
                tokenizer_only=True, full_kyutai_backbone_restored=False)


def control_workset(root, catalog, create=False):
    """Restore the frozen control corpus at the original verifier's relative cwd."""
    prefix = 'source-control40-e658c20/'
    rows = [row for row in catalog['files'] if row['path'].startswith(prefix)]
    carrier = catalog['archive_carriers']['control40']
    if (len(rows) != carrier['expanded_files']
            or sum(row['bytes'] for row in rows) != carrier['expanded_bytes']):
        raise ValueError('Complete catalogued control workset required')
    for row in rows:
        name = safe(row['path'][len(prefix):])
        if Path(name).parts[0] != 'runs' or not Path(name).parts[1].startswith('native-control3-'):
            raise ValueError('Only original control corpus paths may be reconstructed')
        source, target = root/row['path'], ORIGINAL/name
        if create:
            regular_link(source, target, row)
        if (target.is_symlink() or not target.is_file()
                or not os.path.samefile(source, target)
                or target.stat().st_size != row['bytes'] or digest(target) != row['sha256']
                or any(p.is_symlink() for p in target.parents)):
            raise ValueError('Exact regular original-cwd control workset required')
    return dict(files=len(rows), bytes=carrier['expanded_bytes'], root=str(ORIGINAL))


def verify_existing(root, catalog):
    """Read-only recovery after materialization succeeded but its audit failed."""
    mirrored = 0
    for row in catalog['files']:
        prefix = 'native-preservation-v3/'
        if not row['path'].startswith(prefix):
            continue
        relative = safe(row['path'][len(prefix):])
        if Path(relative).parts[0] in {'models', 'runs', 'tmp', 'logs', 'worktrees'}:
            target = RUNTIME/relative
            if (target.is_symlink() or not target.is_file()
                    or not os.path.samefile(target, root/row['path'])
                    or any(p.is_symlink() for p in (target.parent, *target.parent.parents))):
                raise ValueError('Reconstructed regular hardlink changed')
            mirrored += 1
    verify_vendors(root)
    for alias in (*ALIASES, root/'duplex-live'):
        if not alias.is_symlink() or alias.readlink() != RUNTIME:
            raise ValueError('Exact reconstructed alias required')
    source_runs = ORIGINAL/'runs'
    if source_runs.is_symlink() or not source_runs.is_dir():
        raise ValueError('Physical original source runs required')
    control_workset(root, catalog)
    temporary = []
    for group in catalog['temporary_materializations']:
        source = root/safe(group['source_prefix'])
        relative = Path(group['destination']).relative_to('/tmp/duplex-reasoning32-verifier-v1')
        target = root/'gcp-verifier-inputs-v1'/safe(str(relative))
        count = verify_copy_tree(source, target)
        temporary.append(dict(container_destination=group['destination'], host_source=str(target), regular_files=count))
    spec = catalog['snapshot_materialization']
    snapshot = root/safe(spec['destination_prefix'])
    if digest(snapshot/'inventory.json') != spec['inventory_sha256']:
        raise ValueError('Reconstructed snapshot inventory changed')
    rows = json.loads((snapshot/'inventory.json').read_text())['files']
    if len(rows) != 82:
        raise ValueError('Complete closed snapshot required')
    for row in rows:
        target = snapshot/safe(row['path'])
        if target.is_symlink() or not os.path.samefile(target, root/safe(spec['source_prefix'])/row['path']):
            raise ValueError('Reconstructed snapshot regular hardlink changed')
    return mirrored, temporary, snapshot


def verify_copy_tree(source, target):
    for tree in (source, target):
        if tree.is_symlink() or not tree.is_dir() or any(p.is_symlink() for p in tree.rglob('*')):
            raise ValueError('Physical regular verifier trees required')
    files = {p.relative_to(source) for p in source.rglob('*') if p.is_file()}
    if files != {p.relative_to(target) for p in target.rglob('*') if p.is_file()}:
        raise ValueError('Complete verifier copy required')
    for relative in files:
        if os.path.samefile(source/relative, target/relative) or digest(source/relative) != digest(target/relative):
            raise ValueError('Independent exact verifier copy required')
    return len(files)


def finish(args, root, catalog, mirrored, temporary, snapshot):
    tokenizer = historical_tokenizer()
    control = control_workset(root, catalog)
    spec = catalog['snapshot_materialization']
    checked = verified_inputs(snapshot, root/safe(spec['original_observation']))
    command = checked['controller']['planned_commands']['bounded8_training']
    missing = [word for word in command if word.startswith('/')
               and word != command[command.index('--out')+1] and not Path(word).exists()]
    if missing:
        raise ValueError('Original command still has missing physical inputs: '+repr(missing))
    receipt = dict(schema='dl-gcp-native-exact-inputs-v1', source_commit=git('rev-parse', 'HEAD'),
        destination_report_sha256=digest(args.destination_report), mirrored_regular_files=mirrored,
        temporary_inputs=temporary, snapshot_regular_files=82, original_saved8_metadata_verified=True,
        original_command_inputs_present=True, new_optimizer_updates=0,
        historical_code_source_only=True, model_loaded=False, native_behavior_evaluated=False,
        runtime_root=str(RUNTIME), original_da=str(ORIGINAL),
        existing_materialization_independently_reverified=args.verify_existing,
        historical_tokenizer=tokenizer, original_cwd_control_workset=control,
        required_pythonpath=str(root/'control40-vendor-moshi-rag/moshi'))
    receipt['required_temporary_bind_mount'] = dict(source=str(root/'gcp-verifier-inputs-v1'),
        destination='/tmp/duplex-reasoning32-verifier-v1', readonly=True, symlink_root_allowed=False)
    with args.out.open('x') as writer:
        json.dump(receipt, writer, indent=2); writer.write('\n')
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true',
                        help='Reverify every existing materialization and finish its audit; never overwrite or recreate')
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
    if args.verify_existing:
        finish(args, root, catalog, *verify_existing(root, catalog))
        return
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
    for target, commit, prefix in vendor_checkouts(root):
        if target == RUNTIME/('vendor/moshi-finetune-'+FINETUNE):
            target.symlink_to(root/'control40-vendor-moshi-finetune', target_is_directory=True)
        else:
            clone_bundle(bundles/(prefix+'-'+commit+'.bundle'), target, commit,
                         ('contextspan', 'docs') if prefix == 'context-spanning' else None)
    verify_vendors(root)
    historical_tokenizer(create=True)
    source_runs = ORIGINAL/'runs'
    if source_runs.is_symlink() or source_runs.exists() and any(source_runs.iterdir()):
        raise ValueError('Original source runs directory must be empty and physical')
    source_runs.mkdir(exist_ok=True)
    control_workset(root, catalog, create=True)
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
    finish(args, root, catalog, mirrored, temporary, snapshot)


if __name__ == '__main__':
    main()
