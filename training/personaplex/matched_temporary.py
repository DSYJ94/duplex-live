"""Restore five exact regular verifier trees at their original temporary alias.

Reuse the already reviewed byte-copy and independent verification functions;
no symlinks, altered receipt paths, generated audio, approvals or model calls.
"""
import gzip,hashlib,json,subprocess
import importlib.metadata
from pathlib import Path
from portable.context.native_gcp_inputs import regular_copy_tree,verify_copy_tree

BASE=Path('/projects/u6ada/duplex-live')
ROOT=BASE/'native-reasoning32-verifier-inputs-v1'
CATALOG_SHA='94b42bae0152e8150ba7842146f66711b13df3b8ece9125bcb5db92a761196fb'


def main():
    versions={name:importlib.metadata.version(name) for name in ('openai-whisper','whisper-timestamped')}
    if versions!={'openai-whisper':'20250625','whisper-timestamped':'1.15.9'}:
        raise ValueError('Exact original ASR package versions required; no recognition is executed')
    closure=json.loads(Path('docs/GCP-NATIVE-SEMANTIC-RECEIPT-CLOSURE-2026-10-25.json').read_text())
    semantic=BASE/'native-reasoning32-timing-v11-semantic-v1'
    if len(closure['files'])!=33 or {p.name for p in semantic.iterdir()}!={r['name'] for r in closure['files']}:
        raise ValueError('All original33 final semantic receipts required')
    for row in closure['files']:
        file=semantic/row['name']
        if file.is_symlink() or file.stat().st_size!=row['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Original complete32 semantic receipt changed')
    path=Path('docs/GCP-NATIVE-TRANSFER-INVENTORY-2026-10-24.json.gz')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CATALOG_SHA:raise ValueError('Original complete transfer catalogue required')
    groups=json.loads(gzip.decompress(path.read_bytes()))['temporary_materializations']
    if len(groups)!=5:raise ValueError('All five original verifier input trees required')
    ROOT.mkdir(parents=True,exist_ok=True);rows=[]
    for group in groups:
        relative=Path(group['source_prefix']).relative_to('migration')
        source=BASE/'legacy-native-v1/migration'/relative
        name=Path(group['destination']).relative_to('/tmp/duplex-reasoning32-verifier-v1')
        if len(name.parts)!=1 or not group['regular_copy_required'] or not group['no_symlink_roots']:
            raise ValueError('Only original regular-copy aliases allowed')
        target=ROOT/name
        if not target.exists():regular_copy_tree(source,target)
        count=verify_copy_tree(source,target)
        rows.append(dict(source=str(source),destination=str(target),files=count,all_hashes_exact=True,independent_regular_copies=True))
    report=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        passed=True,all_five=True,groups=rows,GPU_allocated=False,optimizer_updates=0,
        original_metadata_and_paths_unchanged=True,original_semantic33_exact=True,
        original_ASR_package_versions=versions,ASR_executed=False,external_model_API_calls=0)
    (ROOT/'materialization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
