"""Validate the closed migration catalog; never transfer or authorize anything."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path,PurePosixPath

SCHEMA='dl-native-gcp-closed-transfer-inventory-v1'
LIMIT=100_000_000_000
MAX_JSON_BYTES=40_000_000
FORBIDDEN={'.private','keys','credentials','owner-conversations','session-token',
    '.aws','.ssh','.env','.hf_token','.typesafe_key'}


def safe(name):
    if not isinstance(name,str) or not name or '\\' in name or any(ord(c)<32 or ord(c)==127 for c in name):
        raise ValueError('Canonical relative artifact path required')
    p=PurePosixPath(name)
    if (not p.parts or p.is_absolute() or '..' in p.parts or str(p)!=name or {x.casefold() for x in p.parts}&FORBIDDEN
            or p.name=='core' or any(x.startswith('core-') or x.startswith('core.') and x[5:].isdigit() for x in p.parts)):
        raise ValueError('Excluded or escaping artifact path')
    return name


def unique(pairs):
    value={}
    for key,item in pairs:
        if key in value:raise ValueError('Duplicate catalog object key')
        value[key]=item
    return value


def rows(files):
    seen=set();total=0
    for row in files:
        path=safe(row['path'])
        if path in seen:raise ValueError('Duplicate artifact path')
        seen.add(path)
        if type(row['bytes']) is not int or row['bytes']<0:raise ValueError('Exact nonnegative byte size required')
        value=row['sha256']
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Every artifact requires an explicit SHA256, never an S3 ETag')
        if set(row)-{'path','bytes','sha256','local_workspace_path'}:raise ValueError('Unexpected artifact source or credential field')
        if 'local_workspace_path' in row:safe(row['local_workspace_path'])
        total+=row['bytes']
    if [r['path'] for r in files]!=sorted(seen):raise ValueError('Sorted closed artifact list required')
    if total>LIMIT:raise ValueError('Proposed transfer scope exceeded')
    return total


def validate(value):
    if (value['schema']!=SCHEMA or value['destination_project']!='aurium-duplex-live-20261024'
            or value['source_volume']!='49yytzyoge' or value['status']!='CLOSED_EXPECTED_HASHES_NOT_TRANSFERRED'
            or any(value[k] is not False for k in ('gcp_transfer_authorized','gcp_gpu_started',
                'destination_verified','training_approved','model_adopted'))
            or value['rsync_arguments']!=['-ar','--exclude=session-token','--no-owner','--no-group']):
        raise ValueError('Explicit preparation-only native scope required')
    total=rows(value['files'])
    if total!=value['expanded_artifact_bytes'] or len(value['files'])!=value['regular_files']:
        raise ValueError('Catalog totals changed')
    for prefix,count,size in [('native-preservation-v3/',28470,67764285650),
            ('source-control40-e658c20/',7797,8000842614),
            ('migration/native-control40-precise-pilot-v2/engine/',82,234689568)]:
        group=[r for r in value['files'] if r['path'].startswith(prefix)]
        if len(group)!=count or sum(r['bytes'] for r in group)!=size:
            raise ValueError('Complete preserved component required: '+prefix)
    links=value['archived_symlinks']
    if len(links)!=8 or len({r['path'] for r in links})!=8:raise ValueError('All8 historical archival symlinks required')
    for row in links:
        safe(row['path'])
        if not row['path'].startswith('native-preservation-v3/envs/') or row['target'] not in ('lib','python','/usr/bin/python'):
            raise ValueError('Changed archived runtime link')
    targets={r['path']:r['sha256'] for r in value['files']}
    for path,sha in {
        'native-archive-v3/manifest.json':'4eac15590dc9def6dfb5e34b2cea92f30d9a067fabe0940e2876b6acdcbe6dce',
        'native-archive-v3/index.json':'9dab79ad92f6d95ef76033fe279cd97352c6acbc5aa12fc35057c4e7f0f20149',
        'control40-stage-e658c20/inventory.json':'79401e2c1051361d150da644250e8acf87814bda1c48b167af8a34345214fa8f',
        'migration/native-reasoning32-retained-pilot-v18/training/training-state.pt':'75fd39b1dc9a7f66b8125a1804cab87cfad384446d592df4534c3e1cff5f0367',
        'migration/native-reasoning32-retained-pilot-v18/training/adapters.safetensors':'34edb69c696fef1ac02e498bb1351a7655442d6c7c041562ca145abfb3a77176',
        'migration/native-saved-stage-original-snapshot-v1/inventory.json':'318a30c25a47719ba745475a27c7b240f973ce9010a2dfc7873fa4dbb5535135',
        'migration/native-saved-stage-original-observation-v1.json':'04d8569f4d343271c7b234e87fc40806cb00b80caf0ef4f97ac33f03185bf136',
    }.items():
        if targets.get(path)!=sha:raise ValueError('Pinned original state or manifest changed: '+path)
    carriers=value['archive_carriers']
    if set(carriers)!={'native','control40'}:raise ValueError('Exact two preserved archive alternatives required')
    carrier_bytes=0
    for name,count,expanded in (('native',35,67764285650),('control40',1,8000842614)):
        group=carriers[name]
        if len(group['containers'])!=count or group['expanded_bytes']!=expanded:
            raise ValueError('Complete archive carrier set required')
        carrier_bytes+=rows([dict(path=r['workspace_path'],bytes=r['bytes'],sha256=r['sha256'])
            for r in sorted(group['containers'],key=lambda r:r['workspace_path'])])
    transport=total-67764285650-8000842614+carrier_bytes
    if value['archive_transport_bytes']!=transport or transport>LIMIT:
        raise ValueError('Archive transport exceeds or differs from the proposed scope')
    return dict(schema=SCHEMA,regular_files=len(value['files']),expanded_artifact_bytes=total,
        archived_symlinks=len(links),all_paths_scoped_and_hashes_present=True,
        transfer_or_compute_authorized=False,destination_verified=False,model_adopted=False)


def load(path,expected_sha256):
    if Path(path).stat().st_size>10_000_000:raise ValueError('Bounded compressed catalog required')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Committed compressed inventory bytes changed')
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as reader:decoded=reader.read(MAX_JSON_BYTES+1)
    if len(decoded)>MAX_JSON_BYTES:raise ValueError('Bounded inventory document required')
    value=json.loads(decoded,object_pairs_hook=unique)
    return value,validate(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();_,report=load(a.inventory,a.sha256);print(json.dumps(report))


if __name__=='__main__':main()
