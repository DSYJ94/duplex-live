"""Hash-pin completed native cases; never adopt partial cases or edit originals."""
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('No aliased reuse files')
        if path.is_file():
            result[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def verify(root, manifest):
    root = Path(root)
    for name, expected in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('No aliased reuse manifest')
        path = root / relative
        if (root.is_symlink() or any((root / Path(*relative.parts[:i])).is_symlink()
                                    for i in range(1, len(relative.parts)+1))
                or not path.is_file() or path.stat().st_size != expected['bytes']
                or sha(path) != expected['sha256']):
            raise ValueError('Reused native output hash mismatch')


def pin(parent, rows, expected_profile):
    """Only results.json-committed cases qualify; unfinished folders stay behind."""
    parent = Path(parent)
    profile = json.loads((parent/'profile.json').read_text())
    if any(profile.get(k) != v for k, v in expected_profile.items()):
        raise ValueError('Original evaluation profile differs')
    result = json.loads((parent/'results.json').read_text())
    if result.get('source') != profile['source'] or result.get('required_cases') != 32:
        raise ValueError('Original collection producer/cohort differs')
    expected = {r['id']: r for r in rows}
    if len(expected) != 32 or len(rows) != 32:
        raise ValueError('Full frozen32 input plan required')
    manifest = {'profile.json': dict(bytes=(parent/'profile.json').stat().st_size, sha256=sha(parent/'profile.json')),
                'results.json': dict(bytes=(parent/'results.json').stat().st_size, sha256=sha(parent/'results.json'))}
    seen = set()
    for case in result['cases']:
        identity = case['id']
        if identity in seen or identity not in expected or Path(identity).name != identity:
            raise ValueError('Duplicate or unexpected native case')
        seen.add(identity)
        row = expected[identity]
        folder = parent/identity
        if (case.get('complete') is not True or case.get('input_sha256') != row['input_sha256']
                or case.get('family') != row['family'] or case.get('input_frames') != row['frames']
                or json.loads((folder/'result.json').read_text()) != case):
            raise ValueError('Only complete exact native cases may be reused')
        trace = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        # Legacy timeline frames are one second; native streaming frames are 80ms.
        native_frames=(round(row['frames']*24000)+1919)//1920
        if (len(trace) != native_frames or [t['sequence'] for t in trace] != list(range(native_frames))):
            raise ValueError('Every original native frame required')
        for frame in trace:
            name = frame['chunk_file']
            if Path(name).name != name or not (folder/name).is_file():
                raise ValueError('Every original chunk required')
        manifest.update({identity+'/'+k: v for k, v in files(folder).items()})
    # Keep complete independent audit only for a complete cohort.
    if len(seen) == 32 and (parent/'report.json').exists():
        report = json.loads((parent/'report.json').read_text())
        if (report.get('source') != profile['source'] or report.get('trainer_source') != profile['trainer_source']
                or report.get('complete') is not True or len(report.get('cases', [])) != 32
                or {c['id'] for c in report['cases']} != seen):
            raise ValueError('Complete original audit identity required')
        for case in report['cases']:
            for chunk in case['source_chunks']:
                actual = manifest.get(case['id']+'/'+chunk['file'])
                if not actual or actual['sha256'] != chunk['sha256']:
                    raise ValueError('Original audited chunk differs')
        manifest['report.json'] = dict(bytes=(parent/'report.json').stat().st_size, sha256=sha(parent/'report.json'))
    verify(parent, manifest)
    return dict(parent=str(parent), profile=profile, cases=result['cases'], files=manifest,
                original_source=profile['source'], full_audit='report.json' in manifest,
                original_outputs_modified=False)


def materialize(record, target):
    """Copy pinned output bytes through rsync; preserve producer in reuse receipt."""
    parent, target = Path(record['parent']), Path(target)
    verify(parent, record['files'])
    target.mkdir(exist_ok=False)
    # NUL-delimited file list avoids shell interpretation and includes only admitted files.
    names = list(record['files'])
    subprocess.run(['rsync', '-ar', '--no-owner', '--no-group', '--exclude=session-token',
                    '--from0', '--files-from=-', str(parent)+'/', str(target)+'/'],
                   input=('\0'.join(names)+'\0').encode(), check=True)
    verify(target, record['files'])
    (target/'reuse-origin.json').write_text(json.dumps(record, indent=2)+'\n')
    return record['cases']


def main(args):
    from .matched_interrupted import DURABLE
    from .foundation_context import cases
    work=Path.cwd()
    index_path=Path(args.index)
    if sha(index_path)!=args.index_sha:
        raise ValueError('Pinned historical profile index required')
    index=json.loads(index_path.read_text())
    parent=Path(index['parent_root']);parent.resolve().relative_to(DURABLE.resolve())
    output=Path(args.output);output.resolve().relative_to(DURABLE.resolve())
    rows=cases(work)
    records={}
    for name, expected in index['conditions'].items():
        base,capacity=expected['base'],expected['capacity']
        folder=parent/name
        if sha(folder/'profile.json')!=expected['profile_sha256']:
            raise ValueError('Historical profile bytes differ')
        for kind in ('results','report'):
            expected_hash=expected.get(kind+'_sha256')
            if expected_hash and sha(folder/(kind+'.json'))!=expected_hash:
                raise ValueError('Historical collection/audit bytes differ')
        record=pin(folder,rows,dict(source=index['original_source'],
            trainer_source=index['trainer_source'],base=base,capacity=capacity,
            seed=42,temp_audio=.8,temp_text=.7,optimizer_updates=0))
        records[name]=record
    output.mkdir(exist_ok=False)
    summary={}
    for name,record in records.items():
        path=output/(name+'.json');path.write_text(json.dumps(record,indent=2)+'\n')
        summary[name]=dict(record=str(path),sha256=sha(path),cases=len(record['cases']),
                           full_audit=record['full_audit'])
    result=dict(conditions=summary,original_source=index['original_source'],
                trainer_source=index['trainer_source'],models_loaded=0,optimizer_updates=0,
                source_outputs_modified=False,required_comparison_outputs=192)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','index-sha','output'):parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
