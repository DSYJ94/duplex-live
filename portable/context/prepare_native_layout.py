"""Recreate original native paths with Git source after full artifact verification.

Historical .git metadata stays in the archive. No existing file or dirty checkout
is overwritten. Run again on a new container to recreate its external aliases.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

PAYLOAD = '741872fe2f9e49f35d4deba9aba96caff26c51d4'
CONTROLLER = 'd08ef7a09f17dbb3600827aa989bfe23e77f4bc7'
VENDOR = '79df302fa2820cae10c61bb2b7f302c52edb6ed2'
MANIFEST = '4eac15590dc9def6dfb5e34b2cea92f30d9a067fabe0940e2876b6acdcbe6dce'


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise RuntimeError('Refusing to replace an existing path: '+str(target))
    else:
        target.symlink_to(source, target_is_directory=True)


def checkout(source, target, commit):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',str(source),str(target)], check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',commit], check=True)
    if git(target,'rev-parse','HEAD') != commit or git(target,'status','--porcelain'):
        raise RuntimeError('Wrong or dirty immutable checkout: '+str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume', type=Path, required=True)
    p.add_argument('--git-source', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    approved = json.loads((a.volume/'migration/native-full-extracted-verification-v1.json').read_text())
    assert approved['all_size_sha256_and_symlinks_match']
    assert approved['manifest_sha256'] == MANIFEST and approved['files'] == 28470
    assert approved['bytes'] == 67764285650 and approved['symlinks'] == 8
    archive = a.volume/'native-preservation-v3'
    root = a.volume/'duplex-live'
    root.mkdir(exist_ok=True)
    for name in ('models','runs','tmp','logs'):
        link(archive/name, root/name)
    checkout(a.git_source, root/'worktrees/recovery-741872f', PAYLOAD)
    checkout(a.git_source, root/'worktrees/recovery-d08ef7a', CONTROLLER)
    bundle = archive/('runs/local-preservation-vendor-sources-20261019/'
                      'context-spanning-'+VENDOR+'.bundle')
    checkout(bundle, root/('vendor/context-spanning-'+VENDOR), VENDOR)
    for target in ('/projects/u6si/duplex-live', '/lus/lfs1aip2/projects/u6si/duplex-live'):
        link(root, Path(target))
    result = {'schema':'dl-native-runtime-layout-v1',
              'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'full_native_artifact_verification_required_and_passed':True,
              'runtime_root':str(root), 'payload_commit':PAYLOAD,
              'controller_commit':CONTROLLER,'vendor_commit':VENDOR,
              'originals_unchanged':True,'source_deployed_with_git':True,
              'training_or_model_quality_approved':False}
    with a.report.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
