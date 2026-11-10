"""Prepared provider-quota storage check; not wired into an active learner.

Avoid repeated full HOME metadata walks. Conservatively cap the ENTIRE HOME
at32GiB, thereby keeping the32GiB output and80GiB HOME headroom thresholds.
The native scheduler freshness remains45s; storage receipts remain120s.
"""
import json,os,re,subprocess,time
from pathlib import Path
from .matched_interrupted import DURABLE

FILESYSTEM='data1.vastp2.isambard.ac.uk:/p2/home'
HOME=Path('/home/u6ada/dsyj94.u6ada')
WHOLE_HOME_LIMIT=32*1024**3


def parse_quota(text,uid):
    if not re.search(r'Disk quotas for user dsyj94\.u6ada \(uid '+str(uid)+r'\):',text):
        raise ValueError('Actual provider quota must identify this owner UID')
    lines=[line.split() for line in text.splitlines() if line.split() and line.split()[0]==FILESYSTEM]
    if len(lines)!=1:raise ValueError('Exactly one actual allocated NFS HOME quota required')
    fields=lines[0]
    if len(fields)!=7 or any(not value.isdigit() for value in fields[1:]):
        raise ValueError('Strict raw provider KiB quota record required; no grace/over-limit/human units')
    used,soft,hard,files,file_soft,file_hard=map(int,fields[1:])
    if (used*1024>WHOLE_HOME_LIMIT or not 100*1024**2<=hard<=110*1024**2
            or files>file_hard or file_hard<=0):
        raise RuntimeError('Provider quota exceeds the conservative whole-HOME32GiB cap or differs from allocated HOME')
    return dict(home_bytes=used*1024,run_bytes=used*1024,
        home_limit_GiB=100,run_limit_GiB=32,separate_filesystem=True,
        measurement='raw provider quota; entire HOME usage conservatively bounds this run',
        provider_filesystem=FILESYSTEM,provider_block_unit_bytes=1024,provider_home_hard_bytes=hard*1024,
        provider_home_files=files,whole_HOME_limit_bytes=WHOLE_HOME_LIMIT)


def storage_check(root=DURABLE):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Actual allocated output directory required')
    root.resolve().relative_to(HOME.resolve())
    if HOME.stat().st_dev==Path('/projects/u6ada').stat().st_dev:
        raise ValueError('Actual separate allocated HOME filesystem required')
    env=dict(os.environ,LC_ALL='C');began=time.monotonic()
    result=subprocess.run(['quota','-w','-v'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        universal_newlines=True,timeout=15,env=env)
    if result.returncode:raise RuntimeError('Actual bounded provider quota query failed; never refresh stale storage')
    record=parse_quota(result.stdout,os.getuid());record['provider_query_seconds']=time.monotonic()-began
    return record


if __name__=='__main__':print(json.dumps(storage_check()),flush=True)
