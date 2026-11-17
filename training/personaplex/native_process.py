"""Process provenance; container PID numbers alone are not global identities."""
import os
from pathlib import Path


def process_identity():
    tail=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),pid_namespace=os.readlink('/proc/self/ns/pid'),start_ticks=int(tail[19]),node=os.uname().nodename)


def validate_restore(saved,bound,cache_source,execution,lease,current):
    if (saved.get('schema')!='dl-native-zero-ret-restore-v1'or saved['identity']!=bound or saved['source']!=cache_source
            or saved['step']!=0 or saved['execution']==execution):
        raise ValueError('Exact zero-state identity and new execution required')
    parent=lease.get('restore_parent')
    if parent is not None:
        if (parent.get('job_id')==lease['job_id']or parent.get('state')not in ('FAILED','COMPLETED')
                or parent.get('cache_source')!=cache_source or parent.get('cache_independently_verified')is not True):
            raise ValueError('Different allocation, terminal parent and verified original cache required')
    else:
        prior=saved.get('process_identity')
        if not prior or prior==current or (prior.get('pid_namespace'),prior.get('start_ticks'))==(current.get('pid_namespace'),current.get('start_ticks')):
            raise ValueError('Distinct OS process identity required within an allocation')
