"""CPU-only proposed two-clock contract. NOT installed in the live guard.

Provider observation time never becomes a local process observation time.
Requires a separate owner policy amendment before runtime integration.
"""
import math

PROVIDER_INTERVAL=60
PROVIDER_TTL=90
LOCAL_TTL=45


def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)


def provider_due(now,last_attempt):
    if not finite_number(now):raise ValueError('Finite clock required')
    if last_attempt is not None and (not finite_number(last_attempt) or last_attempt>now):
        raise ValueError('Invalid provider attempt clock')
    return last_attempt is None or now-last_attempt>=PROVIDER_INTERVAL


def validate(lease,provider,local,now):
    if not finite_number(now):raise ValueError('Finite clock required')
    for key in ('job_id','account','node'):
        if not isinstance(lease.get(key),str) or not lease[key].strip():
            raise RuntimeError('Complete allocation identity required')
    for key in ('stop_unix','maximum_seconds','prior_GPU_seconds'):
        if not finite_number(lease.get(key)) or lease[key]<0:
            raise RuntimeError('Finite nonnegative original clocks required')
    if type(lease.get('GPU_count')) is not int or lease['maximum_seconds']<=180:
        raise RuntimeError('Usable bounded allocation required')
    for record,ttl in ((provider,PROVIDER_TTL),(local,LOCAL_TTL)):
        stamp=record.get('observed_unix')
        if (type(stamp) not in (int,float) or not math.isfinite(stamp)
                or not 0<=now-stamp<=ttl):raise RuntimeError('Missing, future or expired evidence')
        for key in ('job_id','account','node','GPU_count'):
            if record.get(key)!=lease.get(key):raise RuntimeError('Allocation identity differs')
    if (provider.get('state')!='RUNNING' or provider.get('only_project_job') is not True
            or provider.get('stop_unix')!=lease['stop_unix']):
        raise RuntimeError('Provider allocation changed or another job appeared')
    if (local.get('process_alive') is not True or local.get('same_pid_start') is not True
            or local.get('same_job_cgroup') is not True or local.get('same_UID') is not True
            or local.get('storage_admission_valid') is not True):
        raise RuntimeError('Local process/storage proof failed')
    if now>=lease['stop_unix']-180:raise RuntimeError('Preservation/deadline reserve exhausted')
    if (lease['GPU_count']!=2 or lease['maximum_seconds']>86400
            or lease['prior_GPU_seconds']+lease['GPU_count']*lease['maximum_seconds']>112*3600):
        raise RuntimeError('Original aggregate/resource limits exceeded')
    return dict(local_observed_unix=local['observed_unix'],provider_observed_unix=provider['observed_unix'],
        local_max_age_seconds=LOCAL_TTL,provider_max_age_seconds=PROVIDER_TTL,
        owner_amendment_still_required=True,runtime_integration=False)
