"""Bounded capacity-only starts of the one explicitly approved existing VM."""
import json,time
from pathlib import Path
from datetime import datetime
from . import controller as c
from portable.venus.guard import bounded_runpod_read


def read_pods():
    result=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip())
    return result.get('pods',[]) if isinstance(result,dict) else result


def start(old,claim):
    assert old['id']=='3873014408897233554' and old['status']=='TERMINATED'
    assert claim['maximum_prestart_attempts']==3
    assert claim['prestart_deadline_unix']==claim['prestart_started_unix']+2700
    previous_terminal=None
    for attempt in range(1,4):
        if previous_terminal is not None:
            due=previous_terminal+600
            c.record('capacity_retry_wait',attempt=attempt,earliest_start_unix=due,deadline_unix=claim['prestart_deadline_unix'],paid_GPU_running=False)
            while time.time()<due:
                assert time.time()<claim['prestart_deadline_unix'],'Immutable prestart window expired'
                time.sleep(min(20,due-time.time()))
        assert time.time()<claim['prestart_deadline_unix'],'Immutable prestart window expired'
        price=json.loads(Path('runs/venus-training-20261101/current-a100-price.json').read_text())
        assert time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600 and 0<price['hourly_estimate_usd']<=4
        assert all(v['status']=='TERMINATED' for v in c.cloud('compute','instances','list'))
        assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in read_pods())
        before=c.cloud('compute','instances','describe',old['name'],'--zone=us-central1-a')
        assert before['id']==old['id'] and before['status']=='TERMINATED' and before.get('lastStartTimestamp')==old.get('lastStartTimestamp')
        assert time.time()<claim['prestart_deadline_unix']
        op=c.cloud('compute','instances','start',old['name'],'--zone=us-central1-a','--async');op=op[0] if isinstance(op,list) else op
        c.record('same_vm_capacity_attempt',attempt=attempt,operation=op['name'],source=c.source)
        until=min(time.time()+900,claim['prestart_deadline_unix'])
        while op['status']!='DONE':
            assert time.time()<until,'Observe SAME operation after timeout; never resubmit'
            time.sleep(4);op=c.cloud('compute','operations','describe',op['name'],'--zone=us-central1-a')
        (c.R/('start-attempt-%d-terminal.json'%attempt)).write_text(json.dumps(op,indent=2)+'\n')
        actual=c.cloud('compute','instances','describe',old['name'],'--zone=us-central1-a')
        if not op.get('error'):
            assert actual['id']==old['id'] and actual['status']=='RUNNING'
            assert actual['lastStartTimestamp']!=old.get('lastStartTimestamp')
            return actual,attempt
        capacity=any(e.get('code','').startswith('ZONE_RESOURCE_POOL_EXHAUSTED') for e in op['error'].get('errors',[]))
        assert capacity and actual['status']=='TERMINATED' and actual.get('lastStartTimestamp')==old.get('lastStartTimestamp'),'No retry of access, partial start, or noncapacity failure'
        previous_terminal=time.time()
        c.record('same_vm_terminal_stockout',attempt=attempt,operation=op['name'],observed_terminal_unix=previous_terminal,gpu_intervals=0)
    raise RuntimeError('All three approved prestart attempts stockout; zero model/GPU intervals')
