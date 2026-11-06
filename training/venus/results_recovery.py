"""One CPU-only missing-results recovery; never starts a GPU or loads a model.

Run --plan offline first. Execution requires the separate owner-approved amendment.
The original VM and its unique disk are never deleted, even on transfer failure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request

PROJECT = 'aurium-duplex-live-20261024'
ZONE = 'europe-west4-a'
GPU = 'duplex-venus-large-gpu-20261104-eua-v67'
GPU_ID = '7521667153519853425'
DISK = 'duplex-venus-large-data-20261104-eua-v67'
DISK_ID = '2297341858915206463'
CPU = 'duplex-venus-results-cpu-20261106-v1'
R = Path('runs/venus-large-capacity-20261104/results-recovery-v82')
PRIOR = Path('runs/venus-large-capacity-20261104/v65/gpu-12')
PROPOSAL = Path('docs/NATIVE-VENUS-RESULTS-CPU-RECOVERY-PROPOSAL-2026-11-06.json')
AUTH = Path('docs/NATIVE-VENUS-RESULTS-CPU-RECOVERY-AUTHORIZATION-2026-11-06.json')
INVENTORY_SHA = 'e36199a1ad93b745281586c2fc839860f24cf982d7849b01060c6b0845c7d6c2'
REMOTE = '/workspace/migration/venus-large-capacity-v65/native-v75'
DEST = Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v75/resume12')
DEADLINE = 1794052800.0
GCLOUD = '/opt/homebrew/bin/gcloud'
EXCLUDED = frozenset(('.private', 'credentials', 'owner-conversations', 'session-token', 'API keys.md'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def save(name, value):
    R.mkdir(parents=True, exist_ok=True)
    (R / name).write_text(json.dumps(value, indent=2) + '\n')


def run(argv, timeout=60):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr[-2000:])
    return p.stdout


def cloud(*args):
    return json.loads(run([GCLOUD, *args, '--project=' + PROJECT, '--quiet', '--format=json']))


def checked_path(root, value):
    p = PurePosixPath(value)
    if not value or '\x00' in value or '\n' in value or p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe inventory path')
    if any(x in EXCLUDED or x == 'core' or x.startswith('core.') for x in p.parts):
        raise ValueError('Excluded artifact in inventory')
    local = root.joinpath(*p.parts)
    # Do not follow a substituted symlink directory outside the approved backup.
    if not local.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError('Inventory parent escapes destination')
    return local


def manifest():
    p = PRIOR / 'remote-inventory.json'
    if sha(p) != INVENTORY_SHA: raise ValueError('Original full inventory changed')
    inv = json.loads(p.read_text())
    if len(inv['files']) != 1856 or sum(r['bytes'] for r in inv['files']) != 17330768218 or len(inv['links']) != 1:
        raise ValueError('Incomplete original inventory')
    paths = set()
    for row in inv['files'] + inv['links']:
        checked_path(DEST, row['path'])
        if row['path'] in paths: raise ValueError('Duplicate inventory path')
        paths.add(row['path'])
    m = json.loads((PRIOR / 'offline-copyback-materialization.json').read_text())
    rows = m['missing_files']
    originals = {r['path']: r for r in inv['files']}
    if len(rows) != 1692 or sum(r['bytes'] for r in rows) != 204312087:
        raise ValueError('Missing cohort exceeds prepared recovery')
    if len({r['path'] for r in rows}) != len(rows): raise ValueError('Duplicate missing artifact')
    if any(originals.get(r['path']) != r for r in rows): raise ValueError('Missing cohort changed')
    rows = sorted(rows, key=lambda r: (not r['path'].endswith(('.json', '.jsonl', '.txt', '.log', '.py', '.html')), r['path']))
    return inv, rows


def batches(rows):
    batch, size = [], 0
    for row in rows:
        if not 0 <= row['bytes'] <= 32 * 1024**2: raise ValueError('Artifact exceeds32MiB batch')
        if batch and size + row['bytes'] > 32 * 1024**2:
            yield batch
            batch, size = [], 0
        batch.append(row); size += row['bytes']
    if batch: yield batch


def require_authorization():
    a = json.loads(AUTH.read_text())  # Missing approval fails BEFORE any cloud call.
    p = json.loads(PROPOSAL.read_text())
    if a.get('status') != 'APPROVED' or not a.get('human_reply_verbatim'):
        raise ValueError('Explicit owner recovery approval required')
    if a.get('proposal_path') != str(PROPOSAL) or a.get('proposal_sha256') != sha(PROPOSAL):
        raise ValueError('Approval must bind exact recovery proposal')
    for key, value in dict(phase_ceiling_USD=175, project_ceiling_USD=350,
        recovery_maximum_USD=.1, maximum_CPU_running_intervals=1,
        maximum_CPU_running_seconds=900, original_GPU_id=GPU_ID,
        same_disk_id=DISK_ID, new_GPU_intervals=0, new_optimizer_updates=0).items():
        if a.get(key) != value or p.get(key) != value: raise ValueError('Recovery scope mismatch: ' + key)
    if time.time() >= DEADLINE - 1500: raise ValueError('No complete approved recovery window remains')
    return a


def validate_original(vm, disk, attached):
    if vm['id'] != GPU_ID or vm['status'] != 'TERMINATED' or disk['id'] != DISK_ID or int(disk['sizeGb']) != 250:
        raise ValueError('Original GPU/disk identity or STOP mismatch')
    bound = [d for d in vm.get('disks', []) if d.get('source', '').endswith('/disks/' + DISK)]
    if attached and (len(bound) != 1 or not bound[0]['boot'] or bound[0]['autoDelete']):
        raise ValueError('Original boot binding or retention mismatch')
    if not attached and bound: raise ValueError('Original disk remains attached')
    expected = [vm['selfLink']] if attached else []
    if disk.get('users', []) != expected: raise ValueError('Unexpected disk users')


def wait_operation(op, seconds=180):
    op = op[0] if isinstance(op, list) else op
    until = time.monotonic() + seconds
    while op['status'] != 'DONE':
        if time.monotonic() >= until: raise TimeoutError('Observe SAME operation; do not resubmit')
        time.sleep(3)
        op = cloud('compute', 'operations', 'describe', op['name'], '--zone=' + ZONE)
    if op.get('error'): raise RuntimeError(json.dumps(op['error']))
    return op


def validate_cpu(vm, allow_detached=False):
    bound = [d for d in vm.get('disks', []) if d.get('source', '').endswith('/disks/' + DISK)]
    if vm['name'] != CPU or vm.get('guestAccelerators') or not vm['machineType'].endswith('/e2-standard-2'):
        raise ValueError('Unexpected CPU machine')
    detached = allow_detached and not vm.get('disks') and vm['status'] == 'TERMINATED'
    if not detached and (len(vm.get('disks', [])) != 1 or len(bound) != 1 or not bound[0]['boot'] or bound[0]['autoDelete']):
        raise ValueError('CPU boot disk retention mismatch')
    scheduling = vm['scheduling']
    if int(scheduling['maxRunDuration']['seconds']) > 900 or scheduling['instanceTerminationAction'] != 'STOP':
        raise ValueError('Native CPU STOP bound missing')
    if vm.get('serviceAccounts') or vm['tags']['items'] != ['duplex-venus-train-bounded']:
        raise ValueError('CPU credentials or ingress scope changed')


def quote():
    """Read exact current regional OnDemand SKUs; no billing/resource mutation."""
    token = run([GCLOUD, 'auth', 'print-access-token']).strip()
    prefixes = {'core': 'E2 Instance Core running', 'ram': 'E2 Instance Ram running'}
    found, page = {}, ''
    now = datetime.now(timezone.utc)
    for pages in range(20):
        url = 'https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?currencyCode=USD&pageSize=5000'
        if page: url += '&pageToken=' + urllib.parse.quote(page, safe='')
        request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
        with urllib.request.urlopen(request, timeout=45) as response: data = json.load(response)
        for row in data.get('skus', []):
            if 'europe-west4' not in row.get('serviceRegions', []) or row['category']['usageType'] != 'OnDemand': continue
            for kind, prefix in prefixes.items():
                if not row['description'].startswith(prefix): continue
                active = [p for p in row['pricingInfo'] if datetime.fromisoformat(p['effectiveTime'].replace('Z', '+00:00')) <= now]
                price = max(active, key=lambda p: p['effectiveTime'])
                expression = price['pricingExpression']
                expected_unit = 'h' if kind == 'core' else 'GiBy.h'
                if expression['usageUnit'] != expected_unit: raise ValueError('Unexpected E2 pricing unit')
                value = expression['tieredRates'][0]['unitPrice']
                rate = dict(sku_id=row['skuId'], description=row['description'], regions=row['serviceRegions'],
                    usage_unit=expected_unit, effective_time=price['effectiveTime'],
                    usd=int(value.get('units', 0)) + value.get('nanos', 0) / 1e9)
                if kind in found and found[kind]['sku_id'] != rate['sku_id']: raise ValueError('Ambiguous regional E2 SKU')
                found[kind] = rate
        if len(found) == 2: break
        page = data.get('nextPageToken', '')
        if not page: break
    if len(found) != 2: raise ValueError('Complete Netherlands E2 quote required')
    price = dict(at_utc=now.isoformat(), source='Official Cloud Billing Catalog', region='europe-west4',
        machine_type='e2-standard-2', vcpus=2, ram_GiB=8, rates=found, catalog_pages=pages+1,
        hourly_usd=2*found['core']['usd'] + 8*found['ram']['usd'],
        existing_durable_disk_already_within_retention_grant=True, transfer_rounding_reserve_USD=.06)
    save('price.json', price)
    return price


def stop_cpu(allow_detached=False):
    vm = cloud('compute', 'instances', 'describe', CPU, '--zone=' + ZONE)
    lease_path = R / 'lease.json'
    if lease_path.exists():
        expected_id = json.loads(lease_path.read_text())['vm_id']
    else:
        # A create response can be lost although allocation succeeded. Bind
        # that exact request via its public receipt fingerprint, never retry it.
        metadata = {x['key']: x.get('value') for x in vm.get('metadata', {}).get('items', [])}
        expected_id = None
        if metadata.get('duplex-results-recovery-claim') == sha(R / 'claim.json'):
            expected_id = vm['id']
    if not expected_id or vm['id'] != expected_id:
        raise ValueError('Refuse to stop an unbound machine')
    validate_cpu(vm, allow_detached=allow_detached)
    if vm['status'] != 'TERMINATED':
        run([GCLOUD, 'compute', 'instances', 'stop', CPU, '--zone=' + ZONE, '--project=' + PROJECT, '--quiet'], 150)
    vm = cloud('compute', 'instances', 'describe', CPU, '--zone=' + ZONE)
    save('provider-final.json', vm)
    if vm['status'] != 'TERMINATED': raise ValueError('CPU not confirmed stopped')
    return vm


def guard():
    lease = json.loads((R / 'lease.json').read_text())
    try:
        while True:
            vm = cloud('compute', 'instances', 'describe', CPU, '--zone=' + ZONE)
            if vm['id'] != lease['vm_id']: raise ValueError('CPU identity changed')
            if vm['status'] == 'TERMINATED': return
            validate_cpu(vm)
            if time.time() > lease['stop_unix'] - 30: raise TimeoutError('Stop at recovery deadline')
            if run(['/bin/ps', '-p', str(lease['controller_pid']), '-o', 'lstart=']).strip() != lease['controller_birth']:
                raise ValueError('Controller gone or PID reused')
            original = cloud('compute', 'instances', 'describe', GPU, '--zone=' + ZONE)
            if original['id'] != GPU_ID or original['status'] != 'TERMINATED': raise ValueError('Original GPU not OFF')
            save('guard.json', dict(checked_unix=time.time(), expires_unix=min(time.time()+45, lease['stop_unix']), vm_id=vm['id']))
            time.sleep(10)
    finally:
        stop_cpu()


def prepare_ssh(vm):
    keys = []
    for _ in range(15):
        try:
            keys = cloud('compute', 'instances', 'get-guest-attributes', CPU, '--zone=' + ZONE, '--query-path=hostkeys/')
        except RuntimeError as error:
            if '404' not in str(error) or 'Guest Attribute' not in str(error): raise
            keys = []  # Guest agent has not published authenticated keys yet.
        if keys: break
        time.sleep(3)
    if not keys: raise RuntimeError('No authenticated host keys; no insecure fallback')
    known = R / 'known-hosts'
    alias = 'duplex-results-cpu-v82'
    known.write_text(''.join(alias + ' ' + r['key'] + ' ' + r['value'] + '\n' for r in keys))
    known.chmod(0o600)
    proxy = shlex.join([GCLOUD, 'compute', 'start-iap-tunnel', CPU, '22', '--listen-on-stdin', '--zone=' + ZONE, '--project=' + PROJECT, '--verbosity=error'])
    argv = ['ssh', '-i', 'runs/local-migration-20261024/.private/gcp-native-ssh',
        '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'HostKeyAlias=' + alias, '-o', 'UserKnownHostsFile=' + str(known),
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', '-o', 'ProxyCommand=' + proxy,
        'daryl_aurium_labs_com@' + vm['networkInterfaces'][0]['accessConfigs'][0]['natIP']]
    save('ssh.json', argv)
    return argv


def restore_binding(watch=None):
    """Idempotent cleanup of ONLY the temporary CPU; original VM/disk survive."""
    original = cloud('compute', 'instances', 'describe', GPU, '--zone=' + ZONE)
    disk = cloud('compute', 'disks', 'describe', DISK, '--zone=' + ZONE)
    if original['id'] != GPU_ID or original['status'] != 'TERMINATED' or disk['id'] != DISK_ID:
        raise ValueError('Refuse cleanup with unknown original identity/state')
    cpus = [v for v in cloud('compute', 'instances', 'list') if v['name'] == CPU]
    if cpus:
        cpu = stop_cpu(allow_detached=True)
        if watch:
            try: watch.wait(timeout=15)
            except subprocess.TimeoutExpired: watch.terminate()
        binding = [d for d in cpu.get('disks', []) if d.get('source', '').endswith('/disks/' + DISK)]
        if binding:
            if binding[0]['autoDelete']: raise ValueError('Refuse cleanup of auto-deleted unique disk')
            run([GCLOUD, 'compute', 'instances', 'detach-disk', CPU, '--disk=' + DISK, '--zone=' + ZONE, '--project=' + PROJECT, '--quiet'], 90)
    disk = cloud('compute', 'disks', 'describe', DISK, '--zone=' + ZONE)
    if not disk.get('users'):
        run([GCLOUD, 'compute', 'instances', 'attach-disk', GPU, '--disk=' + DISK, '--boot', '--device-name=persistent-disk-0', '--zone=' + ZONE, '--project=' + PROJECT, '--quiet'], 90)
        run([GCLOUD, 'compute', 'instances', 'set-disk-auto-delete', GPU, '--device-name=persistent-disk-0', '--no-auto-delete', '--zone=' + ZONE, '--project=' + PROJECT, '--quiet'], 90)
    original = cloud('compute', 'instances', 'describe', GPU, '--zone=' + ZONE)
    disk = cloud('compute', 'disks', 'describe', DISK, '--zone=' + ZONE)
    validate_original(original, disk, attached=True)
    save('original-binding-restored.json', dict(original_vm_id=GPU_ID, disk_id=DISK_ID, status=original['status'], auto_delete=False))
    if cpus:
        cpu = cloud('compute', 'instances', 'describe', CPU, '--zone=' + ZONE)
        if cpu['status'] != 'TERMINATED' or cpu.get('disks'): raise ValueError('Temporary CPU must be stopped and disk-free before deletion')
        run([GCLOUD, 'compute', 'instances', 'delete', CPU, '--zone=' + ZONE, '--project=' + PROJECT, '--keep-disks=all', '--quiet'], 90)
        save('temporary-cpu-removed.json', dict(name=CPU, original_vm_preserved=True, original_disk_preserved=True))


def execute():
    require_authorization()
    inv, rows = manifest()
    price = json.loads((R / 'price.json').read_text())
    age = time.time() - datetime.fromisoformat(price['at_utc']).timestamp()
    if price.get('source') != 'Official Cloud Billing Catalog' or price.get('region') != 'europe-west4' or price.get('machine_type') != 'e2-standard-2':
        raise ValueError('Wrong regional CPU quote')
    computed = 2*price['rates']['core']['usd'] + 8*price['rates']['ram']['usd']
    if not (computed == price['hourly_usd'] and 0 <= age < 3600 and 0 < computed <= .12 and computed * 1000 / 3600 + .06 <= .1):
        raise ValueError('Fresh CPU price and whole recovery reserve required')
    if not all(v['status'] == 'TERMINATED' for v in cloud('compute', 'instances', 'list')):
        raise ValueError('Other compute running')
    if any(v['name'] == CPU for v in cloud('compute', 'instances', 'list')):
        raise ValueError('Observe existing CPU, never duplicate')
    original = cloud('compute', 'instances', 'describe', GPU, '--zone=' + ZONE)
    disk = cloud('compute', 'disks', 'describe', DISK, '--zone=' + ZONE)
    validate_original(original, disk, attached=True)
    # Existing OSLogin public key must be valid; credentials never leave Mac.
    run([GCLOUD, 'auth', 'print-access-token'])  # Keep output in memory, never log it.
    source = run(['git', 'rev-parse', 'HEAD']).strip()
    if run(['git', '-c', 'core.filemode=false', 'status', '--porcelain', '--', __file__, str(PROPOSAL)]).strip():
        raise ValueError('Recovery source/proposal must be committed')
    with (R / 'claim.json').open('x') as f:
        json.dump(dict(controller_pid=os.getpid(), source=source, authorization_sha256=sha(AUTH)), f)
    watch = None
    detached = False
    try:
        detached = True  # A detach response may be lost after the actual move.
        run([GCLOUD, 'compute', 'instances', 'detach-disk', GPU, '--disk=' + DISK, '--zone=' + ZONE, '--project=' + PROJECT, '--quiet'], 90)
        validate_original(cloud('compute', 'instances', 'describe', GPU, '--zone=' + ZONE), cloud('compute', 'disks', 'describe', DISK, '--zone=' + ZONE), attached=False)
        boot = R / 'bootstrap.sh'
        boot.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-results-stop --on-active=810s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        op = cloud('compute', 'instances', 'create', CPU, '--zone=' + ZONE,
            '--machine-type=e2-standard-2', '--provisioning-model=STANDARD',
            '--no-restart-on-failure', '--no-service-account', '--no-scopes',
            '--tags=duplex-venus-train-bounded', '--disk=name=' + DISK + ',boot=yes,auto-delete=no,mode=rw',
            '--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE,duplex-results-recovery-claim=' + sha(R / 'claim.json'),
            '--metadata-from-file=startup-script=' + str(boot), '--max-run-duration=900s',
            '--instance-termination-action=STOP', '--async')
        save('create-operation.json', op)
        wait_operation(op)
        vm = cloud('compute', 'instances', 'describe', CPU, '--zone=' + ZONE)
        validate_cpu(vm)
        if vm['status'] != 'RUNNING': raise ValueError('CPU not actually running')
        save('provider-start.json', vm)
        end = datetime.fromisoformat(vm['lastStartTimestamp']).timestamp() + 900
        save('lease.json', dict(vm_id=vm['id'], stop_unix=end, controller_pid=os.getpid(),
            controller_birth=run(['/bin/ps', '-p', str(os.getpid()), '-o', 'lstart=']).strip()))
        watch = subprocess.Popen([sys.executable, '-u', '-m', 'training.venus.results_recovery', '--guard'], stdout=(R / 'guard.log').open('x'), stderr=subprocess.STDOUT)
        ssh = prepare_ssh(vm)
        run(ssh + ['sudo systemctl is-active duplex-results-stop.timer'], 30)
        active = run(ssh + ['sudo systemctl list-units --type=service --state=running --no-legend --plain'], 30)
        containers = run(ssh + ['sudo docker ps -q'], 30).split()
        probe = """import json,os,pathlib
found=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
    try: args=(p/'cmdline').read_bytes().decode(errors='replace').split('\\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if int(p.name)!=os.getpid() and any(a.startswith(('training.venus.','portable.venus.native_app')) or pathlib.PurePosixPath(a).name in ('torchrun','native_app.py','large_train.py') for a in args): found.append(int(p.name))
print(json.dumps(found))
"""
        model_pids = json.loads(run(ssh + ['/usr/bin/python3 -c ' + shlex.quote(probe)], 30))
        save('pre-transfer-runtime.json', dict(active_services=active, running_container_ids=containers, model_process_ids=model_pids))
        if containers or model_pids or any(x.startswith('duplex-large65-') for x in active.split()):
            raise ValueError('Model unit/container unexpectedly active')
        until = time.time() + 40
        while not (R / 'guard.json').exists():
            if watch.poll() is not None or time.time() >= until: raise RuntimeError('Independent guard not ready')
            time.sleep(1)
        for n, batch in enumerate(batches(rows)):
            if watch.poll() is not None or end - time.time() < 120 or time.time() > json.loads((R / 'guard.json').read_text())['expires_unix']:
                raise RuntimeError('Live recovery guard/time admission failed')
            filelist = R / ('batch-%03d.files' % n)
            filelist.write_bytes(b''.join(r['path'].encode() + b'\0' for r in batch))
            argv = ['/opt/homebrew/bin/rsync', '-ar', '--no-owner', '--no-group',
                *['--exclude=' + x for x in sorted(EXCLUDED | {'core', 'core.*'})],
                '--from0', '--files-from=' + str(filelist), '-e', shlex.join(ssh[:-1]), ssh[-1] + ':' + REMOTE + '/', str(DEST) + '/']
            output = run(argv, min(90, int(end - time.time() - 90)))
            for row in batch:
                p = checked_path(DEST, row['path'])
                if p.is_symlink() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
                    raise ValueError('Recovered artifact hash mismatch')
            save('batch-%03d-verified.json' % n, dict(paths=[r['path'] for r in batch], bytes=sum(r['bytes'] for r in batch), stdout=output))
        # No link transfer needed: the one original link was already verified.
        for row in inv['files']:
            p = checked_path(DEST, row['path'])
            if p.is_symlink() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
                raise ValueError('Final whole inventory mismatch')
        for row in inv['links']:
            p = checked_path(DEST, row['path'])
            if not p.is_symlink() or os.readlink(p) != row['target']: raise ValueError('Final link mismatch')
        save('copyback-verified.json', dict(all_hashes_verified=True, all_link_targets_verified=True,
            files=1856, bytes=17330768218, links=1, inventory_sha256=INVENTORY_SHA, new_optimizer_updates=0))
    finally:
        if detached:
            try: restore_binding(watch)
            except Exception as error:
                save('cleanup-incomplete.json', dict(error_type=type(error).__name__, error=str(error), original_disk_must_be_preserved=True))
                raise
        if watch:
            try: watch.wait(timeout=15)
            except subprocess.TimeoutExpired: watch.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--guard', action='store_true')
    mode.add_argument('--quote', action='store_true')
    args = parser.parse_args()
    if args.plan:
        inventory, rows = manifest()
        print(json.dumps(dict(missing_files=len(rows), missing_bytes=sum(r['bytes'] for r in rows), batches=len(list(batches(rows))), authorized=AUTH.exists(), cloud_mutations=0)))
    elif args.quote:
        price = quote()
        print(json.dumps(dict(hourly_usd=price['hourly_usd'], region=price['region'], cloud_mutations=0)))
    elif args.guard: guard()
    else: execute()
