"""Host-only fail-safe: stop this owned GPU if its Mac control guard is lost.

No cloud credentials or service account. Native provider STOP remains mandatory.
Run as a transient systemd service on the GPU host, never inside its container.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from portable.context.gcp_budget_watchdog import AUTH_SHA, PROJECT, stamp, validate_lease


def stop_reason(guard, lease, now):
    deadline = stamp(lease['stop_utc'])
    if now >= deadline:
        return 'approved_native_deadline_reached'
    if not isinstance(guard, dict):
        return 'control_guard_unreadable'
    if (guard.get('provider') != 'gcp' or guard.get('resource_id') != str(lease['id'])
            or guard.get('authorization_sha256') != AUTH_SHA
            or guard.get('deadline_unix') != deadline):
        return 'control_scope_changed'
    for field in ('execution_authorized', 'live_provider_deadline_verified',
                  'live_external_cost_guard_verified', 'single_gpu_across_providers_verified'):
        if guard.get(field) is not True:
            return 'control_permission_missing'
    checked = guard.get('checked_unix')
    if (type(checked) not in (int, float) or not math.isfinite(checked)
            or not 0 <= now-checked <= 90):
        return 'mac_guard_lost_or_stale'
    return None


def metadata(name):
    request = urllib.request.Request(
        'http://169.254.169.254/computeMetadata/v1/'+name,
        headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.headers.get('Metadata-Flavor') != 'Google':
            raise ValueError('Actual GCP metadata response required')
        return response.read(1024).decode().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lease', 'destination', 'guard', 'status'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0 or Path('/proc/1/comm').read_text().strip() != 'systemd':
        raise ValueError('Run on owned systemd host as root, not in a container')
    lease = json.loads(args.lease.read_text())
    authority = Path(__file__).resolve().parents[2]/'docs/GCP-NATIVE-CONTINUATION-AUTHORIZATION-2026-10-24.json'
    validate_lease(lease, authority.read_bytes(), args.destination.read_bytes())
    # Never shut down a staging CPU or some other instance on a bad invocation.
    if (metadata('project/project-id') != PROJECT
            or metadata('instance/id') != str(lease['id'])):
        raise ValueError('Guest identity differs from approved recorded GPU')

    def record(phase, reason=None):
        value = dict(schema='dl-gcp-guest-cost-guard-v1', phase=phase, reason=reason,
                     pid=os.getpid(), resource_id=str(lease['id']),
                     deadline_unix=stamp(lease['stop_utc']),
                     at_utc=datetime.now(timezone.utc).isoformat(),
                     cloud_credentials_required=False, optimizer_updates=0)
        tmp = args.status.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(args.status)

    while True:
        try:
            if args.guard.is_symlink() or args.guard.stat().st_size > 65536:
                raise ValueError('Bounded regular control guard required')
            guard = json.loads(args.guard.read_text())
            reason = stop_reason(guard, lease, time.time())
        except (OSError, ValueError, TypeError):
            reason = 'control_guard_unreadable'
        if reason:
            record('guest_shutdown_requested', reason)
            # Google's documented clean guest shutdown preserves the persistent
            # disk and discards only mandatory disposable local SSD contents.
            subprocess.run(['/sbin/shutdown', '-h', 'now'], check=True)
            return
        record('armed')
        time.sleep(5)


if __name__ == '__main__':
    main()
