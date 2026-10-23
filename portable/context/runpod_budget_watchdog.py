"""Local deadline guard for one explicitly recorded, volume-backed Runpod trial.

No provisioning, training or credential transfer. The credential stays in an
ignored mode0600 file on this Mac. Run this before creating the execution receipt.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def budget_seconds(receipt):
    """An explicit bounded trial must fit inside the user's existing50USD budget."""
    seconds = receipt['approved_gpu_wallclock_seconds']
    if type(seconds) is not int or seconds not in (7200, 14400, 21600, 28800, 36000):
        raise ValueError('Only explicit bounded2/4/6/8/10-hour trials are supported')
    names = ('gpu_hourly_usd', 'maximum_gpu_usd', 'total_compute_budget_usd', 'prior_attributed_spend_usd')
    values = {name: receipt[name] for name in names}
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values.values()):
        raise ValueError('Finite nonnegative budget figures required')
    maximum = values['gpu_hourly_usd'] * seconds / 3600
    if (not 0 < values['total_compute_budget_usd'] <= 50 or values['gpu_hourly_usd'] <= 0
            or abs(values['maximum_gpu_usd'] - maximum) > 1e-8
            or values['prior_attributed_spend_usd'] + maximum + .25 > values['total_compute_budget_usd']):
        raise ValueError('Trial must fit the existing at-most50USD budget including0.25USD disk reserve')
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    if args.key_file.stat().st_mode & 0o077:
        raise RuntimeError('Cost-control credential must have mode0600')
    key = args.key_file.read_text().strip()

    def api(path, method='GET', body=None):
        req = urllib.request.Request('https://api.runpod.io/v2' + path,
            data=None if body is None else json.dumps(body).encode(), method=method,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read()
        return json.loads(raw) if raw else {}

    def record(**values):
        values.update(at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid())
        tmp = args.status.with_suffix('.tmp')
        tmp.write_text(json.dumps(values, indent=2) + '\n')
        tmp.replace(args.status)

    awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        record(phase='waiting_for_owned_pod_receipt', can_create_resources=False)
        wait_deadline = time.monotonic() + 1800
        while not args.receipt.exists():
            if time.monotonic() > wait_deadline:
                raise RuntimeError('No trial Pod receipt appeared within30minutes')
            time.sleep(2)
        receipt = json.loads(args.receipt.read_text())
        seconds = budget_seconds(receipt)
        pod_id = receipt['pod_id']
        name = receipt['pod_name']
        volume = receipt['network_volume_id']
        assert name.startswith('duplex-native-validation-') and volume
        created = dt.datetime.fromisoformat(receipt['created_at_utc'].replace('Z', '+00:00'))
        deadline = created + dt.timedelta(seconds=seconds)
        while True:
            now = dt.datetime.now(dt.timezone.utc)
            try:
                result = api('/pods/' + pod_id)
                pod = result.get('pod', result)
                assert pod['id'] == pod_id and pod['name'] == name
                assert any(m['volumeId'] == volume for m in pod['mounts']['network'])
                state = pod['status']
                if state in ('EXITED', 'STOPPED', 'TERMINATED'):
                    record(phase='pod_not_billing_compute', pod_id=pod_id, state=state,
                           network_volume_retained=volume)
                    return
                if now >= deadline:
                    # Terminate this trial only; all durable artifacts are on the
                    # separately retained network volume. No other Pod is touched.
                    api('/pods/' + pod_id, method='DELETE')
                    record(phase='deadline_termination_requested', pod_id=pod_id,
                           deadline_utc=deadline.isoformat(), network_volume_retained=volume)
                    time.sleep(2)
                    continue
                record(phase='armed', pod_id=pod_id, state=state,
                       deadline_utc=deadline.isoformat(), network_volume_retained=volume)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    record(phase='pod_terminated', pod_id=pod_id, network_volume_retained=volume)
                    return
                record(phase='api_error_retry', pod_id=pod_id, http_status=error.code,
                       deadline_utc=deadline.isoformat())
            except (TimeoutError, OSError) as error:
                record(phase='network_error_retry', pod_id=pod_id,
                       error_type=type(error).__name__, deadline_utc=deadline.isoformat())
            time.sleep(5 if now >= deadline else min(20, max(1, (deadline-now).total_seconds())))
    finally:
        awake.terminate()


if __name__ == '__main__':
    main()
