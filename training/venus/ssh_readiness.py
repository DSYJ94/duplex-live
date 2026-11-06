"""Bounded SSH readiness with preserved failures; never starts a VM."""
import json
import os
import signal
import subprocess
import time


def probe(argv, timeout):
    # A timed-out SSH must not leave its IAP proxy running in the background.
    child = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        raise RuntimeError('SSH/IAP readiness probe timed out')
    if child.returncode:
        raise RuntimeError(stderr[-2400:])
    return stdout


def wait(c, deadline, *, clock=time.monotonic, sleep=time.sleep, check=probe):
    """Retry transport only until an existing boot deadline, then fail closed."""
    attempts = []
    receipt = c.R / 'ssh-readiness.json'
    while clock() < deadline:
        start = clock()
        try:
            out = check(c.ssh_argv + ['printf DUPLEX_SSH_READY'],
                        min(20, deadline - start))
            if out != 'DUPLEX_SSH_READY':
                raise RuntimeError('Unexpected SSH readiness response')
        except RuntimeError as error:
            message = str(error)[-2400:]
            attempts.append(dict(elapsed_seconds=clock()-start, error=message))
            receipt.write_text(json.dumps(dict(ready=False, attempts=attempts), indent=2)+'\n')
            if any(x in message for x in ('Permission denied (publickey)',
                    'REMOTE HOST IDENTIFICATION HAS CHANGED', 'Host key verification failed',
                    'REAUTHENTICATION_REQUIRED', 'Unable to create private file')):
                raise RuntimeError('Permanent SSH/access failure: '+message) from error
            sleep(max(0, min(3, deadline-clock())))
        else:
            receipt.write_text(json.dumps(dict(ready=True, attempts=attempts,
                                               successful_probe_seconds=clock()-start), indent=2)+'\n')
            return
    last = attempts[-1]['error'] if attempts else 'No boot time remained'
    raise RuntimeError('SSH boot deadline exhausted: '+last)
