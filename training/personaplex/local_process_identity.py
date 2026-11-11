"""Read-only Linux process identity for a proposed paced heartbeat.

Not connected to the running guard. A matching process is local liveness
evidence only: it never renews a Slurm observation or proves job allocation.
"""
from pathlib import Path


def _stat(path, pid):
    text = path.read_text()
    # comm may contain whitespace and closing parentheses; fields follow its
    # final closing parenthesis. starttime is field22, state is field3.
    end = text.rfind(')')
    if end < 0 or text.split('(', 1)[0].strip() != str(pid):
        raise RuntimeError('Invalid process stat identity')
    fields = text[end + 1:].split()
    if len(fields) < 20 or not fields[19].isdigit():
        raise RuntimeError('Missing process start clock')
    if fields[0] in ('Z', 'X', 'x'):
        raise RuntimeError('Process is no longer live')
    return int(fields[19])


def capture(pid, proc_root='/proc'):
    """Capture PID reuse-resistant identity without a scheduler RPC."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('Positive integer PID required')
    root = Path(proc_root)
    process = root / str(pid)
    start = _stat(process / 'stat', pid)
    boot = (root / 'sys/kernel/random/boot_id').read_text().strip()
    status = (process / 'status').read_text().splitlines()
    uid_rows = [line.split()[1:] for line in status if line.startswith('Uid:')]
    if len(uid_rows) != 1 or len(uid_rows[0]) != 4 or not all(x.isdigit() for x in uid_rows[0]):
        raise RuntimeError('Complete process UID identity required')
    cgroup = (process / 'cgroup').read_text().strip()
    if not boot or not cgroup:
        raise RuntimeError('Boot and cgroup identity required')
    # Reject a process disappearing/reusing the PID while evidence was read.
    if _stat(process / 'stat', pid) != start:
        raise RuntimeError('Process identity changed during observation')
    return dict(pid=pid, start_ticks=start, boot_id=boot,
                uids=[int(x) for x in uid_rows[0]], cgroup=cgroup)


def verify(expected, proc_root='/proc'):
    """Compare against an identity captured by the admitted parent launcher."""
    if set(expected) != {'pid', 'start_ticks', 'boot_id', 'uids', 'cgroup'}:
        raise ValueError('Exact launcher identity required')
    actual = capture(expected['pid'], proc_root)
    if actual != expected:
        raise RuntimeError('Parent process identity changed')
    return actual
