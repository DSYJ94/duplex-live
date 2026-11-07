"""Root-agent OPAL broker: committed program edits, no extra model/API agent.

The existing agent reads each request and commits a justified optimizer edit (or
keeps it). This backend waits for that exact request's local decision. It never
authors code, launches an agent, changes data/loss, or makes provider calls.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time
import uuid

from .opal_adapter import load_upstream


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


class GitSupervisor:
    name = 'current-agent-git'

    def __init__(self, root, repo, timeout=240, verify_deployment=None):
        if type(timeout) is not int or not 1 <= timeout <= 300:
            raise ValueError('Finite supervisor decision deadline required')
        self.root, self.repo = Path(root), Path(repo).resolve()
        self.timeout, self.cancelled = timeout, False
        self.verify_deployment = verify_deployment

    def preflight(self):
        subprocess.run(['git', '-C', str(self.repo), 'rev-parse', '--git-dir'],
                       check=True, capture_output=True, timeout=10)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def cancel(self):
        self.cancelled = True

    def setup(self, run_dir, workspace):
        pass  # Upstream owns the mailbox/workspace and request identity.

    def state(self):
        return {}

    def restore(self, state):
        if state:
            raise ValueError('Unexpected state for the stateless Git decision transport')

    def decide(self, workspace, message, system, record_dir):
        started = time.monotonic()
        workspace = Path(workspace)
        current = {}
        for name in ('updater.py', 'analysis.py'):
            path = workspace / name
            if path.is_symlink() or path.stat().st_size > 256 * 1024:
                raise ValueError('Bounded regular OPAL programs required')
            current[name] = path.read_text()
        request = dict(request_id=uuid.uuid4().hex, message=message, system=system,
                       programs=current, created_unix=time.time(), timeout_seconds=self.timeout,
                       code_author='current LOCAL agent', broker_api_calls=0)
        encoded = json.dumps(request, sort_keys=True).encode()
        folder = self.root / request['request_id']
        folder.mkdir(mode=0o700)
        atomic(folder / 'request.json', request)
        atomic(folder / 'identity.json', dict(request_id=request['request_id'], request_sha256=digest(encoded)))
        while time.monotonic() - started < self.timeout:
            if self.cancelled:
                raise RuntimeError('OPAL decision cancelled before program application')
            ready = folder / 'decision.json'
            if ready.exists():
                decision = json.loads(ready.read_text())
                programs, source = self.verify(decision, request, digest(encoded))
                if programs and self.verify_deployment:
                    self.verify_deployment(source, decision['programs'], programs)
                if self.cancelled or time.monotonic() - started >= self.timeout:
                    raise TimeoutError('Decision retired while verifying Git deployment; no programs changed')
                # Validate every program before changing any. Upstream OPAL
                # subsequently owns its trials, state transfer and application.
                for name, content in programs.items():
                    if (workspace / name).is_symlink():
                        raise ValueError('Program destination cannot be a symlink')
                for name, content in programs.items():
                    temporary = workspace / (name + '.supervisor.tmp')
                    temporary.write_text(content)
                    os.replace(temporary, workspace / name)
                atomic(folder / 'committed-decision.json', dict(
                    request_id=request['request_id'], request_sha256=digest(encoded),
                    action=decision['action'], reason=decision['reason'], source=source,
                    program_hashes={n:digest(v.encode()) for n,v in programs.items()},
                    cost_usd=0, external_api_calls=0, optimizer_application_proven=False))
                return dict(text=decision['reason'], cost_usd=0., returncode=0, error=None,
                            seconds=time.monotonic()-started)
            time.sleep(.1)
        raise TimeoutError('No current-agent decision within the declared OPAL deadline')

    def verify(self, decision, request, request_sha256):
        if (not isinstance(decision, dict)
                or set(decision) != {'request_id', 'request_sha256', 'action', 'source', 'programs', 'reason'}
                or decision['request_id'] != request['request_id']
                or decision['request_sha256'] != request_sha256
                or decision['action'] not in ('keep', 'adapt')
                or not isinstance(decision['reason'], str) or not 0 < len(decision['reason']) <= 4000):
            raise ValueError('Decision must bind this exact live request and rationale')
        paths = decision['programs']
        if not isinstance(paths, dict) or not set(paths) <= {'updater.py', 'analysis.py'}:
            raise ValueError('Only optimizer and analysis programs may be proposed')
        if decision['action'] == 'keep':
            if paths or decision['source'] is not None:
                raise ValueError('Keep must not carry an executable change')
            return {}, None
        source = decision['source']
        if (not paths or not isinstance(source, str) or len(source) != 40
                or any(c not in '0123456789abcdef' for c in source)):
            raise ValueError('Adapt requires an immutable Git source commit')
        kind = subprocess.check_output(['git', '-C', str(self.repo), 'cat-file', '-t', source],
                                       text=True, timeout=10).strip()
        if kind != 'commit':
            raise ValueError('Source must identify an actual Git commit')
        programs = {}
        for name, path in paths.items():
            if not isinstance(path, str):
                raise ValueError('Committed optimizer program path required')
            rel = PurePosixPath(path)
            if (rel.is_absolute() or '..' in rel.parts or any(p.startswith('.') for p in rel.parts)
                    or rel.parts[:3] != ('training', 'venus', 'opal_programs') or rel.name != name):
                raise ValueError('Program must be in the committed optimizer program directory')
            content = subprocess.check_output(['git', '-C', str(self.repo), 'show', source + ':' + path], timeout=10)
            if len(content) > 256 * 1024:
                raise ValueError('Optimizer program exceeds the declared bound')
            text = content.decode('utf-8')
            ast.parse(text)  # Syntax only; no code executes in the Mac broker.
            programs[name] = text
        return programs, source


def local_backend(root,repo,timeout=120):
    """Pinned OPAL backend on the training host; current agent authors via Git.

    No paid coding CLI or API is launched. A new exact request is written for
    the LOCAL executor, which can commit/deploy its own code edits or keep.
    This removes the IAP/mailbox transport requirement on Isambard, not the
    request, source, deadline, checkpoint or native quality checks.
    """
    from opal.backends.base import Backend
    supervisor=GitSupervisor(root,repo,timeout=timeout)
    class LocalGitBackend(Backend):
        name='current-agent-git-zero-api'
        def __init__(self):self.timeout=timeout
        def preflight(self):supervisor.preflight()
        def decide(self,workspace,message,system,record_dir):
            result=supervisor.decide(workspace,message,system,record_dir)
            if result['cost_usd']!=0:raise ValueError('LOCAL Git backend cannot report paid API usage')
            return result
        def cancel(self):supervisor.cancel()
        def finish(self,interrupted=False):supervisor.cancel()
        def restore(self,state):
            if state:raise ValueError('Zero-API Git transport has no hidden persistent agent state')
    return LocalGitBackend()


@contextmanager
def existing_ssh(mailbox, argv):
    """Use the existing strict IAP argv; do not create a home-directory SSH alias.

    Narrow, process-local adapter for the exact pinned upstream _Folder seam.
    The caller still owns live provider/lease/guard/budget verification.
    """
    if (not isinstance(argv, list) or not all(isinstance(x, str) for x in argv)
            or not argv or Path(argv[0]).name != 'ssh'
            or any(x not in argv for x in ('StrictHostKeyChecking=yes', 'BatchMode=yes', 'IdentitiesOnly=yes'))
            or not any(x.startswith('UserKnownHostsFile=') for x in argv)
            or not any(x.startswith('ProxyCommand=') and 'start-iap-tunnel ' in x for x in argv)
            or '@' not in argv[-1]):
        raise ValueError('Use the current controller strict SSH/IAP argv unchanged')
    original = mailbox._Folder
    class Folder(original):
        def __init__(self, path, ssh=None):
            if ssh != 'existing-native-host':
                raise ValueError('Unexpected mailbox SSH target')
            super().__init__(path, ssh)
            self.base = list(argv)
    mailbox._Folder = Folder
    try:
        yield
    finally:
        mailbox._Folder = original


def main(a):
    load_upstream(a.source)
    from opal.backends import mailbox
    if bool(a.ssh_argv) != bool(a.remote_repo):
        raise ValueError('Remote broker requires the already Git-deployed repository path')
    argv = json.loads(Path(a.ssh_argv).read_text()) if a.ssh_argv else None
    def verify_deployment(source, paths, programs):
        for name, path in paths.items():
            command = 'git -C ' + shlex.quote(a.remote_repo) + ' show ' + shlex.quote(source + ':' + path)
            actual = subprocess.check_output(argv + [command], timeout=25)
            if actual != programs[name].encode():
                raise ValueError('Git source on training host differs; deploy the exact commit first')
    backend = GitSupervisor(a.decisions, a.repo, verify_deployment=verify_deployment if argv else None)
    kwargs = dict(box=a.box, backend=backend, workdir=a.workdir, run=a.run_id,
                  poll=1, keep_serving=False, give_up_after=20)
    if a.ssh_argv:
        with existing_ssh(mailbox, argv):
            mailbox.run_broker(ssh='existing-native-host', **kwargs)
    else:
        mailbox.run_broker(**kwargs)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','box','decisions','repo','workdir','run-id'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--ssh-argv')
    p.add_argument('--remote-repo')
    main(p.parse_args())
