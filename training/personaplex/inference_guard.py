"""Fresh frame checks without spawning git on the streaming critical path.

Admission validates the unchanged resource contract and clean source once.
Each frame still reads the actual detached HEAD, complete lease and current
heartbeat; job/account/identity/deadline/freshness checks remain identical.
"""
import json,os,subprocess,time
from pathlib import Path
from .expanded_guard import resource,ensure


class FrameGuard:
    def __init__(self,lease_path):
        self.path=Path(lease_path);self.raw=self.path.read_bytes()
        self.lease=resource(json.loads(self.raw))
        self.head=Path(subprocess.check_output(['git','rev-parse','--git-path','HEAD'],universal_newlines=True).strip())
        if not self.head.is_absolute():self.head=Path.cwd()/self.head
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],universal_newlines=True).strip():raise ValueError('Clean immutable inference source required')
        if self.head.read_text().strip()!=self.lease['source']:raise ValueError('Exact detached admitted inference HEAD required')
        self.guard=Path(os.environ['DUPLEX_LIVE_GUARD_FILE'])

    def __call__(self,needed=0):
        if self.path.read_bytes()!=self.raw or self.head.read_text().strip()!=self.lease['source']:
            raise RuntimeError('Immutable inference lease/source changed')
        # Keep the original resource+freshness checks. The git subprocess alone
        # is eliminated; no deadline, heartbeat, source or scope gate is cached.
        ensure(self.lease,json.loads(self.guard.read_text()),needed)
        return self.lease
