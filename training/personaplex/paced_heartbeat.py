"""Proposed two-clock heartbeat engine, not wired to a live launcher.

Callbacks must supply actual validated provider and local process/storage
observations. This module cannot submit jobs or write a native guard. Runtime
integration and the explicit provider-TTL amendment remain required.
"""
import subprocess
import time
from . import paced_guard_contract as contract


class Heartbeat:
    def __init__(self, lease, read_provider, read_local,
                 clock=time.time, monotonic=time.monotonic):
        self.lease = dict(lease)
        self.read_provider = read_provider
        self.read_local = read_local
        self.clock = clock
        self.monotonic = monotonic
        self.last_attempt = None
        self.last_tick = None
        self.provider = None
        self.failed = False

    def tick(self):
        if self.failed:
            raise RuntimeError('Heartbeat has failed; no implicit restart')
        try:
            return self._tick()
        except Exception:
            self.failed = True
            raise

    def _tick(self):
        now = self.monotonic()
        if (not contract.finite_number(now) or now < 0 or
                (self.last_tick is not None and now < self.last_tick)):
            raise RuntimeError('Invalid or reversed monotonic clock')
        self.last_tick = now
        timeout = False
        if contract.provider_due(now, self.last_attempt):
            # Set before attempting I/O: failed calls also consume the spacing.
            self.last_attempt = now
            try:
                observed = self.read_provider()
            except subprocess.TimeoutExpired:
                if self.provider is None:
                    raise
                timeout = True
            else:
                # Never fall back to an old RUNNING receipt after cancellation,
                # overlapping-job evidence, malformed data or identity change.
                self.provider = observed
        local = self.read_local()
        checked = self.clock()  # query latency counts toward actual evidence age
        proof = contract.validate(self.lease, self.provider, local, checked)
        proof.update(provider_attempt_monotonic=self.last_attempt,
                     provider_query_timed_out=timeout,
                     expires_unix=min(local['observed_unix'] + contract.LOCAL_TTL,
                                       self.provider['observed_unix'] + contract.PROVIDER_TTL,
                                       self.lease['stop_unix'] - 180))
        return proof
