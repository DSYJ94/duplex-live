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
        self.failure_receipt = None
        self.last_local = None
        self.provider_query_timed_out = False

    def tick(self):
        if self.failed:
            raise RuntimeError('Heartbeat has failed; no implicit restart')
        try:
            return self._tick()
        except Exception as error:
            self.failed = True
            checked = self.clock()
            def age(record):
                stamp = record.get('observed_unix') if isinstance(record, dict) else None
                return checked-stamp if contract.finite_number(stamp) else None
            provider_age, local_age = age(self.provider), age(self.last_local)
            failed = 'other'
            for label, value, ttl in (('provider', provider_age, contract.PROVIDER_TTL),
                                      ('local', local_age, contract.LOCAL_TTL)):
                if value is None or not 0 <= value <= ttl:
                    failed = label
                    break
            self.failure_receipt = dict(checked_unix=checked, failed_evidence=failed,
                provider_age_seconds=provider_age, local_age_seconds=local_age,
                provider_query_timed_out=self.provider_query_timed_out,
                provider_attempt_monotonic=self.last_attempt,
                error_type=type(error).__name__, error=str(error),
                guard_renewed=False, implicit_restart=False)
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
            self.provider_query_timed_out = False
            try:
                observed = self.read_provider()
            except subprocess.TimeoutExpired:
                if self.provider is None:
                    raise
                timeout = True
                self.provider_query_timed_out = True
            else:
                # Never fall back to an old RUNNING receipt after cancellation,
                # overlapping-job evidence, malformed data or identity change.
                self.provider = observed
        local = self.read_local()
        self.last_local = local
        checked = self.clock()  # query latency counts toward actual evidence age
        proof = contract.validate(self.lease, self.provider, local, checked)
        proof.update(provider_attempt_monotonic=self.last_attempt,
                     provider_query_timed_out=timeout,
                     expires_unix=min(local['observed_unix'] + contract.LOCAL_TTL,
                                       self.provider['observed_unix'] + contract.PROVIDER_TTL,
                                       self.lease['stop_unix'] - 180))
        return proof
