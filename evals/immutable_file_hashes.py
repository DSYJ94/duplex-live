"""One full hash per immutable input per process; reject any identity change.

Only explicit readonly roots are eligible. Generated outputs and every default
caller retain uncached hashing. This caches byte digests, never validation results,
teacher decisions, tensors, model outputs, gradients or optimizer state.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import time

_current=ContextVar('native_immutable_hash_session',default=None)


def active():return _current.get()


def identity(path):
    p=Path(path).resolve(strict=True);s=p.stat()
    return str(p),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


class ImmutableHashes:
    def __init__(self,roots):
        self.roots=tuple(Path(r).resolve(strict=True) for r in roots)
        self.entries={};self.hits=0;self.bytes_hashed=0;self.seconds_hashed=0.;self.verified=False

    def digest(self,path,uncached):
        p=Path(path);now=identity(p)
        key=str(p.absolute())
        if key in self.entries:
            old,value=self.entries[key]
            if old!=now:raise ValueError('Previously hashed immutable input changed: '+key)
            self.hits+=1;return value
        if not any(Path(now[0]).is_relative_to(root) for root in self.roots):return uncached(p)
        began=time.monotonic();value=uncached(p)
        if identity(p)!=now:raise ValueError('Immutable input changed during its full hash: '+key)
        self.seconds_hashed+=time.monotonic()-began;self.bytes_hashed+=now[3]
        self.entries[key]=(now,value);return value

    def verify(self):
        for path,(version,_) in self.entries.items():
            if identity(path)!=version:raise ValueError('Immutable input changed before stage completion: '+path)
        self.verified=True

    def receipt(self):
        return dict(mode='full-first-read-plus-file-identity-every-reuse-v1',
                    roots=list(map(str,self.roots)),unique_hashed_files=len(self.entries),
                    reused_hashes=self.hits,bytes_fully_hashed=self.bytes_hashed,
                    full_hash_seconds=self.seconds_hashed,all_cached_identities_rechecked=self.verified,
                    validation_results_or_model_outputs_cached=False)


@contextmanager
def verified_reads(roots):
    if active() is not None:raise ValueError('Do not nest immutable-input verification sessions')
    cache=ImmutableHashes(roots);token=_current.set(cache)
    try:
        yield cache
        cache.verify()
    finally:_current.reset(token)
