"""Download a public pinned model asset in bounded ranges, then verify its SHA256.

No credentials or fallback source. Only exact byte ranges and the caller's pinned
size/hash are accepted. An incomplete file never replaces the destination asset.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bytes', type=int, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    assert args.url.startswith('https://huggingface.co/')
    assert args.bytes > 0 and 1 <= args.workers <= 16
    assert len(args.sha256) == 64
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + '.public-download-v1')
    fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    lock = threading.Lock()
    done = 0
    started = time.monotonic()
    try:
        os.ftruncate(fd, args.bytes)

        def chunk(start):
            nonlocal done
            end = min(start + 32 * 1024 * 1024, args.bytes) - 1
            for attempt in range(3):
                try:
                    request = urllib.request.Request(args.url,
                        headers={'Range': 'bytes=%d-%d' % (start, end)})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if (response.status != 206 or response.headers.get('Content-Range') !=
                                'bytes %d-%d/%d' % (start, end, args.bytes)):
                            raise RuntimeError('Server did not honor the exact pinned range')
                        data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError('Incomplete pinned byte range')
                    offset = 0
                    while offset < len(data):
                        written = os.pwrite(fd, data[offset:], start + offset)
                        if written <= 0:
                            raise RuntimeError('Incomplete local range write')
                        offset += written
                    with lock:
                        old = done
                        done += len(data)
                        if done // (1024**3) > old // (1024**3) or done == args.bytes:
                            print(json.dumps({'bytes_downloaded': done,
                                'total_bytes': args.bytes,
                                'seconds': round(time.monotonic()-started, 2)}), flush=True)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(chunk, range(0, args.bytes, 32 * 1024 * 1024)))
        os.fsync(fd)
    finally:
        os.close(fd)
    h = hashlib.sha256()
    with temporary.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    if temporary.stat().st_size != args.bytes or h.hexdigest() != args.sha256:
        raise RuntimeError('Pinned asset size/SHA256 mismatch; destination not replaced')
    temporary.replace(args.out)
    print(json.dumps({'passed': True, 'bytes': args.bytes, 'sha256': h.hexdigest(),
                      'seconds': round(time.monotonic()-started, 2)}), flush=True)


if __name__ == '__main__':
    main()
