"""Proposed composed-speech RP64 evaluation, no training or alternate score path."""
import asyncio
import json
from pathlib import Path
import subprocess
import time

from . import british_speech_guard as guard


def admit(args):
    lease=json.loads(Path(args.lease).read_text())
    guard.resource(lease)
    plan=json.loads(guard.PLAN.read_text())
    actual=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if args.source!=actual or plan.get('execution_source')!=actual:
        raise ValueError('Actual Git producer must match new evaluation source')
    if args.voice_reference_sha256!=plan['reference_sha256'] or guard.sha(args.voice_reference)!=plan['reference_sha256']:
        raise ValueError('Exact preselected RP native reference required')
    return lease


def guarded(lease_path,needed=0):
    from portable.venus.native_app import ensure_guard
    lease=json.loads(Path(lease_path).read_text());guard.resource(lease);ensure_guard()
    if time.time()+needed>=lease['stop_unix']-180:
        raise TimeoutError('Preserve full64 speech/copyback before native STOP')


def main():
    from .large_behavior import parser,collect,combine
    args=parser().parse_args()
    admit(args)
    if args.combine:combine(args)
    else:asyncio.run(collect(args,british_composition=True))


if __name__=='__main__':main()
