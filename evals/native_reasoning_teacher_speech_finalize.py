"""Complete the unchanged whole32 speech audit without repeated source work.

This CPU-only runner reproduces the frozen FAILED baseline, computes ALL new
verdicts and independently reads them back. Approval is never inferred from its
process exit: the original whole-corpus speech verdict remains authoritative.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from evals import native_reasoning_teacher_quality as original
from evals import native_reasoning_teacher_speech_v2 as audit
from evals.native_reasoning_verification_session import inventory, source_readbacks


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only source/speech readback required')
    if a.receipt.exists(): raise FileExistsError('Never overwrite a verification receipt')
    args = (a.source, a.quality, a.batch, a.engine, a.tokenizer, a.failed)
    baseline = json.loads(a.failed.read_text())
    recognition = json.loads((a.worker / 'recognition.json').read_text())
    if (baseline['complete_cases'] != 32 or baseline['passed'] is not False
            or recognition['complete_cases'] != 32):
        raise ValueError('Complete frozen failure and whole32 recognition required')
    roots = [a.source, a.quality.parent, a.batch, a.engine, a.tokenizer,
             a.failed.parent, a.plan, a.worker / 'recognition.json']
    roots += [a.worker / row['asr_file'] for row in recognition['recordings']]
    state = dict(schema='dl-native-reasoning32-verification-session-result-v1',
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 runner_sha256=audit.sha(Path(__file__)),
                 phase='reproducing_original_failure_and_all32_verdicts',
                 training_approved=False, semantic_approved=False,
                 recognition_restarted=False, provider_requests=0, optimizer_updates=0)
    began = time.monotonic()
    def save():
        state['at_utc'] = datetime.now(timezone.utc).isoformat()
        audit.write(a.receipt, state)
    save()
    try:
        with source_readbacks(roots) as memo:
            reproduced = original.verify(*args, require_pass=False)
            if reproduced != baseline: raise ValueError('Original failed audit differs')
            target = a.worker / 'teacher-speech-quality.json'
            if target.exists():
                value = json.loads(target.read_text())
            else:
                value = audit.finalize(*args, a.plan, a.worker)
            verified = audit.verify(*args, a.plan, a.worker, require_pass=False)
            if value != verified: raise ValueError('Independent whole32 readback differs')
        rows = inventory(roots)
        state.update(phase='complete_all32_speech_only',
                     original_failed32_reproduced_exactly=True,
                     unchanged_input_snapshot=True,
                     snapshot_sha256=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
                     snapshot_members=len(rows), original_failed_audit_sha256=audit.sha(a.failed),
                     complete_cases=32, passed_cases=sum(r['passed'] for r in value['results']),
                     whole_speech_passed=value['passed'],
                     whole_speech_result_sha256=audit.sha(target),
                     source_checker_executions=memo.calls, duplicate_source_reads_reused=memo.hits,
                     elapsed_seconds=time.monotonic()-began)
        save(); return state
    except BaseException as error:
        state.update(phase='failed_preserved_no_retry',error_type=type(error).__name__,
                     reason=str(error),elapsed_seconds=time.monotonic()-began)
        save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','quality','batch','engine','tokenizer','failed','plan','worker','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(parser.parse_args())))
