"""Rerecord only the diagnosed failed TRAIN teacher scenes in a new full24 batch.

Source scripts/audio/folds stay unchanged. The 13 passing raw recordings remain
byte-identical; all 24 must pass independent quality and semantic review again.
"""
import argparse
import asyncio
import json
import shutil
from pathlib import Path

from evals.moshi_model_download import sha256

PARENT_SHA256 = 'f7ccf6a2cb83651b718d34fc16ca6a175b46ba78f85378809040b0db2aa55add'
CHECKS_SHA256 = '142e286897bc8be0e441d24ff7e44c124f3e1adfdf1c123edc44a19edf3dd35f'
FAILED = frozenset('native2-train-' + x for x in
                   ('001','002','003','011','014','015','019','020','021','022','023'))
CLARITY_PARENT_SHA256 = '2552a0897850f25c131a0a905e05a1aeb964010ca6bdc6515b164945894ce4df'
CLARITY_CHECKS_SHA256 = '97c0730d8eb12a0cf8399ae82acbe2533655b2db83b39495be6d4048c5e32d0a'
CLARITY_FAILED = frozenset('native2-train-' + x for x in ('001','014','015','019','022'))


def check_lineage(current, parent, checks):
    before = {r['dir']: r for r in parent['recordings']}
    after = {r['dir']: r for r in current['recordings']}
    failures = {r['scenario_id'] for r in checks['results'] if not r['speech_and_fact_checks_pass']}
    if (len(before) != 24 or len(after) != 24 or set(before) != set(after)
            or failures != FAILED or current['teacher_protocol'] != parent['teacher_protocol']
            or current['source_ingestion_sha256'] != parent['source_ingestion_sha256']
            or current['source_conversion'] != parent['source_conversion']):
        raise RuntimeError('Teacher repair changed the complete scope, protocol or source')
    if any(after[sid] != before[sid] for sid in before if sid not in FAILED):
        raise RuntimeError('Teacher repair changed an already passing recording')
    if any(after[sid]['source_row'] != before[sid]['source_row'] for sid in FAILED):
        raise RuntimeError('Teacher repair changed source words/audio/fold')


def validate_lineage(manifest):
    info = manifest['diagnosed_repair']
    parent_path = Path(info['parent']) / 'teacher.json'
    checks_path = Path(info['checks'])
    if (sha256(parent_path) != PARENT_SHA256 or info['parent_sha256'] != PARENT_SHA256
            or sha256(checks_path) != CHECKS_SHA256 or info['checks_sha256'] != CHECKS_SHA256
            or info['repaired_ids'] != sorted(FAILED) or info['maximum_new_sessions'] != 11):
        raise RuntimeError('Teacher repair lost its diagnosed original hashes or session bound')
    check_lineage(manifest, json.loads(parent_path.read_text()), json.loads(checks_path.read_text()))


def check_clarity_lineage(current, parent, checks):
    before = {r['dir']: r for r in parent['recordings']}
    after = {r['dir']: r for r in current['recordings']}
    if (len(before) != 24 or len(after) != 24 or set(before) != set(after)
            or {r['scenario_id'] for r in checks['results'] if not r['speech_and_fact_checks_pass']} != CLARITY_FAILED
            or current.get('record_protocols') != {sid: 'native-dialogue-v5' for sid in CLARITY_FAILED}
            or current['teacher_protocol'] != parent['teacher_protocol']
            or current['source_ingestion_sha256'] != parent['source_ingestion_sha256']
            or current['source_conversion'] != parent['source_conversion']):
        raise RuntimeError('Clarity repair lost its full24 diagnosis/source/protocol declaration')
    if any(after[sid] != before[sid] for sid in before if sid not in CLARITY_FAILED):
        raise RuntimeError('Clarity repair changed an independently passing recording')
    if any(after[sid]['source_row'] != before[sid]['source_row']
           or after[sid].get('teacher_protocol') != 'native-dialogue-v5' for sid in CLARITY_FAILED):
        raise RuntimeError('Clarity repair changed source words/audio/folds or its declared protocol')


def validate_clarity_lineage(manifest):
    info = manifest['diagnosed_repair_v2']
    parent_path = Path(info['parent']) / 'teacher.json'
    checks_path = Path(info['checks'])
    if (sha256(parent_path) != CLARITY_PARENT_SHA256 or info['parent_sha256'] != CLARITY_PARENT_SHA256
            or sha256(checks_path) != CLARITY_CHECKS_SHA256 or info['checks_sha256'] != CLARITY_CHECKS_SHA256
            or info['repaired_ids'] != sorted(CLARITY_FAILED) or info['maximum_new_sessions'] != 5):
        raise RuntimeError('Clarity repair lost the actual 19/24 diagnosis')
    check_clarity_lineage(manifest, json.loads(parent_path.read_text()), json.loads(checks_path.read_text()))


async def repair(parent, checks_path, out, clarity=False):
    from evals.recovery_teacher_quality import validate_teacher
    from evals.recovery_teacher import generate
    parent_sha = CLARITY_PARENT_SHA256 if clarity else PARENT_SHA256
    checks_sha = CLARITY_CHECKS_SHA256 if clarity else CHECKS_SHA256
    failed = CLARITY_FAILED if clarity else FAILED
    if sha256(parent / 'teacher.json') != parent_sha or sha256(checks_path) != checks_sha:
        raise RuntimeError('Repair must bind the actually diagnosed failed v4 batch')
    manifest = validate_teacher(parent)
    checks = json.loads(checks_path.read_text())
    if (checks['raw_batch'] != str(parent) or checks['raw_manifest_sha256'] != parent_sha
            or checks['speech_report_sha256'] != sha256(checks_path.parent / 'report.json')
            or {r['scenario_id'] for r in checks['results'] if not r['speech_and_fact_checks_pass']} != failed):
        raise RuntimeError('Teacher repair does not match the actual full24 quality failure')
    shutil.copytree(parent, out, symlinks=True)
    archive = out / ('repair-originals-clarity1' if clarity else 'repair-originals-v4')
    archive.mkdir(exist_ok=False)
    for sid in failed:
        shutil.move(str(out / sid), str(archive / sid))
    shutil.copyfile(parent / 'teacher.json', out / ('teacher.original-clarity1.json' if clarity else 'teacher.original-v4.json'))
    manifest['recordings'] = [r for r in manifest['recordings'] if r['dir'] not in failed]
    manifest['diagnosed_repair_v2' if clarity else 'diagnosed_repair'] = dict(parent=str(parent), parent_sha256=parent_sha,
        checks=str(checks_path), checks_sha256=checks_sha, repaired_ids=sorted(failed),
        maximum_new_sessions=len(failed), passing_recordings_preserved=24-len(failed),
        source_audio_words_folds_unchanged=True)
    overrides = {sid: 'native-dialogue-v5' for sid in failed} if clarity else {}
    if clarity:
        manifest['record_protocols'] = overrides
    (out / 'teacher.json').write_text(json.dumps(manifest, indent=2) + '\n')
    await generate(Path(manifest['source_conversion']), out, resume=True,
                   scope='native-v2', protocol='native-dialogue-v4', record_protocols=overrides)
    validate_teacher(out)
    print(json.dumps(dict(total=24, new_sessions=len(failed), unchanged_passing_recordings=24-len(failed),
                          quality_approved=False, training_approved=False)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'checks', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--clarity-retry', action='store_true')
    a = p.parse_args()
    asyncio.run(repair(a.parent, a.checks, a.out, a.clarity_retry))


if __name__ == '__main__':
    main()
