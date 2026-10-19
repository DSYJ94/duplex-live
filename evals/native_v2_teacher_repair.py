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


async def repair(parent, checks_path, out):
    from evals.recovery_teacher_quality import validate_teacher
    from evals.recovery_teacher import generate
    if sha256(parent / 'teacher.json') != PARENT_SHA256 or sha256(checks_path) != CHECKS_SHA256:
        raise RuntimeError('Repair must bind the actually diagnosed failed v4 batch')
    manifest = validate_teacher(parent)
    checks = json.loads(checks_path.read_text())
    if (checks['raw_batch'] != str(parent) or checks['raw_manifest_sha256'] != PARENT_SHA256
            or checks['speech_report_sha256'] != sha256(checks_path.parent / 'report.json')
            or {r['scenario_id'] for r in checks['results'] if not r['speech_and_fact_checks_pass']} != FAILED):
        raise RuntimeError('Teacher repair does not match the actual full24 quality failure')
    shutil.copytree(parent, out, symlinks=True)
    archive = out / 'repair-originals-v4'
    archive.mkdir(exist_ok=False)
    for sid in FAILED:
        shutil.move(str(out / sid), str(archive / sid))
    shutil.copyfile(parent / 'teacher.json', out / 'teacher.original-v4.json')
    manifest['recordings'] = [r for r in manifest['recordings'] if r['dir'] not in FAILED]
    manifest['diagnosed_repair'] = dict(parent=str(parent), parent_sha256=PARENT_SHA256,
        checks=str(checks_path), checks_sha256=CHECKS_SHA256, repaired_ids=sorted(FAILED),
        maximum_new_sessions=11, passing_recordings_preserved=13,
        source_audio_words_folds_unchanged=True)
    (out / 'teacher.json').write_text(json.dumps(manifest, indent=2) + '\n')
    await generate(Path(manifest['source_conversion']), out, resume=True,
                   scope='native-v2', protocol='native-dialogue-v4')
    validate_teacher(out)
    print(json.dumps(dict(total=24, new_sessions=11, unchanged_passing_recordings=13,
                          quality_approved=False, training_approved=False)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'checks', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    asyncio.run(repair(a.parent, a.checks, a.out))


if __name__ == '__main__':
    main()
