"""Independently transcribe every immutable TRAIN-only S3 round trip."""
import argparse
import json
import sys
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    manifest=json.loads((args.source/'manifest.json').read_text())
    if manifest['schema']!='dl-lychee-s3-source-diagnostic-v1' or manifest['training_approved']:
        raise RuntimeError('Expected unapproved codec diagnostics')
    folders=[]
    for row in manifest['examples']:
        folder=Path(row['path']);r=json.loads((folder/'report.json').read_text())
        if (row['source_split']!='train' or r['source_split']!='train'
                or sha256(folder/'report.json')!=row['report_sha256']
                or sha256(folder/'model.wav')!=r['audio_sha256']
                or sha256(folder/'codes.pt')!=r['codes_sha256']):
            raise RuntimeError('TRAIN-only codec provenance changed')
        folders.append(str(folder))
    if not folders:raise RuntimeError('No native codec diagnostics')
    from evals.moshi_speech_audit import main as audit
    old=sys.argv
    try:
        sys.argv=[old[0],'--runs',*folders,'--out',str(args.out),'--trim-trailing-silence','--keep-silent-failures']
        audit()
    finally:sys.argv=old
    report=json.loads((args.out/'report.json').read_text())
    passed=[];failed=[]
    for row in report['results']:
        valid=(not row.get('audit_failure') and row['asr_word_bounds_valid'] and row['asr_time_bounds_valid']
            and row['wer_against_model_text']<=.08)
        (passed if valid else failed).append({'run':row['run'],'wer':row['wer_against_model_text']})
    checks={'schema':'dl-lychee-s3-audit-v1','source_manifest_sha256':sha256(args.source/'manifest.json'),
        'speech_report_sha256':sha256(args.out/'report.json'),'passed':passed,'failed':failed,
        'wer_bar':.08,'training_approved':False,
        'qualification':'codec intelligibility only; causal native input assembly and training proof still required'}
    (args.out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({'codec_segments_passed':len(passed),'codec_segments_failed':len(failed),'training_approved':False}),flush=True)


if __name__=='__main__':main()
