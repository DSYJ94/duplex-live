"""Owner-approved whole20 derived input quality; no candidate or training approval."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.native_fresh_source_profile_probe import recompute, PROPOSAL_SHA256
from evals.native_fresh_inputs_v2 import load_inputs, specifications, PLAN_SHA256

PROFILE = 'owner-approved-turbo-spelling-v1'
AUTHORIZATION = Path(__file__).with_name('fixtures')/'native-fresh2-source-reapproval-authorization-20261019.json'
AUTHORIZATION_SHA256 = 'd80d3334dd3b7913c1d0309ce6e423a35f3f7101551d7984dd15712c2eaae115'


def authorization():
    if sha256(AUTHORIZATION) != AUTHORIZATION_SHA256:
        raise ValueError('Exact committed owner hearing and whole20 procedure approval required')
    return json.loads(AUTHORIZATION.read_text())


def quality(data, primary, diagnostic):
    authorization()
    evidence = recompute(data, primary, diagnostic)
    passed = evidence['all20_diagnostic_meets_proposed_bars']
    return dict(schema='dl-native-fresh2-derived-quality-v1', profile=PROFILE,
        authorization_sha256=AUTHORIZATION_SHA256, proposal_sha256=PROPOSAL_SHA256,
        source_code_sha256=sha256(Path(__file__)),
        primary_quality_path=str(primary.resolve()), turbo_diagnostic_path=str(diagnostic.resolve()),
        evidence=evidence, passed=passed, input_quality_approved=passed,
        owner_hearing='room sign', owner_whole20_procedure_approved=True,
        threshold=.05, peak_bar=.995, complete_cases=20, complete_clips=38,
        original_failed_verdicts_preserved=True, original_audio_changed=False,
        generation_calls=0, rerolls=0, passing_subset_accepted=False,
        room_rune_equivalent=False, do_not_train=True, training_approved=False,
        inference_approved=False, independent_full10_candidate_review_required=True)


def run(data, primary, diagnostic, out):
    report = quality(data, primary, diagnostic)
    out.mkdir(parents=True, exist_ok=False)
    (out/'quality.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence'}), flush=True)
    if not report['passed']:
        raise RuntimeError('Whole20 derived source quality failed; retain all clips frozen/unscored')


def approved_inputs(data, quality_path):
    saved = json.loads(quality_path.read_text())
    if saved.get('schema') != 'dl-native-fresh2-derived-quality-v1' or saved.get('profile') != PROFILE:
        raise ValueError('Explicit new derived whole20 quality version required')
    expected = quality(data, Path(saved['primary_quality_path']), Path(saved['turbo_diagnostic_path']))
    if saved != expected or not expected['passed']:
        raise ValueError('Derived approval does not reproduce complete20/38 raw word/time/waveform checks')
    return load_inputs(data)


def inference_input(data, quality_path, identity):
    manifest = approved_inputs(data, quality_path)
    case = next(c for c in specifications() if c['id']==identity)
    row = next(r for r in manifest['rows'] if r['scenario_id']==identity)
    return dict(scenario=identity, path=row['path'], wav_sha256=row['wav_sha256'], kind=case['kind'],
        backend=case['backend'], worker_only_world=case['world'],
        minimum_lookup_delay_s=case.get('minimum_actual_lookup_delay_s',0.),
        manifest_sha256=sha256(data/'inputs.json'), quality_sha256=sha256(quality_path),
        plan_sha256=PLAN_SHA256, source_split='fresh-evaluation', training_approved=False,
        source_quality_profile=PROFILE, source_quality_authorization_sha256=AUTHORIZATION_SHA256)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify-only',action='store_true')
    for name in ('data','primary','diagnostic','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.verify_only:
        m=approved_inputs(a.data,a.out/'quality.json')
        report=dict(schema='dl-native-fresh2-derived-quality-reader-v1',passed=True,
            fresh_process=True,complete_cases=len(m['rows']),complete_clips=sum(r['utterance_count'] for r in m['rows']),
            quality_sha256=sha256(a.out/'quality.json'),authorization_sha256=AUTHORIZATION_SHA256,
            training_approved=False,inference_approved=False)
        with (a.out/'reader-verification.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(json.dumps(report),flush=True)
    else:run(a.data,a.primary,a.diagnostic,a.out)


if __name__=='__main__':main()
