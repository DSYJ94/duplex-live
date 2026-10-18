"""Independent diagnostic ASR; never replace primary quality or model gates.

All eight synthetic user clips and three complete diagnostic output clips are
checked once, regardless of the first recognizer's result. References are used
only after transcription. No owner recordings are read. API reference:
https://developers.openai.com/api/docs/guides/speech-to-text
"""
import argparse,json,os,time
from pathlib import Path
from evals.moshi_model_download import sha256

MODEL='gpt-4o-transcribe'

def inventory(runs):
    manifest=runs/'duplex-fresh-inputs-7518886/inputs.json';m=json.loads(manifest.read_text());quality=runs/'duplex-fresh-quality-7518887/quality.json';q=json.loads(quality.read_text())
    assert m['do_not_train'] and not m['training_approved'] and q['inputs_manifest_sha256']==sha256(manifest)
    assert len(m['rows'])==4 and {r['scenario_id'] for r in m['rows']}=={'hybrid-fresh-042','hybrid-fresh-043','hybrid-fresh-044','hybrid-fresh-045'}
    clips=[]
    for row in m['rows']:
        folder=Path(row['path']).parent;reference=json.loads(Path(row['reference_path']).read_text())
        for i,u in enumerate(reference['utterances']):
            path=folder/u['source_file'];receipt=row['generation_receipts'][i];assert path.name==receipt['file'] and sha256(path)==receipt['sha256']
            clips.append(dict(id=row['scenario_id']+'-user-'+str(i),path=path,expected=u['text'],kind='synthetic-user',primary_quality_pass=next(r['passed'] for r in q['results'] if r['scenario_id']==row['scenario_id'])))
    for job in (7518838,7518889,7518891):
        folder=runs/('duplex-live-transport-audit-'+str(job));report=json.loads((folder/'report.json').read_text())
        completed=[r for r in report['results'] if r['fully_played']];assert len(completed)==1
        row=completed[0];path=folder/row['scenario']/(row['event_id']+'-played.wav');assert sha256(path)==row['played_audio_sha256']
        clips.append(dict(id=str(job)+'-'+row['event_id'],path=path,expected=row['planned_text'],kind='synthetic-model-output',primary_wer=row['word_error_rate'],primary_timestamp_bounds_valid=row['valid_timestamp_bounds']))
    assert len(clips)==11
    return clips


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import requests
    from data.synthetic.filter import norm_words,wer
    clips=inventory(a.runs)
    results=[]
    for clip in clips:
        path=clip['path'];row={k:v for k,v in clip.items() if k not in ('path','expected')};row.update(path=str(path),wav_sha256=sha256(path));start=time.monotonic()
        try:
            with path.open('rb') as f:
                response=requests.post('https://api.openai.com/v1/audio/transcriptions',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},data=dict(model=MODEL,language='en',response_format='json'),files={'file':(path.name,f,'audio/wav')},timeout=120)
            row.update(http_status=response.status_code,request_id=response.headers.get('x-request-id'))
            if response.status_code!=200:raise RuntimeError('Independent transcription HTTP '+str(response.status_code))
            data=response.json();(a.out/(clip['id']+'-response.json')).write_text(json.dumps(data,indent=2)+'\n')
            text=data['text'];assert isinstance(text,str) and text.strip()
            row.update(completed=True,transcript=text,response_sha256=sha256(a.out/(clip['id']+'-response.json')))
        except Exception as error:row.update(completed=False,error_type=type(error).__name__,message=str(error))
        # Ground truth never enters the actual recognizer request.
        row.update(expected=clip['expected'],seconds=time.monotonic()-start)
        if row['completed']:row['diagnostic_wer']=wer(norm_words(clip['expected']),norm_words(row['transcript']))
        results.append(row);(a.out/(clip['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    report=dict(schema='dl-secondary-asr-diagnostic-v1',model=MODEL,hosted_model_alias_not_local_weight_pin=True,results=results,actual_requests=len(results),component_execution_completed=all(r['completed'] for r in results),all_predeclared_clips_retained=True,reference_supplied_to_recognizer=False,primary_gates_unchanged=True,training_approved=False,qualification_replacement=False,physical_device_acceptance=False,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(rows=len(results),completed=report['component_execution_completed'])),flush=True)
    if not report['component_execution_completed']:raise SystemExit(1)

if __name__=='__main__':main()
