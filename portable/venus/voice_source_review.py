"""Serve only complete synthetic voice references, never credentials or owner audio."""
import argparse,hashlib,html,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(plan_path,source,output):
    plan_path,source,output=Path(plan_path),Path(source),Path(output);plan=json.loads(plan_path.read_text());path=source/'render-receipts.json';report=json.loads(path.read_text());rows={r['id']:r for r in report['utterances']}
    assert report['complete'] and report['plan_sha256']==sha(plan_path) and len(rows)==67
    assert set(rows)=={r['id'] for r in plan['cases'] if r['assistant_text']}|{'voice-v1-reference'}
    output.mkdir(exist_ok=False);media=output/'media';media.mkdir();sections=[];entries=[]
    cases=plan['cases']+[dict(id='voice-v1-reference',split='REFERENCE_ONLY',family='neutral_reference',user_context='Native conditioning reference',assistant_text=plan['reference_text'])]
    for case in cases:
        ident=case['id'];text=case['assistant_text'];info=''
        if text:
            row=rows[ident];assert row['text']==text and row['split']==case['split'];wav=source/ident/'source.wav';assert sha(wav)==row['wav_sha256']
            name=ident+'.wav';shutil.copy2(wav,media/name);info='<audio controls preload="none" src="media/'+name+'"></audio><p>Original peak check: '+('pass' if row['peak_pass'] else 'FAIL')+'</p>'
            entries.append(dict(id=ident,source_wav_sha256=row['wav_sha256'],split=case['split']))
        else:info='<p class="quiet">Silent target. No speech was requested or generated.</p>'
        rating=''.join('<label>'+label+' <select data-case="'+ident+'" data-metric="'+metric+'"><option value="">Rate</option>'+''.join('<option>'+str(i)+'</option>' for i in range(1,6))+'</select></label>' for metric,label in (('naturalness','Naturalness'),('clarity','Clarity'),('delivery','Delivery fits the context'))) if text else ''
        sections.append('<section id="'+ident+'"><span class="badge">'+html.escape(case['split'])+' · '+html.escape(case['family'].replace('_',' '))+'</span><p class="context">'+html.escape(case.get('user_context',''))+'</p><h2>'+html.escape(text or 'Stay silent')+'</h2>'+info+'<div class="ratings">'+rating+'</div></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Natural voice references</title><style>body{font:17px/1.55 system-ui;background:#f5f5f1;color:#192624;margin:0}main{max-width:900px;margin:48px auto;padding:0 24px}h1{font-size:36px;line-height:1.2}h2{font-size:21px}section{background:#fff;border:1px solid #d9dfda;border-radius:16px;padding:22px;margin:22px 0}.badge{font-size:13px;color:#36594a;background:#edf4ed;padding:6px 10px;border-radius:20px}.context{color:#53635e}.quiet{color:#36594a}audio{width:100%;max-width:560px}.ratings{display:flex;gap:16px;flex-wrap:wrap}select,button{font:inherit;padding:8px;border:1px solid #a3b4ac;border-radius:8px;background:white}button{background:#245f4d;color:#fff;cursor:pointer}</style><main><h1>Natural voice references</h1><p>ElevenLabs synthetic source recordings for the voice curriculum. All66 spoken scripts and the neutral reference are included, alongside six silent targets. Original audio is unchanged. Independent word checks and complete listening review are still required before training use.</p><p>Rate naturalness from1(artificial) to5(like a person), clarity from1(hard to understand) to5(effortless), and delivery from1(inappropriate) to5(fits the context). Ratings stay in this browser until exported.</p><button id="export">Save listening ratings</button>'''+''.join(sections)+'''</main><script>const storageKey='duplex-voice-source-listening-v1';let ratings={};try{ratings=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}for(const el of document.querySelectorAll('select')){const key=el.dataset.case+'|'+el.dataset.metric;el.value=ratings[key]||'';el.onchange=()=>{ratings[key]=el.value;try{localStorage.setItem(storageKey,JSON.stringify(ratings))}catch(e){}}}document.getElementById('export').onclick=()=>{const payload={schema:'dl-voice-source-ratings-v1',at:new Date().toISOString(),ratings,training_approved:false};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='voice-source-listening-ratings.json';a.click();URL.revokeObjectURL(url)};</script></html>'''
    page=page.replace("const storageKey='duplex-voice-source-listening-v1'",'const storageKey='+json.dumps('duplex-voice-source-listening-'+sha(path)[:16]))
    page=page.replace("schema:'dl-voice-source-ratings-v1',","schema:'dl-voice-source-ratings-v1',source_receipts_sha256:'"+sha(path)+"',plan_sha256:'"+sha(plan_path)+"',")
    (output/'index.html').write_text(page)
    summary=dict(source_receipts_sha256=sha(path),plan_sha256=sha(plan_path),recordings=len(entries),silent_targets=6,original_audio_unchanged=True,entries=entries,training_approved=False)
    (output/'source-index.json').write_text(json.dumps(summary,indent=2)+'\n');return {k:v for k,v in summary.items() if k!='entries'}

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('plan','source','output')];a=p.parse_args();print(json.dumps(build(a.plan,a.source,a.output)))
