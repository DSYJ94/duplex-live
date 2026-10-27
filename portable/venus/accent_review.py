"""Complete source-bound accent listening page, with matched scripts per voice."""
import argparse,array,hashlib,html,json,shutil,wave
from pathlib import Path

PROFILES=[('rp-ben','Received Pronunciation · Ben'),('british-alice','British English · Alice'),('british-george','British English · George'),('northern-mark','Northern English · Mark'),('welsh-hannah','Welsh · Hannah'),('scottish-edinburgh','Scottish · Edinburgh'),('scottish-glasgow','Scottish · Glasgow'),('irish-rory','Irish English · Rory'),('american-rachel','American English · Rachel')]
MATCHED=['voice-v1-warm_conversation-01','voice-v1-calm_pacing-05','voice-v1-curious_questions-06','voice-v1-reference']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(base,output):
    base,output=Path(base),Path(output);plan=json.loads(Path('evals/fixtures/venus-voice-quality-source-v1-20261027.json').read_text());cases={r['id']:r for r in plan['cases']};cases['voice-v1-reference']=dict(assistant_text=plan['reference_text'],split='REFERENCE_ONLY',user_context='Neutral reference passage')
    roots=[('voice-eleven-v3-sources-v3-final',67),('british-accent-sources-v2-repaired',95),('british-accent-supplement-v2-repaired',12)];rows=[];reports=[]
    output.mkdir(exist_ok=False,parents=True);media=output/'media';media.mkdir()
    for name,count in roots:
        root=base/name;receipt=root/'render-receipts.json';r=json.loads(receipt.read_text());assert r['complete'] and len(r['utterances'])==count
        reports.append(dict(name=name,sha256=sha(receipt)))
        for source in r['utterances']:
            profile=source.get('profile','american-rachel');ident=source['id'];source_case=source.get('source_case_id',ident);case=cases[source_case]
            assert source['text']==case['assistant_text'] and source['split']==case['split']
            wav=root/ident/'source.wav';raw=(root/ident/'source.pcm').read_bytes();assert sha(wav)==source['wav_sha256'] and hashlib.sha256(raw).hexdigest()==source['raw_pcm_sha256']
            with wave.open(str(wav)) as w:assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(24000,1,2) and w.readframes(w.getnframes())==raw
            peak=max(abs(x) for x in array.array('h',raw))/32768;assert peak==source['peak'] and source['peak_pass']==(peak<=.995)
            filename=profile+'--'+source_case+'.wav';shutil.copy2(wav,media/filename)
            rows.append(dict(profile=profile,id=ident,source_case_id=source_case,text=source['text'],context=case.get('user_context',''),split=source['split'],voice_id=source['voice_id'],model_id=source['model_id'],source_wav_sha256=source['wav_sha256'],peak=peak,peak_pass=source['peak_pass'],filename=filename))
    assert len(rows)==len({r['filename'] for r in rows})==174 and {r['profile'] for r in rows}=={p for p,_ in PROFILES}
    index={(r['profile'],r['source_case_id']):r for r in rows}
    assert all((p,c) in index for p,_ in PROFILES for c in MATCHED)
    def player(r,rating=False):
        return '<div class="clip"><audio controls preload="none" src="media/'+r['filename']+'"></audio><span class="check">'+('Peak check passed' if r['peak_pass'] else 'FAILED peak check · retained')+'</span>'+('<label>Naturalness <select data-profile="'+r['profile']+'" data-case="'+r['source_case_id']+'"><option value="">Rate</option>'+''.join('<option>'+str(i)+'</option>' for i in range(1,6))+'</select></label>' if rating else '')+'</div>'
    sections=[]
    for profile,label in PROFILES:
        clips=[]
        for c in MATCHED:
            r=index[(profile,c)];clips.append('<div class="sample"><p class="context">'+html.escape(r['context'])+'</p><p>'+html.escape(r['text'])+'</p>'+player(r,True)+'</div>')
        sections.append('<section id="'+profile+'"><h2>'+html.escape(label)+'</h2>'+''.join(clips)+'</section>')
    full=[]
    for profile,label in PROFILES:
        selected=[r for r in rows if r['profile']==profile]
        full.append('<details><summary>'+html.escape(label)+' · '+str(len(selected))+' recordings</summary>'+''.join('<p>'+html.escape(r['text'])+' <small>'+html.escape(r['split'])+'</small></p>'+player(r) for r in selected)+'</details>')
    binding=hashlib.sha256(json.dumps(reports,sort_keys=True).encode()).hexdigest()
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>British voices and regional accents</title><style>body{font:17px/1.55 system-ui;background:#f5f5f1;color:#192624;margin:0}main{max-width:950px;margin:40px auto;padding:0 24px}h1{font-size:36px;line-height:1.15}h2{font-size:24px}section,details{background:#fff;border:1px solid #d9dfda;border-radius:16px;padding:22px;margin:22px 0}.sample{border-top:1px solid #edf0ed;padding:10px 0}.context,small,.check{color:#59685f;font-size:14px}.clip{display:flex;align-items:center;gap:12px;flex-wrap:wrap}audio{width:390px;max-width:100%}select,button{font:inherit;padding:8px;border:1px solid #a3b4ac;border-radius:8px;background:white}button{background:#245f4d;color:white;cursor:pointer}summary{cursor:pointer;font-weight:600}.note{padding:14px;background:#edf3ed;border-radius:12px}</style><main><h1>British voices and regional accents</h1><p class="note">Generated voice references for the next training stage. Native model accent transfer and independent word checks are pending. No voice has been adopted.</p><p>RP, British English, Northern English, Welsh, Edinburgh and Glasgow Scottish, Irish English and the original American reference. Each voice reads the same three sample scripts and neutral passage. All174 recordings are retained below.</p><label>Preferred direction <select id="preferred"><option value="">Choose a voice</option>'''+''.join('<option value="'+p+'">'+html.escape(label)+'</option>' for p,label in PROFILES)+'''</select></label> <button id="export">Save preferences</button><p>Naturalness:1=artificial,3=noticeably synthetic,5=like a person speaking naturally. Preferences stay in this browser until exported.</p>'''+''.join(sections)+'''<h2>Complete source collection</h2><p>All scripts, folds and original audio are included. Six silent targets have no generated audio. Selecting a preference does not change the running model.</p>'''+''.join(full)+'''</main><script>const sourceBinding='BINDING';const key='duplex-accent-preferences-'+sourceBinding;let data={ratings:{},preferred:''};try{data=JSON.parse(localStorage.getItem(key)||JSON.stringify(data))}catch(e){}const save=()=>{try{localStorage.setItem(key,JSON.stringify(data))}catch(e){}};const pref=document.getElementById('preferred');pref.value=data.preferred||'';pref.onchange=()=>{data.preferred=pref.value;save()};for(const el of document.querySelectorAll('select[data-profile]')){const k=el.dataset.profile+'|'+el.dataset.case;el.value=data.ratings[k]||'';el.onchange=()=>{data.ratings[k]=el.value;save()}}document.getElementById('export').onclick=()=>{const payload={schema:'dl-accent-listening-v1',source_binding:sourceBinding,at:new Date().toISOString(),...data,training_approved:false};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='accent-listening-preferences.json';a.click();URL.revokeObjectURL(url)};</script></html>'''
    (output/'index.html').write_text(page.replace('BINDING',binding))
    report=dict(recordings=174,profiles=[dict(id=p,label=l) for p,l in PROFILES],matched_scripts_per_profile=4,source_binding=binding,source_receipts=reports,all_original_hashes_verified=True,all_original_peaks_pass=all(r['peak_pass'] for r in rows),entries=rows,independent_words='PENDING_GPU_AUDIT',naturalness='PENDING_LISTENING',training_approved=False)
    (output/'source-index.json').write_text(json.dumps(report,indent=2)+'\n');return {k:v for k,v in report.items() if k not in ('profiles','entries','source_receipts')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.base,a.output)))
