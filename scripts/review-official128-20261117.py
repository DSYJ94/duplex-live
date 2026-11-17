"""Preserved complete generated-text review; no inference, ASR or API calls."""
import hashlib,json
from pathlib import Path
ROOT=Path('runs/container-startup-20261115')
gold=json.loads((ROOT/'evaluation8511770-reports/scoring-only-reference.json').read_text())
reports={voice:json.loads((ROOT/'official8526401-reports'/(voice+'.json')).read_text())for voice in ('NATF2','RPBen-normalized')}
review=[];scores={}
for voice,report in reports.items():
    assert report['complete']and len(report['cases'])==64 and report['source']=='f926661581d09d123f04b4e83f699e580cd804c4'and report['optimizer_updates']==0
    final={50,55,63}if voice=='NATF2'else{1,19,59}
    clean={63}if voice=='NATF2'else{1,19,59}
    for i,(case,reference)in enumerate(zip(report['cases'],gold)):
        assert case['id']==reference['id']and case['complete']and not case['question_or_gold_in_prompt']
        note='Complete generated answer does not provide the requested correct result; wrong, missing or contradictory answer.'
        if i in clean:note='Requested final result correct without a false numerical explanation in generated text; no speech-quality claim.'
        if voice=='NATF2'and i==50:note='Final15 matches, but explanation subtracts house counts instead of840-825 candy pieces. Correct result with invalid reasoning.'
        if voice=='NATF2'and i==55:note='Final19 matches, but the claim of being12 earlier conflicts with the stated age relation. Correct result with unsupported explanation.'
        if voice=='NATF2'and i==28:note='Mentions126 as a difference, with false sum645 and middle129; does not correctly answer the mean. Target mention alone is not success.'
        if voice=='NATF2'and i==20:note='Correct intermediate118 and3*118, but missing requested final354; retained as partial computation, not exact final success.'
        if voice=='NATF2'and i==47:note='Correct marked-up26 but misses requested50% sale price13; partial computation only.'
        if i==33:note+=' Original twice-more source ambiguity retained; neither output supplies3200 or the plausible alternative total2400.'
        if i==63:note+=' Original numeric_target is null; independent reference280/(2+20/60)=120mph is recorded separately without editing original metadata.'
        review.append(dict(voice=voice,id=case['id'],question=reference['question'],original_reference=reference['response'],original_numeric_target=reference['numeric_target'],independent_numeric_reference='120'if i==63 else None,generated_text=case['generated_text'],final_answer_correct=i in final,answer_without_false_explanation=i in clean,source_ambiguity_retained=i==33,native_RET=case['actual_native_RET'],backend_requests=case['actual_backend_requests'],review_note=note))
    path=ROOT/'official8526401-reports'/(voice+'.json')
    scores[voice]=dict(required=64,final_answer_correct=len(final),answer_without_false_explanation=len(clean),native_RET=sum(c['actual_native_RET']for c in report['cases']),backend_requests=sum(c['actual_backend_requests']for c in report['cases']),source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
result=dict(status='COMPLETE128_GENERATED_TEXT_REVIEW',job='8526401',producer='f926661581d09d123f04b4e83f699e580cd804c4',scores=scores,cases=review,reviewer='Coding agent reviewed every complete generated answer against original question/reference; no model judge or inference rerun.',new_optimizer_updates=0,external_API_calls=0,spoken_audio_independently_scored=False,product_accepted=False,limitations=['Quantitative cohort is not representative of all native reasoning or domains.','These familiar/public/templated examples are not fresh generalization evidence.','Stronger text-backend success does not prove identical7B native text competence.','No connected backend was present; native RET is a route signal, not task execution.'],finding='Both untouched official voice conditions remain poor on quantitative reasoning; the expanded adapter cannot be the sole cause. Changing voice reference alone does not solve this cohort. Native routing, functional retention and broader reasoning data remain separate required interventions.')
Path('docs/NATIVE-OFFICIAL128-SEMANTIC-REVIEW-2026-11-17.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(scores,indent=2))
