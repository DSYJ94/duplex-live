from collections import Counter
import hashlib,json
from data.venus.retention_speech import prepare,PLAN,MODEL,VOICES


def test_all96_train_sources_and_spoken_targets_have_exact_complete_binding():
    p=prepare();source=json.loads(PLAN.read_text());by={r['id']:r for r in p['requests']}
    assert len(p['requests'])==182 and len(p['scenes'])==96 and p['characters']==12746
    assert p['maximum_additional_cash_charge_usd']==0
    assert p['conservative_cost_upper_usd']==6.373
    for scene,original in zip(p['scenes'],source['scenes']):
        assert scene['id']==original['id'] and scene['fold']=='TRAIN'
        assert [by[x]['body']['text'] for x in scene['user_utterance_ids']]==original['user_turns']
        for target in scene['assistant_spoken_targets']:
            assert by[target['utterance_id']]['body']['text']==original['teacher_program'][target['program_field']]
    assert all(r['body']['model_id']==MODEL for r in by.values())
    assert all(set(r['body'])=={'text','model_id','language_code','seed','voice_settings'} for r in by.values())
    assert all('assistant-british'==by[t['utterance_id']]['profile'] for s in p['scenes'] for t in s['assistant_spoken_targets'])
    assert p['training_approved'] is False


def test_only_existing_british_and_scottish_profiles_no_owner_or_holdout_payloads():
    p=prepare()
    assert {r['voice_id'] for r in p['requests']}==set(VOICES.values())
    assert all(r['fold']=='TRAIN' for r in p['requests'])
    assert not p['owner_conversations'] and not p['human_ratings_sent'] and not p['score_annotations_sent']
