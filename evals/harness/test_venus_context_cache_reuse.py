import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from training.venus.context_cache_reuse import admitted_rows,load_verified

ROOT=Path('runs/venus-training-20261102/placement-v50/copyback/cache/new')
TIMELINES=Path('runs/venus-training-20261101/context-timelines-v3')


def duplex():
    base=Path('runs/venus-training-20261029/action-target-audit-v1')
    tokenizer=Tokenizer.from_file(str(base/'tokenizer.json'))
    conf=json.loads((base/'tokenizer_config.json').read_text())
    ids={r['content']:int(k) for k,r in conf['added_tokens_decoder'].items()}
    return SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text,**kw:tokenizer.encode(text,add_special_tokens=False).ids),listen_token_id=ids['<|listen|>'],speak_token_id=ids['<|speak|>'],turn_eos_token_id=ids['<|turn_eos|>'],chunk_eos_token_id=ids['<|chunk_eos|>'])


@pytest.mark.skipif(not ROOT.exists(),reason='Actual copied-back native histories required')
def test_all14_real_histories_match_exact_current_words_folds_and_native_targets():
    rows=admitted_rows(ROOT);d=duplex()
    for ident,row in rows.items():
        timeline=json.loads((TIMELINES/ident/'timeline.json').read_text())
        ex,proof=load_verified(ROOT/(ident+'.pt'),row,timeline,d)
        assert ex['fold']=='TRAIN' and proof['passed'] and not proof['independently_recomputed_this_process']
        assert not proof['original_whole_job_terminal_success']
    assert len(rows)==14


@pytest.mark.skipif(not ROOT.exists(),reason='Actual copied-back native histories required')
def test_changed_current_timing_or_fold_cannot_relabel_reused_history():
    rows=admitted_rows(ROOT);ident,row=next(iter(rows.items()))
    timeline=json.loads((TIMELINES/ident/'timeline.json').read_text());timeline['fold']='DEV'
    with pytest.raises(ValueError):load_verified(ROOT/(ident+'.pt'),row,timeline,duplex())


@pytest.mark.skipif(not ROOT.exists(),reason='Actual copied-back native histories required')
def test_wrong_hash_is_rejected_before_deserializing():
    rows=admitted_rows(ROOT);ident,row=next(iter(rows.items()))
    timeline=json.loads((TIMELINES/ident/'timeline.json').read_text())
    with pytest.raises(ValueError):load_verified(ROOT/(ident+'.pt'),dict(row,sha256='0'*64),timeline,duplex())
