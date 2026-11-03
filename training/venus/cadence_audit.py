"""Read actual saved label tensors; no model weights, forward pass or mutations."""
import argparse,collections,hashlib,json
from pathlib import Path
from types import SimpleNamespace


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames_from_targets(targets,listen,speak,chunk_end):
    """Recover native decode frames without inferring time from word lengths."""
    frames=[];current=[]
    for token in targets:
        if not current and token==listen:frames.append([token]);continue
        if not current and token!=speak:raise ValueError('Target frame lacks native listen/speak opening')
        current.append(token)
        if token==chunk_end:frames.append(current);current=[]
    if current:raise ValueError('Unclosed native target frame')
    return frames


def speech_tokens(frames,ids):
    """Exclude native private action contents across frame boundaries."""
    inside=False;result=[];special=set(ids.values())
    for labels in frames:
        visible=[]
        for token in labels:
            if token==ids['<delegate>']:
                if inside:raise ValueError('Nested action in approved target')
                inside=True
            elif token==ids['</delegate>']:
                if not inside:raise ValueError('Unopened action in approved target')
                inside=False
            elif not inside and token not in special:visible.append(token)
        result.append(visible)
    if inside:raise ValueError('Incomplete action in approved target')
    return result


def run(archive,old_cache,old_timelines,tokenizer_dir):
    import torch
    from tokenizers import Tokenizer
    from .native_schedule import schedule
    archive,old_cache,old_timelines,tokenizer_dir=map(Path,(archive,old_cache,old_timelines,tokenizer_dir))
    manifest=json.loads((archive/'cache/manifest.json').read_text());old=json.loads((old_cache/'manifest.json').read_text())
    assert sha(old_cache/'manifest.json')==manifest['old_prefix_manifest_sha256']
    sources=json.loads((archive/'cache/timelines-manifest.json').read_text())
    assert sha(archive/'cache/timelines-manifest.json')==manifest['source_manifest_sha256']
    assert sha(old_timelines/'manifest.json')==sources['old_manifest_sha256']
    assert sha(archive/'timelines/manifest.json')==sources['new_manifest_sha256']
    new_source=json.loads((archive/'timelines/manifest.json').read_text())
    assert sha(tokenizer_dir/'tokenizer.json')==new_source['tokenizer_sha256']
    index=json.loads((archive/'cache/action-index.json').read_text());witness={r['id']:r for r in index['rows']}
    assert index['prefix_manifest_sha256']==sha(archive/'cache/manifest.json') and len(witness)==304
    ids=index['token_ids'];tok=Tokenizer.from_file(str(tokenizer_dir/'tokenizer.json'))
    for text,ident in ids.items():assert tok.token_to_id(text)==ident
    tokenizer=SimpleNamespace(encode=lambda text,add_special_tokens=False:tok.encode(text,add_special_tokens=add_special_tokens).ids)
    duplex=SimpleNamespace(tokenizer=tokenizer,listen_token_id=ids['<|listen|>'],speak_token_id=ids['<|speak|>'],turn_eos_token_id=ids['<|turn_eos|>'],chunk_eos_token_id=ids['<|chunk_eos|>'])
    by_id={r['id']:r for r in sources['scenes']};old_by={r['id']:r for r in old['scenes']}
    rows=[];groups=collections.defaultdict(list)
    assert len({r['id'] for r in manifest['scenes']})==304 and {r['id'] for r in manifest['scenes']}==set(witness)
    assert len(manifest['scenes'])==304 and collections.Counter(x['fold'] for x in manifest['scenes'])==dict(TRAIN=240,DEV=64)
    for entry in manifest['scenes']:
        ident=entry['id'];is_old=entry['file'].startswith('old/')
        if is_old:
            prior=old_by[ident];assert prior['sha256']==entry['sha256']
            binary=old_cache/prior['file'];timeline_path=old_timelines/ident/'timeline.json'
        else:
            binary=archive/'cache'/entry['file'];timeline_path=archive/'timelines'/ident/'timeline.json'
        assert sha(timeline_path)==by_id[ident]['timeline_sha256']
        timeline=json.loads(timeline_path.read_text())
        assert (timeline['fold'],timeline['family'])==(entry['fold'],entry['family'])
        # mmap leaves embeddings/prefix-hidden storage untouched. Only actual
        # labels and decision bits are paged in; never import/load native9B.
        ex=torch.load(binary,mmap=True,weights_only=True,map_location='cpu')
        assert (ex['id'],ex['fold'],ex['family'])==(ident,entry['fold'],entry['family'])
        assert ex['timeline_sha256']==hashlib.sha256(json.dumps(timeline,sort_keys=True).encode()).hexdigest()
        targets=ex['targets'].tolist();digest=hashlib.sha256(json.dumps(targets,separators=(',',':')).encode()).hexdigest()
        assert digest==witness[ident]['target_sha256'] and entry['sha256']==witness[ident]['prefix_sha256']
        assert len(targets)==entry['labels']
        actual=frames_from_targets(targets,duplex.listen_token_id,duplex.speak_token_id,duplex.chunk_eos_token_id)
        expected=schedule(duplex,timeline)
        # Pinned public duplex defaults force_listen_count=0. The preserved
        # collection used that configuration; equality checks every frame.
        planned=[expected.get(n,[duplex.listen_token_id]) for n in range(timeline['frames'])]
        assert actual==planned,'Actual label/timeline frame mismatch: '+ident
        special={ids[k] for k in ('<|listen|>','<|speak|>','<|turn_eos|>','<|chunk_eos|>','<|chunk_tts_eos|>')}
        assert ex['decision_mask'].tolist()==[token in special for token in targets]
        visible=speech_tokens(actual,ids);units=[]
        for n,(labels,words) in enumerate(zip(actual,visible)):
            if labels[0]==duplex.listen_token_id:continue
            units.append(dict(frame=n,visible_text_tokens=len(words),final=duplex.turn_eos_token_id in labels,has_action=any(t in (ids['<delegate>'],ids['</delegate>']) for t in labels),text=tok.decode(words,skip_special_tokens=False)))
        row=dict(id=ident,fold=ex['fold'],family=ex['family'],origin='old268' if is_old else 'new36',frames=len(actual),target_sha256=digest,timeline_sha256=sha(timeline_path),actual_targets_equal_native_schedule=True,label_count=len(targets),control_counts={k:targets.count(v) for k,v in ids.items()},speech_units=units)
        rows.append(row);groups[(ex['fold'],ex['family'])].extend(units)
        del ex
    stats=[]
    for (fold,family),units in sorted(groups.items()):
        spoken=[u for u in units if u['visible_text_tokens']];final=[u for u in units if u['final']]
        stats.append(dict(fold=fold,family=family,voice_bearing_frames=len(spoken),final_frames=len(final),max_voice_tokens=max([u['visible_text_tokens'] for u in spoken],default=0),max_final_voice_tokens=max([u['visible_text_tokens'] for u in final],default=0),voice_token_histogram=dict(sorted(collections.Counter(u['visible_text_tokens'] for u in spoken).items())),final_voice_token_histogram=dict(sorted(collections.Counter(u['visible_text_tokens'] for u in final).items()))))
    return dict(complete=True,scenes=len(rows),train=sum(r['fold']=='TRAIN' for r in rows),dev=sum(r['fold']=='DEV' for r in rows),all304_target_digests_match_saved_action_index=True,all304_native_schedules_match_actual_label_tensors=True,all304_decision_masks_match=True,statistics=stats,rows=rows,inputs=dict(cache_manifest_sha256=sha(archive/'cache/manifest.json'),old_prefix_manifest_sha256=sha(old_cache/'manifest.json'),timeline_manifest_sha256=sha(archive/'cache/timelines-manifest.json'),tokenizer_sha256=sha(tokenizer_dir/'tokenizer.json')),limits='Descriptive label cadence, not speech quality or a new approval. Whitespace/phonetic duration thresholds are not invented. Fullbinary/embedding/prefix hashes inherited from verified original backups; only small actual target+mask storages re-read and checked against saved target digests. No model forward/hidden storage scan, originalfiles/approvals unchanged.',new_optimizer_updates=0,new_model_or_API_calls=0,qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','old-cache','old-timelines','tokenizer-dir','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output);assert not out.exists(),'Never overwrite original audit'
    result=run(a.archive,a.old_cache,a.old_timelines,a.tokenizer_dir)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,scenes=result['scenes'],train=result['train'],dev=result['dev'],statistics=result['statistics'])),flush=True)
