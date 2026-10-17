"""Keep full-reply and explicitly partial clarification speech audits distinct."""


def approved_words(manifest, side, origin):
    words=[a[0] for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    if manifest.get('native_late_private_version')==1:
        from evals.recovery_prepare_data import spoken_words
        s=manifest['selection'];left,right=s['source_first_word_index'],s['source_end_word_index']
        phrase=spoken_words(next(iter(next(iter(manifest['world'].values())).values())))
        selected=words[left:right]; normalized=spoken_words(' '.join(selected))
        if (manifest.get('target_kind')!='private_result' or not manifest.get('actual_private_receipt')
                or manifest.get('conditioning_policy')!='verified-trained-rollout'
                or manifest['future_recorded_receipts'] or not 0<=left<right<=len(words)
                or not phrase or sum(normalized[i:i+len(phrase)]==phrase for i in range(len(normalized)))!=1):
            raise RuntimeError('Late private speech scope must retain its complete measured factual sentence')
        return selected,'complete_measured_private_sentence_only'
    if manifest.get('target_kind')=='clarification_question':
        from evals.context_clarification_future import question_scope
        count,_=question_scope(origin,manifest['teacher'],side)
        selection=manifest['question_only_selection']
        if (manifest['all_teacher_reply_and_request_tokens_retained'] is not False
                or manifest['later_teacher_reply_and_request_tokens_deliberately_excluded'] is not True
                or manifest['clarification_question_without_backend_context'] is not True
                or manifest['actual_private_receipt'] is not None or manifest['future_recorded_receipts']
                or selection['source_first_word_index']!=0 or selection['source_end_word_index']!=count
                or (manifest.get('native_response_boundary_version')!=3
                    and manifest['future_end_real_frame']!=selection['source_end_frame'])):
            raise RuntimeError('Partial question speech scope changed or acquired a result')
        if manifest.get('native_response_boundary_version')==3 and (
                manifest.get('question_retiming_before_project_verified') is not True
                or manifest['future_end_real_frame']!=manifest['future_first_real_frame']+selection['source_end_frame']-manifest['source_start_frame']
                or manifest['future_end_real_frame']*.08+.24>manifest['question_deadline_s']):
            raise RuntimeError('Retimed trained question reaches the later project utterance')
        return words[:count],'first_clarification_question_only'
    if manifest['all_teacher_reply_and_request_tokens_retained'] is not True:
        raise RuntimeError('Full-reply speech audit cannot silently crop teacher words')
    return words,'complete_teacher_reply'
