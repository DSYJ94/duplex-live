"""Keep full-reply and explicitly partial clarification speech audits distinct."""


def approved_words(manifest, side, origin):
    words=[a[0] for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    if manifest.get('target_kind')=='clarification_question':
        from evals.context_clarification_future import question_scope
        count,_=question_scope(origin,manifest['teacher'],side)
        selection=manifest['question_only_selection']
        if (manifest['all_teacher_reply_and_request_tokens_retained'] is not False
                or manifest['later_teacher_reply_and_request_tokens_deliberately_excluded'] is not True
                or manifest['clarification_question_without_backend_context'] is not True
                or manifest['actual_private_receipt'] is not None or manifest['future_recorded_receipts']
                or selection['source_first_word_index']!=0 or selection['source_end_word_index']!=count
                or manifest['future_end_real_frame']!=selection['source_end_frame']):
            raise RuntimeError('Partial question speech scope changed or acquired a result')
        return words[:count],'first_clarification_question_only'
    if manifest['all_teacher_reply_and_request_tokens_retained'] is not True:
        raise RuntimeError('Full-reply speech audit cannot silently crop teacher words')
    return words,'complete_teacher_reply'
