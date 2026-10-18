"""Actual generated replies in chronological context, not a heard transcript.

Only the media actor calls these after starting a validated worker reply. A
server playout completion does not establish physical device/user hearing.
"""


def started(context,item,event_id):
    task=item['task_id'];text=item['result']['content']
    if not isinstance(task,str) or not task or not isinstance(text,str) or not text.strip() or not isinstance(event_id,str) or not event_id:
        raise ValueError('Generated speech context requires an actual identified nonempty reply')
    matches=[e for e in context if e.get('control_source')=='actual-speech-generation' and e.get('speech_task_id')==task]
    if len(matches)>1:raise ValueError('Duplicate generated assistant turn')
    if matches:
        entry=matches[0]
        if entry['text']!=text or entry['speech_delivery']=='server-playout-complete':raise ValueError('Conflicting or already completed generated speech')
        if entry['latest_speech_event_id']==event_id:raise ValueError('Duplicate playback start')
        entry.update(latest_speech_event_id=event_id,playback_segments=entry['playback_segments']+1)
        return False
    context.append(dict(role='assistant',text=text,control_source='actual-speech-generation',speech_task_id=task,
        first_speech_event_id=event_id,latest_speech_event_id=event_id,playback_segments=1,
        speech_delivery='generated-not-fully-played',physical_hearing_confirmed=False))
    return True


def completed(context,item):
    matches=[e for e in context if e.get('control_source')=='actual-speech-generation' and e.get('speech_task_id')==item['task_id']]
    if len(matches)!=1 or matches[0]['text']!=item['result']['content']:
        raise ValueError('Completion lacks its actual generated assistant turn')
    matches[0]['speech_delivery']='server-playout-complete'


def retired(context,identities):
    for entry in context:
        if (entry.get('control_source')=='actual-speech-generation' and entry.get('speech_task_id') in identities
                and entry['speech_delivery']!='server-playout-complete'):
            entry['speech_delivery']='retired-before-full-playout'


def delivery_states(context):
    result=[]
    for entry in context:
        if entry.get('control_source')!='actual-speech-generation':continue
        if (entry.get('role')!='assistant' or not isinstance(entry.get('text'),str)
                or entry.get('speech_delivery') not in ('generated-not-fully-played','server-playout-complete','retired-before-full-playout')
                or entry.get('physical_hearing_confirmed') is not False):
            raise ValueError('Invalid generated-speech delivery provenance')
        result.append(dict(task_id=entry['speech_task_id'],delivery=entry['speech_delivery'],physical_hearing_confirmed=False))
    return result
