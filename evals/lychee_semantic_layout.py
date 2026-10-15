"""Published Lychee target grammar after an actual student listening prefix."""
import math


def continuation(history_length, text_ids, codes, config, token_ids, earliest_slot):
    if (history_length<1 or not text_ids or not codes or earliest_slot<0
            or any(not isinstance(v,int) or not 0<=v<6561 for v in codes)):
        raise ValueError('Invalid native semantic continuation')
    # The listening processor decides S-S at the last slot of a ten-slot chunk.
    ss=math.ceil((max(history_length,earliest_slot)+1)/10)*10-1
    text=[config['text_pad_token_id']]*(ss+1-history_length)
    speech=[config['stoken_pad_token_id']]*len(text)
    controls=[config['detect_token_id'] if slot%10==8 else
        config['keep_listening_token_id'] if slot%10==9 else config['sleep_token_id']
        for slot in range(history_length,ss+1)]
    controls[-1]=config['start_speaking_token_id']
    start=ss+1
    words=list(text_ids)+[token_ids['eot']]
    sound=[config['stoken_delay_token_id']]*config['stoken_delay_num']+[token_ids['tts_start']]
    sound += [v+151696 for v in codes]+[token_ids['tts_end']]
    if len(words)>len(sound):raise ValueError('Native layout would truncate supervised words')
    length=math.ceil((len(sound)+1)/10)*10
    text+=words+[token_ids['tts_pad']]*(len(sound)-len(words))+[config['text_pad_token_id']]*(length-len(sound))
    speech+=sound+[config['stoken_pad_token_id']]*(length-len(sound))
    for chunk in range(length//10):
        controls += [config['sleep_token_id']]*8+[config['detect_token_id']]
        controls += [config['start_listening_token_id'] if chunk==length//10-1 else config['keep_speaking_token_id']]
    if len(text)!=len(speech) or len(text)!=len(controls):raise RuntimeError('Native continuation channels diverged')
    return {'text_ids':text,'stoken_ids':speech,'control_ids':controls,
        'first_word_slot':start,'supervised_slots':list(range(start,start+len(words))),
        'supervised_ids':words,'ss_slot':ss,'codec_units':len(codes)}
