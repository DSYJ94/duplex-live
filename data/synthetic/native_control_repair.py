"""Author source-only native control repair scripts, without generating audio.

All forty scenes are TRAIN proposals. None is an approved teacher, new holdout,
or authorization to run the historical renderer on unsupported silent endings.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


# No cue or wording is copied from the owner's interaction or sealed fresh tests.
CUES = ('beacon', 'orbit', 'lantern', 'walnut', 'copper', 'harvest', 'marble', 'velvet')
NEAR = ('bacon', 'order', 'laundry', 'wallet', 'coffee', 'harness', 'maple', 'helmet')
QA = (
    ('Every small basket holds apples. This basket is small. What does it hold?', 'It holds apples.'),
    ('Mia came before Leo. Leo came before Sam. Who came first?', 'Mia came first.'),
    ('There are five stones. I add three and remove two. How many remain?', 'Six stones remain.'),
    ('The coin is in one of two jars. The round jar is empty. Where is the coin?', 'The coin is in the other jar.'),
    ('Why does a shadow move when a lamp moves?', 'The light reaches the object from a different direction.'),
    ('Why does water form drops on a cold glass?', 'Water vapour in the air cools and condenses on the glass.'),
    ('Which carries light to my eyes, a sound wave or reflected light?', 'Reflected light carries light to your eyes.'),
    ('If the path is blocked, we use the bridge. The path is blocked. Which way do we go?', 'We use the bridge.'),
)
STEPS = (
    ('Turn the card over.', 'Move the pencil to the left.'),
    ('Set the cup on the mat.', 'Fold the cloth in half.'),
    ('Lift the red block.', 'Put the green block beside it.'),
    ('Open the small case.', 'Place the coin inside it.'),
    ('Draw a short line.', 'Add a circle below it.'),
    ('Point at the window.', 'Touch the table.'),
    ('Put down the brush.', 'Pick up the sponge.'),
    ('Move the chair back.', 'Place the cushion on it.'),
)
INSTRUCTIONS = (
    'Stay completely silent until I say {cue} in a later turn.',
    'Please listen without replying. Wait until I later say {cue}.',
    'Do not answer or acknowledge this. Speak only after I say {cue} again.',
    'Keep listening quietly. I will release you by saying {cue} later.',
)


def turn(role, text, gap=0.56):
    return dict(role=role, text=text, gap_before_s=gap)


def build():
    rows = []
    for family in ('no-release', 'release-reason', 'distractor-release', 'pause-resume', 'replace-cue'):
        for i, cue in enumerate(CUES):
            question, answer = QA[i]
            instruction = INSTRUCTIONS[(i + len(rows) // 8) % len(INSTRUCTIONS)].format(cue=cue)
            release = 1
            quiet_start = 0
            trailing = 1.0
            waits = (4.0, 8.0, 10.0)
            if family == 'no-release':
                # Retain a genuine spoken target group before the quiet request.
                # A silent-only teacher would fail the unchanged native streamed
                # objective's required spoken+clock groups. Do not weaken it.
                turns = [turn('user', question), turn('assistant', answer),
                         turn('user', instruction)]
                quiet_start = 2
                release = None
                trailing = (10.0, 20.0, 30.0)[i % 3]
            elif family == 'release-reason':
                turns = [turn('user', instruction),
                         turn('user', cue.capitalize() + '. ' + question, waits[i % 3]),
                         turn('assistant', answer)]
            elif family == 'distractor-release':
                turns = [turn('user', instruction),
                         turn('user', 'I am still thinking. There is no need to reply.', 8.0),
                         turn('user', NEAR[i].capitalize() + '.', 8.0),
                         turn('user', cue.capitalize() + '. ' + question, 8.0),
                         turn('assistant', answer)]
                release = 3
            elif family == 'pause-resume':
                first, second = STEPS[i]
                turns = [turn('user', f'Say only the first step now: {first} After I say {cue}, say {second}'),
                         turn('assistant', first),
                         turn('user', f'Pause here. Stay silent until I say {cue} in a later turn.'),
                         turn('user', cue.capitalize() + '.', waits[i % 3]),
                         turn('assistant', second)]
                quiet_start, release = 2, 3
            else:
                replacement = CUES[(i + 1) % len(CUES)]
                turns = [turn('user', instruction),
                         turn('user', f'Change the cue to {replacement}. Keep quiet until I say it in another turn.', 8.0),
                         turn('user', cue.capitalize() + '.', 8.0),
                         turn('user', replacement.capitalize() + '. ' + question, 8.0),
                         turn('assistant', answer)]
                release = 3
            row = dict(id=f'native-control3-train-{len(rows):03d}', split='train',
                       family=family, cue=cue,
                       user_voice=('alloy', 'echo')[(i + len(rows) // 8) % 2],
                       assistant_voice='marin', turns=turns,
                       quiet_start_after_turn=quiet_start, release_turn=release,
                       trailing_observation_s=trailing,
                       external_requests_allowed=False,
                       expected_assistant_turns=[n for n, t in enumerate(turns) if t['role'] == 'assistant'],
                       required='No acknowledgment, clarification, words or audible speech while quiet; '
                                'after a valid later cue complete the stated task without inventing facts.',
                       training_approved=False)
            if family == 'replace-cue':
                row['replacement_cue'] = replacement
            if family == 'pause-resume':
                row['no_restart_text'] = first
            rows.append(row)
    return dict(schema='dl-native-control3-source-plan-v1',
                status='source-only; no audio generated; all approvals pending',
                owner_conversations_used=False, external_speech_api_used=False,
                model_or_training_recipe_changed=False, training_approved=False,
                collection_scope='40 entirely authored TRAIN source scenes; preserve all existing folds and heldouts',
                model_inputs='User PCM, ordinary native persona/voice and actual arrived worker receipts only; '
                             'no cue fields, phase state, scoring references or answer text injected at inference.',
                required_chain=['source audio', 'actual native teacher histories', 'independent complete quality',
                                'independent semantic review', 'conversion', 'native numerical approval',
                                'bounded pilot', 'independent state restore', 'complete behavioral review'],
                prohibited_shortcuts=['No output keyword gate or forced silence', 'No passing subset',
                                      'No gain lowering or threshold weakening', 'No heldout relabeling',
                                      'No unchanged extension of a failed candidate'],
                generation_blockers=['Use the separately tested native_control_render adapter for raw source '
                                     'assembly. The historical renderer cannot represent terminal no-release silence.',
                                     'Historical semantic/quiet checks need multiple non-release user turns and '
                                     'cue-replacement support; do not silently use weaker old semantics.',
                                     'Real speech/teacher API requests require a locally configured credential; '
                                     'no remote request is made by this module.'],
                existing_28_histories_and_49_teachers_retained=True,
                retention_kl_weight=1.0, retention_hidden_mse_weight=10.0,
                cases=rows)


def validate(plan):
    if plan['training_approved'] is not False or plan['owner_conversations_used'] is not False:
        raise ValueError('This is an unapproved synthetic source plan only')
    rows = plan['cases']
    if len(rows) != 40 or len({r['id'] for r in rows}) != 40:
        raise ValueError('Complete40 distinct source cases required')
    families = Counter(r['family'] for r in rows)
    if families != Counter({family: 8 for family in
                            ('no-release', 'release-reason', 'distractor-release', 'pause-resume', 'replace-cue')}):
        raise ValueError('Complete five-family coverage required')
    total_chars = total_turns = 0
    for row in rows:
        turns = row['turns']
        if row['split'] != 'train' or row['training_approved'] is not False or row['external_requests_allowed'] is not False:
            raise ValueError('No heldout, approved or external-worker scene in this source expansion')
        if row['user_voice'] not in ('alloy', 'echo') or row['cue'] not in CUES:
            raise ValueError('Unexpected source voice/cue')
        if not turns or turns[0]['role'] != 'user':
            raise ValueError('User instruction required')
        for t in turns:
            if t['role'] not in ('user', 'assistant') or not t['text'].strip():
                raise ValueError('Nonempty authored source words required')
            gap = t['gap_before_s']
            if type(gap) not in (int, float) or not math.isfinite(gap) or not 0 <= gap <= 10:
                raise ValueError('Invalid bounded source gap')
        start, release = row['quiet_start_after_turn'], row['release_turn']
        if not isinstance(start, int) or not 0 <= start < len(turns) or turns[start]['role'] != 'user':
            raise ValueError('Quiet starts after a user instruction')
        if release is not None and (not isinstance(release, int) or not start < release < len(turns)
                                    or turns[release]['role'] != 'user'):
            raise ValueError('Invalid later release')
        end = len(turns) if release is None else release
        if any(t['role'] == 'assistant' for t in turns[start + 1:end]):
            raise ValueError('Speech inside a no-speech interval')
        actual_assistant = [n for n, t in enumerate(turns) if t['role'] == 'assistant']
        if actual_assistant != row['expected_assistant_turns']:
            raise ValueError('Expected speaker inventory mismatch')
        if release is None:
            if any(n > start for n in actual_assistant) or not actual_assistant or not 10 <= row['trailing_observation_s'] <= 30:
                raise ValueError('No-release requires actual terminal silence, no fake assistant utterance')
        elif not any(n > release for n in actual_assistant):
            raise ValueError('Silence alone cannot satisfy a released task')
        if row['family'] == 'pause-resume' and len(actual_assistant) != 2:
            raise ValueError('Pause needs both original and remaining steps')
        total_turns += len(turns)
        total_chars += sum(len(t['text']) for t in turns)
    if total_turns > 200 or total_chars > 10000:
        raise ValueError('Source proposal exceeds explicit local plan bounds')
    return dict(cases=40, families=dict(families), source_turns=total_turns,
                source_characters=total_chars, training_approved=False,
                no_remote_requests=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = build()
    summary = validate(plan)
    with args.out.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(summary, sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
