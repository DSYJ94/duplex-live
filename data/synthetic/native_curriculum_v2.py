"""New native source conversations: varied reasoning, quiet cues and task changes.

These are authored source scripts, not captured model/evaluation outputs. The
renderer, measured speech checks and actual GPT-Live teacher receipts must still
approve the resulting recordings before any native supervision is eligible.
"""
import argparse
import json
from pathlib import Path
import random


QA = {
    'train': [
        ('All the towels are in the blue box. Where is my towel?', 'Your towel is in the blue box.', [['towel'], ['blue'], ['box']]),
        ('The train arrived after the bus. Which arrived first?', 'The bus arrived first.', [['bus'], ['first']]),
        ('Why does a ball roll down a slope?', 'Gravity pulls the ball down the slope.', [['gravity'], ['down']]),
        ('What does a thermometer measure?', 'A thermometer measures temperature.', [['temperature']]),
        ('What do roots take in from the soil?', 'Roots take in water and dissolved minerals.', [['water'], ['minerals']]),
        ('Which tool measures length, a ruler or a mug?', 'A ruler measures length.', [['ruler'], ['length']]),
    ],
    'validation': [
        ('What does a map show?', 'A map shows places and their positions.', [['places'], ['positions', 'locations']]),
        ('What does a calendar help track?', 'A calendar helps track dates.', [['dates']]),
    ],
    'test': [
        ('What does a scale measure?', 'A scale measures weight.', [['weight']]),
        ('What does a stamp on a letter show?', 'A stamp shows that postage has been paid.', [['postage'], ['paid']]),
    ],
}


def specifications():
    rng = random.Random(18102026)
    values = rng.sample([c + ' ' + s for c in ('red', 'blue', 'green', 'yellow', 'purple', 'orange', 'silver', 'gold', 'white', 'black', 'pink', 'brown')
        for s in ('circle', 'square', 'triangle', 'star', 'diamond', 'oval', 'heart', 'cross')], 28)
    scenes, value_index = [], 0
    for split, counts, voices, projects, fields in [
        ('train', (8, 8, 4, 4), ('alloy', 'echo'), ('Laurel', 'Hazel', 'Juniper', 'Cypress'), ('cabinet code', 'parcel label')),
        ('validation', (4, 2, 1, 1), ('coral',), ('Spruce', 'Acacia'), ('booking phrase',)),
        ('test', (4, 2, 1, 1), ('verse',), ('Poplar', 'Rowan'), ('storage emblem',)),
    ]:
        kinds = [kind for kind, count in zip(('private', 'simple', 'clarify', 'cancel'), counts) for _ in range(count)]
        simple_index = 0
        for index, kind in enumerate(kinds):
            field = fields[index % len(fields)]
            project = projects[index % len(projects)] + ' ' + ('North', 'South', 'East', 'West')[index // len(projects) % 4]
            phrase = None if kind == 'simple' else values[value_index]
            value_index += kind != 'simple'
            question, answer, facts = QA[split][index % len(QA[split])]
            extra = {}
            if kind == 'simple':
                if split == 'train' and simple_index == 6 or split != 'train' and simple_index == 1:
                    cue = {'train': 'ready', 'validation': 'start', 'test': 'begin'}[split]
                    question, answer, facts = ('What does a timer measure?', 'A timer measures elapsed time.', [['elapsed'], ['time']]) if split == 'train' else QA[split][0]
                    utterances = [('user', f'Do not speak until I say {cue}. Listen quietly.'),
                        ('user', cue.capitalize() + '. ' + question), ('assistant', answer)]
                    extra = dict(gaps_before_s={'1': 4.}, quiet_until_user_ordinal=1, behavior='listen-until-cue')
                elif split == 'train' and simple_index == 7:
                    utterances = [('user', 'Say only the first step now: put the book on the shelf. After I say go, say close the cupboard.'),
                        ('assistant', 'Put the book on the shelf.'),
                        ('user', 'Pause. Stay quiet until I say go.'), ('user', 'Go.'),
                        ('assistant', 'Close the cupboard.')]
                    facts = [['book'], ['shelf'], ['close'], ['cupboard']]
                    extra = dict(gaps_before_s={'3': 4.}, quiet_between_user_ordinals=[1, 2],
                        no_restart_after_cue=['book', 'shelf'], behavior='pause-and-continue')
                else:
                    question, answer, facts = QA[split][simple_index % len(QA[split])]
                    utterances = [('user', question), ('assistant', answer)]
                simple_index += 1
            elif kind == 'clarify':
                utterances = [('user', f'What is the {field}?'), ('assistant', 'Which project do you mean?'),
                    ('user', 'Project ' + project + '.'), ('assistant', 'I will check.'),
                    ('user', 'While you check, ' + question[0].lower() + question[1:]),
                    ('assistant', answer), ('assistant', f'The {field} is {phrase}.')]
            else:
                utterances = [('user', f'Please check the {field} for Project {project}.'),
                    ('assistant', 'I will check.'),
                    ('user', 'While you check, ' + question[0].lower() + question[1:]), ('assistant', answer)]
                if kind == 'cancel':
                    nq, na, nf = QA[split][(index + 1) % len(QA[split])]
                    utterances += [('user', 'Cancel that lookup. Instead, ' + nq[0].lower() + nq[1:]),
                        ('assistant', 'The lookup is cancelled. ' + na)]
                    facts = facts + nf
                else:
                    utterances += [('assistant', f'The {field} is {phrase}.')]
            scenes.append(dict(scenario_id=f'native2-{split}-{index:03d}', split=split, kind=kind,
                script_family=f'native2-{split}-general' if kind == 'simple' else f'native2-{split}-{field}',
                project=project, field=field, phrase=phrase, user_voice=voices[index % len(voices)],
                assistant_voice='marin', nominal_worker_delay_s=(3., 8., 12., 16.)[index % 4],
                utterances=utterances, curriculum_version=2, created_date='2026-10-18',
                required_facts=facts, **extra))
    if len(scenes) != 40 or value_index != 28:
        raise ValueError('Changed native2 source inventory')
    work = [text for s in scenes for _, text in s['utterances']]
    if len(work) > 200 or sum(map(len, work)) > 10000:
        raise ValueError('New source exceeds the existing bounded generation budget')
    return scenes


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(specifications(), indent=2) + '\n')
