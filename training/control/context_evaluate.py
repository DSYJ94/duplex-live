"""Full saved worker outputs, with separate new/previous authored holdouts."""
import json
import time
from pathlib import Path
from training.control.paired_curriculum import payload, assess
from training.control.curriculum import specifications
from training.control.evaluate import evaluate_suite
from training.control.context_adapter import sha


def evaluate(executor, paired, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    results = []
    for scene in paired:
        before = time.monotonic()
        row = dict(id=scene['id'], kind=scene['kind'], scene=scene, passed=False)
        try:
            p, _, _, _ = payload(scene, executor.model_id)
            response = executor(p)
            quality = assess(scene, response)
            row.update(response=response, quality=quality, passed=quality['passed'])
        except Exception as error:
            row.update(error_type=type(error).__name__, message=str(error), evidence=getattr(error, 'result', None))
        row['seconds'] = time.monotonic() - before
        results.append(row)
        (folder / (scene['id'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    old = [s for s in specifications() if s['split'] == 'heldout']
    if len(paired) != 54 or len(old) != 72:
        raise ValueError('Expected all54 new and72 original heldouts')
    summary = dict(paired=dict(cases=len(results), passed=sum(r['passed'] for r in results),
                              by_kind={k: dict(total=sum(r['kind'] == k for r in results), passed=sum(r['kind'] == k and r['passed'] for r in results)) for k in sorted({r['kind'] for r in results})}),
                   previous=evaluate_suite(executor, old, folder / 'previous'))
    (folder / 'paired.json').write_text(json.dumps(results, indent=2) + '\n')
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def retention(baseline, adapted):
    baseline, adapted = Path(baseline), Path(adapted)
    lost = {}
    for name in ('paired', 'previous'):
        a = json.loads((baseline / ('paired.json' if name == 'paired' else 'previous/report.json')).read_text())
        b = json.loads((adapted / ('paired.json' if name == 'paired' else 'previous/report.json')).read_text())
        if name == 'previous':
            a, b = a['results'], b['results']
        aa, bb = {r['id']: r for r in a}, {r['id']: r for r in b}
        if set(aa) != set(bb) or len(aa) != len(a) or len(bb) != len(b):
            raise ValueError('Changed evaluation inventory')
        lost[name] = [n for n, r in aa.items() if r['passed'] and not bb[n]['passed']]
    return dict(all_baseline_passes_retained=not any(lost.values()), lost_passes=lost,
                baseline_summary_sha256=sha(baseline / 'summary.json'), adapted_summary_sha256=sha(adapted / 'summary.json'))
