"""Reproduce the 2026-09-21 review's three-tone, five-pass counterexample.

Uses the repository's actual metric, verdict, and tone implementations.
No model, GPU, audio-file writes, or changes to the evaluated code.
Run with a Python environment containing numpy, from the repo root.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evals.harness.metrics import measure
from evals.harness.protocol import THRESH, T_PROTOCOL


def load_function(path, name, namespace):
    # Extract the actual function so optional model/audio dependencies are
    # unnecessary for this CPU-only check. No copied verdict logic.
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"),
         namespace)
    return namespace[name]


verdict = load_function("evals/harness/run.py", "verdict", {"THRESH": THRESH})
tone = load_function("evals/harness/targets.py", "_tone_80ms", {"np": np})()
placements = {"t1": [2.40], "t2": [16.64], "t3": [], "t4": [9.28], "t5": []}
results = []
for test, starts in placements.items():
    wav = np.zeros(T_PROTOCOL[test]["total_s"] * 24000, np.float32)
    for start in starts:
        i = round(start * 24000)
        wav[i:i + len(tone)] = tone
    measured = measure(test, T_PROTOCOL[test]["marks"], wav, 24000)
    results.append({"pulse_starts_s": starts, **measured,
                    "verdict": verdict(test, measured)})
print(json.dumps(results, indent=2))
