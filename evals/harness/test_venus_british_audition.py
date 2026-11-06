"""Audition stays matched, frozen and separately gated from full qualification."""
import asyncio
from types import SimpleNamespace

import pytest
from training.venus import british_audition as b


def test_exact_three_cases_are_selected_from_existing_approved_inputs(monkeypatch):
    rows=[dict(id=i,input_sha256=str(n)) for n,i in enumerate(b.IDS)]
    monkeypatch.setattr(b,'preflight',lambda plan,*a:rows[:1] if plan=='old' else rows[1:]+[dict(id='unused')])
    a=SimpleNamespace(plan='fresh',timelines='fresh',approval='fresh',old_plan='old',old_timelines='old',old_approval='old')
    selected=b.selected_cases(a)
    assert [r['id'] for r in selected]==list(b.IDS)
    assert [r['input_sha256'] for r in selected]==['0','1','2']


def test_no_substitute_for_missing_case(monkeypatch):
    monkeypatch.setattr(b,'preflight',lambda *a:[dict(id=b.IDS[0])])
    a=SimpleNamespace(plan='',timelines='',approval='',old_plan='',old_timelines='',old_approval='')
    with pytest.raises(ValueError,match='Exact matched'):b.selected_cases(a)


def test_no_native_collection_before_admission(monkeypatch):
    def deny(a):raise ValueError('no approved lease')
    monkeypatch.setattr(b,'admit',deny)
    with pytest.raises(ValueError,match='no approved lease'):asyncio.run(b.collect(SimpleNamespace()))


def test_reference_comparators_are_explicit_and_no_default_voice():
    from portable.venus.speech_composition import REFERENCES
    assert set(b.REFS)=={'rp-ben','british-alice','british-george-comparator'}
    assert set(b.REFS.values())==REFERENCES
