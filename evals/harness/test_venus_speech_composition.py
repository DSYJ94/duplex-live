"""Pinned native call-path reproduction; tiny CPU tensors, no model weights.

These tests do not authorize the changed runtime or qualify native audio.
"""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evals.harness.test_venus_speech_completion import duplex, FIXTURE, SAMPLER, speaking
from portable.venus import speech_completion as completion
from portable.venus.speech_channel import install as install_channel


def setup(monkeypatch, eos_at=60):
    monkeypatch.setattr(completion, 'SOURCE_HASHES',
                        {hashlib.sha256(p.read_bytes()).hexdigest() for p in (FIXTURE, SAMPLER)})
    d, state, frame = duplex(eos_at)
    names = {'<delegate>': 8, '</delegate>': 9, '</unit>': 5}
    d.tokenizer = SimpleNamespace(convert_tokens_to_ids=names.__getitem__,
        all_special_ids=list(range(1, 10)),
        decode=lambda ids, skip_special_tokens=False: ' '.join(
            next((name for name, token in names.items() if token == i), str(i))
            for i in ids if not skip_special_tokens or i >= 10))
    conditions = []
    def convert(rows):
        conditions.append([r[0] for r in rows])
        return torch.zeros(1, len(rows) + 1, 2)
    d._convert_results_to_tts_input = convert
    return d, state, frame, conditions


@pytest.mark.parametrize('terminal', [False, True])
def test_action_only_unit_never_samples_acoustics_or_rng(monkeypatch, terminal):
    d, state, frame, conditions = setup(monkeypatch)
    completion.install(d)
    channel = install_channel(d)
    ids = [2, 8, 20, 21, 9] + ([3] if terminal else []) + [4]
    torch.manual_seed(42)
    rng = torch.get_rng_state()
    result = frame(ids)
    assert result['n_tts_tokens'] == 0, 'Filtered action-only conditioning must not bypass the no-speech sampler'
    assert not np.any(result['audio_waveform'])
    assert torch.equal(torch.get_rng_state(), rng)
    assert not result['speech_completion']['pending']
    assert channel.hidden_tokens == 4
    assert d.total_ids == ids + [5], 'Internal task tokens must remain in native history'


def test_native_turn_eos_missing_from_hf_special_map_never_becomes_spoken_content(monkeypatch):
    # The pinned public tokenizer hides turn_eos when decoding, but omits it
    # from all_special_ids. Observed delegate-only native units retain this ID.
    d, state, frame, conditions = setup(monkeypatch)
    d.tokenizer.all_special_ids.remove(3)
    completion.install(d)
    install_channel(d)
    ids = [2, 8, 20, 21, 9, 3, 4]
    torch.manual_seed(42)
    before = torch.get_rng_state()
    result = frame(ids)
    assert result['n_tts_tokens'] == 0, 'Native turn delimiter must not trigger speech from an action-only unit'
    assert not np.any(result['audio_waveform'])
    assert torch.equal(torch.get_rng_state(), before)
    assert d.total_ids == ids + [5]


def test_ordinary_terminal_completion_is_unchanged_by_action_filter(monkeypatch):
    base, bs, bf, _ = setup(monkeypatch)
    completion.install(base)
    torch.manual_seed(42)
    expected = [bf(speaking()), bf([1]), bf([1])]
    rng = torch.get_rng_state()
    d, state, frame, conditions = setup(monkeypatch)
    completion.install(d)
    install_channel(d)
    torch.manual_seed(42)
    actual = [frame(speaking()), frame([1]), frame([1])]
    assert state.rendered == bs.rendered and len(state.rendered) == 60
    assert torch.equal(torch.get_rng_state(), rng)
    assert d.total_ids == base.total_ids
    assert all(np.array_equal(a['audio_waveform'], b['audio_waveform']) for a, b in zip(actual, expected))
    assert actual[-1]['speech_completion']['acoustic_complete']


def test_delegate_spanning_units_then_real_speech_keeps_only_visible_conditioning(monkeypatch):
    d, state, frame, conditions = setup(monkeypatch)
    completion.install(d)
    channel = install_channel(d)
    first = frame([2, 8, 20, 4])
    assert first['n_tts_tokens'] == 0 and channel.inside
    result = frame([2, 21, 9, 12, 13, 3, 4])
    assert conditions[-1] == [12, 13, 3]
    frame([1]); frame([1])
    assert len(state.rendered) == 60 and not channel.inside
    assert result['text'], 'Language text is retained for the existing task parser'


def test_prepare_clears_unclosed_action_and_pending_audio(monkeypatch):
    d, state, frame, conditions = setup(monkeypatch)
    handle = completion.install(d)
    channel = install_channel(d)
    frame([2, 8, 20, 4])
    assert channel.inside
    d.prepare()
    assert not channel.inside and channel.hidden_tokens == 0
    assert handle.pending is None
    frame([2, 12, 3, 4])
    assert conditions[-1] == [12, 3]


def test_action_after_pending_turn_cancels_tail_without_sampling_tool_audio(monkeypatch):
    d, state, frame, conditions = setup(monkeypatch)
    handle = completion.install(d)
    install_channel(d)
    frame(speaking())
    pending = handle.pending
    assert pending is not None
    rendered = list(state.rendered)
    rng = torch.get_rng_state()
    result = frame([2, 8, 20, 9, 3, 4])
    assert pending.gi_frame is None and handle.pending is None
    assert state.rendered == rendered and torch.equal(torch.get_rng_state(), rng)
    assert result['n_tts_tokens'] == 0
    assert result['speech_completion']['events'][-1]['kind'] == 'action_only_without_acoustic_sampling'
    assert not any(e.get('reason') == 'acoustic_eos' for e in result['speech_completion']['events'])
    frame([1])
    assert state.rendered == rendered, 'Cancelled acoustic tail must not return on a listening frame'


def test_reverse_installation_and_duplicate_channel_fail_without_more_mutation(monkeypatch):
    d, state, frame, conditions = setup(monkeypatch)
    channel = install_channel(d)
    original = d.streaming_generate
    convert = d._convert_results_to_tts_input
    with pytest.raises(ValueError, match='pinned'):
        completion.install(d)
    assert d.streaming_generate is original and d._convert_results_to_tts_input is convert
    assert not hasattr(d, '_speech_completion')
    with pytest.raises(ValueError, match='already installed'):
        install_channel(d)
    assert d._action_channel is channel and d._convert_results_to_tts_input is convert


def test_runtime_entrypoint_still_rejects_unqualified_composition(monkeypatch):
    from portable.venus import native_app
    monkeypatch.setattr(native_app, 'ensure_guard', lambda: None)
    with pytest.raises(ValueError, match='separate native validation'):
        native_app.NativeEngine('no-weights-loaded', nonspoken_actions=True, speech_completion=True)
