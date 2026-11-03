"""Target diagnostics are distinct from free-running task/speech qualification."""
import pytest
import torch
from training.venus.action_targets import masks, metrics, NAMES

IDS = dict(zip(NAMES, range(10, 17)))
LISTEN, SPEAK, TURN, CHUNK, TTS, OPEN, CLOSE = range(10, 17)


def labelled(values):
    return torch.tensor(values), torch.tensor([v in range(10, 15) for v in values])


def test_multiframe_arguments_do_not_absorb_native_framing_or_spoken_reply():
    values = [LISTEN, SPEAK, OPEN, 2, CHUNK, SPEAK, 3, CLOSE, TURN, CHUNK,
              LISTEN, SPEAK, 4, TURN, CHUNK]
    selected, spans = masks(*labelled(values), IDS)
    assert spans == [[2, 8]]
    assert selected['action'].nonzero().flatten().tolist() == [2, 3, 6, 7]
    assert selected['argument'].nonzero().flatten().tolist() == [3, 6]
    assert selected['frame_decision'].nonzero().flatten().tolist() == [0, 1, 5, 10, 11]


@pytest.mark.parametrize('values', [[SPEAK, OPEN, 2, CHUNK], [SPEAK, CLOSE, CHUNK],
    [SPEAK, OPEN, CLOSE, CHUNK], [SPEAK, OPEN, OPEN, 2, CLOSE, CHUNK],
    [SPEAK, OPEN, 2, CHUNK, LISTEN], [SPEAK, 2], [CHUNK], [2, LISTEN]])
def test_bad_control_or_partial_action_cannot_be_counted_as_complete(values):
    with pytest.raises(ValueError):
        masks(*labelled(values), IDS)


def test_cache_decision_mask_drift_rejected():
    values, mask = labelled([SPEAK, OPEN, 2, CLOSE, TURN, CHUNK])
    mask[2] = True
    with pytest.raises(ValueError, match='decision labels'):
        masks(values, mask, IDS)


def test_argument_error_not_hidden_by_correct_control_tokens():
    targets, decisions = labelled([LISTEN, SPEAK, OPEN, 2, CLOSE, TURN, CHUNK])
    selected, _ = masks(targets, decisions, IDS)
    logits = torch.full((len(targets), 20), -10.)
    logits[torch.arange(len(targets)), targets] = 10.
    correct = metrics(logits, targets, selected)
    logits[3, 2], logits[3, 3] = -10., 10.
    broken = metrics(logits, targets, selected)
    assert correct['argument_token_accuracy'] == 1.
    assert broken['frame_decision_token_accuracy'] == 1.
    assert broken['argument_token_accuracy'] == 0.
    assert broken['argument_ce'] > correct['argument_ce'] + 10


def test_non_action_scene_has_no_fabricated_argument_measurement():
    targets, decisions = labelled([LISTEN, SPEAK, 2, TURN, CHUNK])
    selected, _ = masks(targets, decisions, IDS)
    result = metrics(torch.zeros((len(targets), 20)), targets, selected)
    assert result['argument_tokens'] == 0 and result['argument_ce'] is None


def test_prepared_objective_default_and_diagnostic_gradients_are_unchanged(monkeypatch):
    from training.venus import lora
    targets, decisions = labelled([LISTEN, SPEAK, OPEN, 2, CLOSE, TURN, CHUNK,
                                  SPEAK, 4, TURN, CHUNK])
    logits = torch.zeros((len(targets), 20), requires_grad=True)
    monkeypatch.setattr(lora, 'selected_forward', lambda *args: (logits, torch.zeros(len(targets), 3)))
    example = dict(targets=targets, decision_mask=decisions, family='delegate', fold='TRAIN')
    original, _ = lora.scene_loss(None, example, 'cpu')
    diagnostic, parts = lora.scene_loss(None, example, 'cpu', action_token_ids=IDS)
    weighted, _ = lora.scene_loss(None, example, 'cpu', action_token_ids=IDS, action_aux_weight=.25)
    old_grad = torch.autograd.grad(original, logits, retain_graph=True)[0]
    same_grad = torch.autograd.grad(diagnostic, logits, retain_graph=True)[0]
    new_grad = torch.autograd.grad(weighted, logits)[0]
    assert torch.equal(original, diagnostic) and torch.equal(old_grad, same_grad)
    selected, _ = masks(targets, decisions, IDS)
    assert torch.equal(old_grad[~selected['action']], new_grad[~selected['action']])
    assert not torch.equal(old_grad[selected['action']], new_grad[selected['action']])
    assert weighted.item() == pytest.approx(original.item() + .25 * parts['action_auxiliary'])


def test_ordinary_retention_is_present_and_unchanged_by_action_auxiliary(monkeypatch):
    from training.venus import lora
    targets, decisions = labelled([LISTEN, SPEAK, 2, TURN, CHUNK])
    logits = torch.zeros((len(targets), 20), requires_grad=True)
    monkeypatch.setattr(lora, 'selected_forward', lambda *args: (logits, torch.ones(len(targets), 3)))
    example = dict(targets=targets, decision_mask=decisions, family='ordinary', fold='DEV')
    model = torch.nn.Linear(1, 1)
    original, before = lora.scene_loss(model, example, 'cpu')
    weighted, after = lora.scene_loss(model, example, 'cpu', action_token_ids=IDS, action_aux_weight=.25)
    assert torch.equal(original, weighted)
    assert all(after[key] == value for key, value in before.items())
    assert after['action_auxiliary'] == 0


def test_critical_metrics_merge_real_token_counts_and_exclude_missing_action_scenes():
    from training.venus.diagnostics import aggregate
    action = dict(ce=1., argument_ce=2., argument_tokens=3)
    missing = dict(ce=1., argument_ce=None, argument_tokens=0)
    group = aggregate([('delegate', action, 1), ('delegate', missing, 1)])
    merged = aggregate([('delegate', group['delegate'], 2), ('delegate', action, 1)])['delegate']
    assert merged['argument_ce'] == 2 and merged['argument_ce_count'] == 2
    assert merged['argument_tokens'] == 6 and merged['count'] == 3


@pytest.mark.parametrize('weight', [-1, float('nan'), 1.])
def test_prepared_objective_rejects_invalid_weights_before_model_loading(weight):
    from training.venus.lora import scene_loss
    with pytest.raises(ValueError, match='weight'):
        scene_loss(None, {}, 'cpu', action_aux_weight=weight)


def test_bad_chunk_end_is_visible_even_when_listen_speak_decisions_are_correct():
    targets,decisions=labelled([LISTEN,SPEAK,2,CHUNK,SPEAK,3,TURN,CHUNK])
    selected,_=masks(targets,decisions,IDS)
    logits=torch.full((len(targets),20),-10.)
    logits[torch.arange(len(targets)),targets]=10.
    logits[3,CHUNK],logits[3,4]=-10.,10.
    result=metrics(logits,targets,selected,control_token_ids=IDS)
    assert result['frame_decision_token_accuracy']==1.
    assert result['native_chunk_end_tokens']==2 and result['native_chunk_end_token_accuracy']==.5
    assert result['native_chunk_end_ce']>9 and result['native_turn_end_token_accuracy']==1.
    assert result['native_chunk_tts_end_tokens']==0 and result['native_chunk_tts_end_ce'] is None


def test_native_cadence_fields_reach_opal_review_without_inventing_absent_classes():
    from training.venus.diagnostics import aggregate,review
    parts=dict(ce=1.,native_chunk_end_ce=4.,native_chunk_end_token_accuracy=.5,native_chunk_end_tokens=2)
    groups=aggregate([('ordinary',parts,1),('ordinary',dict(ce=1.),1)])
    packet,_=review([dict(step=1,family_train_metrics=groups)],1,{})
    assert packet['agent_metrics']['train_window_ordinary_native_chunk_end_ce']==4.
    assert packet['agent_metrics']['train_window_ordinary_native_chunk_end_tokens']==2.
    assert packet['train']['ordinary']['native_chunk_end_ce_count']==1
    assert packet['train']['ordinary']['native_chunk_tts_end_ce'] is None
    assert not packet['causal_diagnosis_established'] and not packet['heldout_available']
