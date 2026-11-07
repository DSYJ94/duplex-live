import copy
import pytest
from data.venus.retention_timeline import verify_receipt


def test_teacher_cancellation_requires_actual_matching_pending_target_not_empty_ack():
    scene=dict(id='scene',family='cancel',synthetic_private_world={'lantern':'green triangle'},teacher_program=dict(query='lookup lantern'))
    receipt=dict(scene_id='scene',world=scene['synthetic_private_world'],teacher_forced_request=True,
        native_student_dispatch_claimed=False,events=[dict(type='native_delegate',work_id='real-task',query='lookup lantern'),
         dict(type='worker_cancelled',work_id='real-task'),dict(type='cancellation_confirmed',targets=['real-task'])])
    verify_receipt(scene,receipt)
    wrong=copy.deepcopy(receipt);wrong['events'][-1]['targets']=[]
    with pytest.raises(ValueError,match='genuinely pending'):verify_receipt(scene,wrong)
    wrong=copy.deepcopy(receipt);wrong['events'][0]['query']='lookup cupboard'
    with pytest.raises(ValueError,match='Exact real'):verify_receipt(scene,wrong)


def test_result_from_wrong_world_cannot_ground_teacher_reply():
    scene=dict(id='scene',family='delegate',synthetic_private_world={'lantern':'green triangle'},teacher_program=dict(query='lookup lantern'))
    receipt=dict(scene_id='scene',world=scene['synthetic_private_world'],teacher_forced_request=True,
        native_student_dispatch_claimed=False,events=[dict(type='native_delegate',work_id='real-task',query='lookup lantern'),
         dict(type='worker_completed',work_id='real-task',result='Lookup lantern: red circle')])
    with pytest.raises(ValueError,match='matching result'):verify_receipt(scene,receipt)
