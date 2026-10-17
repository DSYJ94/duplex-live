"""Weight an independently verified answer/request decision in supervised loss."""
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def validate(weight):
    if not math.isfinite(weight) or not 1.<=weight<=32.:
        raise ValueError('First-word decision weight must be finite and between one and32')


def weights(labels,weight=1.,response_boundary=False):
    import torch
    validate(weight)
    if labels.ndim!=1 or labels.dtype!=torch.int64 or not len(labels) or (labels<0).any():
        raise RuntimeError('Decision loss requires already masked valid text targets only')
    result=torch.where(labels==3,.5,1.)
    if response_boundary:
        if int(labels[0])<5:
            raise RuntimeError('Verified response boundary must begin with the actual first answer word')
        result[0]=weight
    return result


def policy(weight,boundaries):
    validate(weight)
    if weight==1.:return None
    if not any(boundaries):raise RuntimeError('Decision weighting has no verified response-boundary component')
    return {'first_word_weight':weight,'component_indices':[i for i,b in enumerate(boundaries) if b],
        'scope':'first supervised word at independently verified native RET prediction step only',
        'code_sha256':sha256(Path(__file__)),'past_loss_targets':0,'audio_loss_targets':0,'user_loss_targets':0}
