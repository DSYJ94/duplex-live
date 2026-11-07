import pytest
from training.venus.isambard_usage import summarize


def test_allocation_counts_failures_and_multiple_gpus_without_step_double_counting():
    rows='1|train|COMPLETED|0:0|3600|cpu=72,gres/gpu=1|start|end\n'
    rows+='1.batch|batch|COMPLETED|0:0|3600|gres/gpu=1|start|end\n'
    rows+='2|failed|FAILED|1:0|1800|gres/gpu=2|start|end\n'
    rows+='3|pending|PENDING|0:0|0||Unknown|Unknown\n'
    result=summarize(rows)
    assert result['allocated_gpu_hours']==2 and result['failed_or_cancelled_gpu_hours']==1
    assert result['top_level_jobs']==3 and result['official_portal_charged_node_hours'] is None


def test_duplicate_jobs_and_truncated_records_are_not_silently_summed():
    row='1|train|COMPLETED|0:0|3600|gres/gpu=1|start|end\n'
    with pytest.raises(ValueError):summarize(row+row)
    with pytest.raises(ValueError):summarize('1|train|COMPLETED')
