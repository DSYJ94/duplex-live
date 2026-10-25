"""Pure authorization/budget checks; no model import or provider mutation."""
import hashlib
import json
import math
from pathlib import Path
import time


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


PROPOSAL_SHA = '2f6ee724a101d69be0f79261b5f033ee06bf33358df6f9606e3cde7bf2c4335f'
ORIGINAL_GRANT_SHA = '97c8d43fe7be11595e7d00775af4e1f2c7666a469041c720bf7586ee8a020322'
RESOURCE_ID = '4443302983961393208'


def authorization(path):
    """Read a NEW direct-owner authorization; the old zero-update grant fails."""
    path = Path(path)
    record = json.loads(path.read_text())
    expected = dict(schema='dl-native-changed-rate-eight-owner-authorization-v1',
        status='APPROVED', approval_source='Direct human reply in active LOCAL chat',
        proposal_sha256=PROPOSAL_SHA, original_grant_sha256=ORIGINAL_GRANT_SHA,
        project='aurium-duplex-live-20261024', additional_optimizer_updates=8,
        initial_global_adam_step=528, final_global_adam_step=536,
        learning_rate=1e-6, maximum_additional_gcp_usage_usd=75,
        cumulative_gcp_usage_cap_usd=175, additional_gpu_hours=9,
        cumulative_gpu_hours=21, maximum_concurrent_gpus=1,
        rejected_parent_bounded_exploration_approved=True,
        new_credit_purchase=False, new_vm_or_storage=False,
        further_extension_authorized=False, model_adoption_authorized=False)
    if (any(record.get(k) != v or type(record.get(k)) is not type(v)
            for k, v in expected.items())
            or type(record.get('owner_reply')) is not str or not record['owner_reply'].strip()):
        raise ValueError('Exact separate changed-rate/update/budget owner authorization required')
    return dict(record=record, sha256=_sha(path))


def check_guard(guard, approved):
    raw = guard()
    from portable.context.native_saved_stage_runtime import guard_receipt
    checked = guard_receipt(lambda: raw)
    now = time.time()
    numeric = ('native_start_unix', 'maximum_new_gpu_seconds', 'hourly_price_usd',
               'prior_gpu_seconds_upper', 'prior_scope_cost_upper_usd')
    if (any(type(raw.get(k)) not in (int, float) or not math.isfinite(raw[k]) for k in numeric)
            or checked['provider'] != 'gcp' or checked['resource_id'] != RESOURCE_ID
            or checked['authorization_sha256'] != approved['sha256']
            or raw.get('optimizer_updates_authorized') != 8
            or raw.get('parent_global_adam_step') != 528
            or raw.get('final_global_adam_step_limit') != 536
            or not 0 < raw['native_start_unix'] <= now
            or not 0 < raw['maximum_new_gpu_seconds'] <= 32400
            or checked['deadline_unix'] > raw['native_start_unix'] + raw['maximum_new_gpu_seconds']
            or raw['prior_gpu_seconds_upper'] < 40351
            or raw['prior_gpu_seconds_upper'] + raw['maximum_new_gpu_seconds'] > 75600
            or raw['prior_scope_cost_upper_usd'] < 91.98448055555555
            or checked['planned_maximum_cost_usd'] > 75
            or checked['remaining_budget_usd'] > 75
            or not 0 < raw['hourly_price_usd'] <= 7.25
            or checked['planned_maximum_cost_usd'] < raw['hourly_price_usd'] * raw['maximum_new_gpu_seconds'] / 3600 + 6.5
            or checked['planned_maximum_cost_usd'] + raw['prior_scope_cost_upper_usd'] > 175):
        raise ValueError('Fresh unchanged provider/budget/update bounds required')
    return dict(checked, **{k:raw[k] for k in numeric},
                optimizer_updates_authorized=raw['optimizer_updates_authorized'],
                parent_global_adam_step=raw['parent_global_adam_step'],
                final_global_adam_step_limit=raw['final_global_adam_step_limit'])

