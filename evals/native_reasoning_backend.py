"""One asynchronous actual Responses call per native reasoning request.

Only the lifecycle builds the arrived-USER payload and applies semantic actions.
HTTP runs outside its lock. All request/body/identity clocks are retained, with
no redirect, retry, fallback, API-key logging or model-output substitution.
This opt-in component is not a teacher collector or a native routing approval.
"""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from evals.native_control_teacher import atomic_json, local_key
from evals.native_reasoning_lifecycle import canonical_hash
from evals.recovery_transport import ENDPOINT


async def request(world, identity, model, out, env_file=None, *, post=None, started=None):
    """Caller owns source qualification, collection budget, pacing and final drain.

    Cancelling this await cannot abort remote computation. A late thread result
    is still saved, while the lifecycle retires delivery and fails the episode.
    Normal user cancellation uses the semantic ledger rather than task.cancel().
    """
    actual = post is None
    if actual:
        import requests
        post = requests.post; key = local_key(env_file)
    else:
        key = 'fixture-only-no-credential'
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    payload = await world.start_backend(identity, model)
    atomic_json(out/'request.json', payload)
    trace = dict(schema='dl-native-reasoning-backend-transport-v1', task_id=identity,
        request_payload_sha256=canonical_hash(payload), actual_provider_transport=actual,
        client_request_id='duplex-reasoning-'+uuid.uuid4().hex, endpoint=ENDPOINT,
        model=model, maximum_attempts=1, automatic_retry=False, redirects_allowed=False,
        provider_credential_recorded_or_transferred=False, teacher_or_training_approved=False,
        native_request_selection_proven=False, phase='prepared')
    atomic_json(out/'transport.json', trace); loop = asyncio.get_running_loop()

    def run_http():
        response = None
        trace.update(phase='http_call_started', started_utc=datetime.now(timezone.utc).isoformat(),
                     http_call_started_monotonic=time.monotonic())
        atomic_json(out/'transport.json', trace)
        if started is not None: loop.call_soon_threadsafe(started.set)
        try:
            response = post(ENDPOINT, headers={'Authorization':'Bearer '+key,
                'X-Client-Request-Id':trace['client_request_id']}, json=payload,
                stream=True, timeout=(10,90), allow_redirects=False)
            trace.update(http_status=response.status_code, provider_request_id=response.headers.get('x-request-id'))
            partial = out/'response.partial'; size = 0
            with partial.open('xb') as writer:
                for block in response.iter_content(65536):
                    if size+len(block)>2*1024*1024:
                        raise ValueError('Bounded provider body exceeded; partial bytes preserved')
                    writer.write(block); writer.flush(); size += len(block)
            target = out/'response.json'; partial.rename(target)
            data = target.read_bytes()
            trace.update(response_bytes=size, response_sha256=hashlib.sha256(data).hexdigest(),
                         http_call_finished_monotonic=time.monotonic())
            if response.status_code!=200:
                raise RuntimeError('Actual reasoning HTTP failure; body preserved')
            value = json.loads(data)
            if not isinstance(value,dict): raise ValueError('Provider JSON object required')
            trace.update(phase='provider_returned_unvalidated', response_id=value.get('id'))
            return value
        except BaseException as error:
            trace.update(phase='transport_failed_preserved', error_type=type(error).__name__,
                         http_call_finished_monotonic=time.monotonic())
            raise
        finally:
            if response is not None: response.close()
            atomic_json(out/'transport.json', trace)

    try:
        response = await asyncio.to_thread(run_http)
    except BaseException as error:
        await world.fail_backend(identity, type(error).__name__)
        raise
    try:
        receipt = await world.complete_backend(identity, response, actual_transport=actual)
    except BaseException as error:
        trace.update(phase='semantic_action_failed_preserved', error_type=type(error).__name__)
        atomic_json(out/'transport.json', trace); raise
    trace.update(phase='stale_completion_preserved' if receipt is None else 'semantic_action_committed',
                 response_may_be_delivered=receipt is not None)
    atomic_json(out/'transport.json', trace)
    return receipt
