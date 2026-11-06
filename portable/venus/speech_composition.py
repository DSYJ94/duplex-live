"""Opt-in source admission only; never a default or voice adoption claim."""
import json
import os
from pathlib import Path

REFERENCES=frozenset({
    '552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0',
    '33a4b2813cd3736d9e54f2d840385f0ec7530681b70b1d793271c72ae0336049',
    '40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674',
})


def require_authorization(reference=None,reference_sha256=None):
    message='Speech completion/action-channel composition requires separate native validation'
    lease_path=os.environ.get('DUPLEX_LIVE_LEASE_FILE')
    if not lease_path:raise ValueError(message)
    lease=json.loads(Path(lease_path).read_text())
    from training.venus.british_speech_guard import KEY,resource,sha
    if lease.get('resource_key')!=KEY:raise ValueError(message)
    resource(lease)
    if not reference or reference_sha256 not in REFERENCES:
        raise ValueError('Explicit approved English/British reference required; no default reference fallback')
    if sha(reference)!=reference_sha256:raise ValueError('Native audition reference bytes changed')
