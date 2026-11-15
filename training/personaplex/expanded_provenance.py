"""Reuse verified captures only with their original, unchanged capture code."""
import hashlib,re,subprocess
from pathlib import Path

CAPTURE_FILES=('training/personaplex/expanded_capture.py','training/personaplex/expanded_codec.py',
    'evals/native_control_engine.py','evals/native_control_cache.py',
    'evals/context_native_bf16_stream_diagnostic.py','evals/context_native_layout.py',
    'evals/context_execution.py','data/personaplex/expanded_source_export.py')


def capture_provenance(report,consumer,repository=None):
    producer=report['source'];repo=Path(repository)if repository is not None else Path(__file__).resolve().parents[2]
    if any(not isinstance(x,str)or re.fullmatch('[0-9a-f]{40}',x)is None for x in (producer,consumer)):
        raise ValueError('Exact original producer and current consumer commits required')
    hashes={}
    for name in CAPTURE_FILES:
        original=subprocess.check_output(['git','-C',str(repo),'show',producer+':'+name])
        if original!=(repo/name).read_bytes():raise ValueError('Capture code changed; independent requalification required: '+name)
        hashes[name]=hashlib.sha256(original).hexdigest()
    return dict(original_capture_source=producer,current_training_source=consumer,unchanged_capture_code_sha256=hashes)
