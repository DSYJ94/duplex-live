"""Strict inference attachment of a separately restored full controller candidate."""
import json
from pathlib import Path


def attach(executor, folder, assets, expected_updates):
    from evals.moshi_model_download import sha256
    from training.control.context_adapter import install, restore
    folder = Path(folder)
    proof = json.loads((folder / 'restore-proof.json').read_text())
    trained = json.loads((folder / 'training-proof.json').read_text())
    if (not proof['fresh_process_restore_passed'] or trained['optimizer_steps'] != expected_updates or trained['phase'] != 'full'
            or proof['pid'] == proof['training_pid'] or not trained['all_source_hashes_and_versions_unchanged']
            or proof['training_proof_sha256'] != sha256(folder / 'training-proof.json')
            or proof['training_state_sha256'] != sha256(folder / 'candidate-adapter/training-state.pt')
            or proof['adapter_weights_sha256'] != sha256(folder / 'candidate-adapter/adapter.safetensors')):
        raise ValueError('Requires exact full-candidate independent restoration proof')
    adapters = install(executor.model)
    meta = restore(folder / 'candidate-adapter', adapters, sha256(assets))
    if meta['provenance']['settings_sha256'] != sha256(folder / 'settings.json'):
        raise ValueError('Candidate settings lineage differs')
    return dict(restore_proof_sha256=sha256(folder / 'restore-proof.json'), adapter_weights_sha256=proof['adapter_weights_sha256'], optimizer_steps=expected_updates)
