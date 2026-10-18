"""Strict inference attachment of a separately restored full controller candidate."""
import json
from pathlib import Path


def metadata(folder, assets, expected_updates):
    from evals.moshi_model_download import sha256
    folder = Path(folder)
    proof = json.loads((folder / 'restore-proof.json').read_text())
    trained = json.loads((folder / 'training-proof.json').read_text())
    if (not proof['fresh_process_restore_passed'] or trained['optimizer_steps'] != expected_updates or trained['phase'] != 'full'
            or proof['pid'] == proof['training_pid'] or not trained['all_source_hashes_and_versions_unchanged']
            or proof['training_proof_sha256'] != sha256(folder / 'training-proof.json')
            or proof['training_state_sha256'] != sha256(folder / 'candidate-adapter/training-state.pt')
            or proof['adapter_weights_sha256'] != sha256(folder / 'candidate-adapter/adapter.safetensors')
            or trained['base_assets_sha256'] != sha256(assets)):
        raise ValueError('Requires exact full-candidate independent restoration proof')
    meta = json.loads((folder / 'candidate-adapter/adapter.json').read_text())
    if (meta['provenance']['settings_sha256'] != sha256(folder / 'settings.json')
            or meta['weights_sha256'] != proof['adapter_weights_sha256']
            or meta['base_assets_sha256'] != sha256(assets)
            or meta['implementation_sha256'] != sha256(Path(__file__).parents[1] / 'training/control/context_adapter.py')):
        raise ValueError('Candidate settings lineage differs')
    return dict(restore_proof_sha256=sha256(folder / 'restore-proof.json'), adapter_weights_sha256=proof['adapter_weights_sha256'],
                optimizer_steps=expected_updates, attachment_code_sha256=sha256(Path(__file__)),
                adapter_code_sha256=meta['implementation_sha256'], development_only=True, production_adopted=False)


def attach(executor, folder, assets, expected_updates):
    from training.control.context_adapter import install, restore
    from evals.moshi_model_download import sha256
    proof = metadata(folder, assets, expected_updates)
    adapters = install(executor.model)
    restore(Path(folder) / 'candidate-adapter', adapters, sha256(assets))
    for param in adapters.values():
        param.requires_grad_(False)
    executor.model.eval()
    if any(param.requires_grad for param in executor.model.parameters()):
        raise ValueError('Inference attachment left a trainable parameter')
    return proof
