"""Load our verified adapter format before native Engine builds CUDA graphs."""
from contextlib import contextmanager
import json
from evals.moshi_model_download import sha256


@contextmanager
def before_engine(engine_module, proof, disabled=False):
    import torch
    from safetensors.torch import load_file
    from evals.context_spanning_download import SOURCE_COMMIT
    from evals.personaplex_adapters import install_adapters, adapter_parameters
    if proof is None:
        if disabled: raise RuntimeError('Disabled comparison requires a verified adapter proof')
        yield None
        return
    report = json.loads((proof / 'report.json').read_text())
    reload = json.loads((proof / 'reload-report.json').read_text())
    digest = sha256(proof / 'adapters.safetensors')
    if (report['schema'] != 'dl-context-adapter-proof-v1' or not report['passed']
            or not report['frozen_weights_unchanged'] or not report['zero_adapter_all_valid_logits_exact']
            or not reload['passed'] or not reload['fresh_process'] or not reload['all_valid_logits_exact']
            or report['source_commit'] != SOURCE_COMMIT or report['checkpoint_sha256'] != digest
            or reload['checkpoint_sha256'] != digest or report['rank'] != 8 or report['scaling'] != 2.
            or report['voice_sha256'] != sha256(proof / 'voice.pt')
            or report['data_sha256'] != sha256(proof / 'data.json')):
        raise RuntimeError('Unverified or changed Context adapter proof')
    original = engine_module.load_model
    calls = []
    def load(*args, **kwargs):
        lm, mimi, spm = original(*args, **kwargs)
        if calls: raise RuntimeError('Unexpected repeated model loading')
        lm.requires_grad_(False)
        if not disabled:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                names = install_adapters(lm, last_layers=report['last_layers'])
            state = load_file(proof / 'adapters.safetensors', device=str(lm.device))
            learned = adapter_parameters(lm)
            if names != report['adapter_modules'] or set(state) != set(learned):
                raise RuntimeError('Cold adapter module/key coverage differs')
            with torch.no_grad():
                for n, p in learned.items(): p.copy_(state[n])
            lm.requires_grad_(False).eval()
        calls.append(True)
        return lm, mimi, spm
    engine_module.load_model = load
    evidence = {'proof': str(proof), 'checkpoint_sha256': digest, 'disabled': disabled,
        'installed_before_native_graphs': not disabled, 'voice_codes_sha256': report['voice_sha256'],
        'voice_reference_sha256': report['voice_reference_sha256'], 'quality_qualified': False}
    try:
        yield evidence
        if not calls: raise RuntimeError('Native engine did not invoke verified model loader')
    finally:
        engine_module.load_model = original
