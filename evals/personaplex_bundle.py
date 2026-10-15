"""A native PersonaPlex language adapter, reference bridge and foreground head.

The original depth decoder, Mimi and QKV weights stay frozen. The head predicts
idle, delegate or cancel from native temporal hidden states, before any worker
result or current teacher word is available. Routing still needs free-run tests.
"""
import torch
from torch import nn
from evals.personaplex_adapters import install_adapters, adapter_parameters, GatedReferenceBridge
from evals.personaplex_depth import native_audio_logits


class PersonaBundle(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.replaced = install_adapters(model)
        self.bridge = GatedReferenceBridge(4096, model.dim, rank=64, device=model.device, dtype=torch.float32)
        self.policy = nn.Linear(model.dim, 3, device=model.device, dtype=torch.float32)
        nn.init.zeros_(self.policy.weight)
        with torch.no_grad():
            self.policy.bias.copy_(torch.tensor([2., 0., 0.], device=model.device))

    def learned_parameters(self):
        return {**{"language."+n:p for n,p in adapter_parameters(self.model).items()},
                **{"bridge."+n:p for n,p in self.bridge.named_parameters()},
                **{"policy."+n:p for n,p in self.policy.named_parameters()}}

    def load_learned(self, state):
        params = self.learned_parameters()
        if set(params) != set(state):
            raise RuntimeError("Missing or unexpected native adapter parameters")
        with torch.no_grad():
            for name, parameter in params.items():
                if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                    raise RuntimeError("Invalid learned parameter: "+name)
                parameter.copy_(state[name])

    def forward(self, inputs, reference, targets, prefix_frames):
        embedded = self.model.embed_codes(inputs)
        embedded = embedded + self.bridge.transform(reference).to(embedded)
        hidden, text = self.model.forward_embeddings(embedded)
        hidden, text = hidden[:, prefix_frames:], text[:, :, prefix_frames:]
        return {"text": text, "audio": native_audio_logits(self.model, targets, hidden),
                "policy": self.policy(hidden.float())}
