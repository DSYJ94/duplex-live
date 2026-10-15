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
    def __init__(self, model, control_feedback=False):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.replaced = install_adapters(model)
        self.bridge = GatedReferenceBridge(4096, model.dim, rank=64, device=model.device, dtype=torch.float32)
        self.policy = nn.Linear(model.dim, 3, device=model.device, dtype=torch.float32)
        nn.init.zeros_(self.policy.weight)
        with torch.no_grad():
            self.policy.bias.copy_(torch.tensor([2., 0., 0.], device=model.device))
        self.control = None
        if control_feedback:
            from evals.personaplex_control import CONTROL_STATES
            self.control = nn.Embedding(len(CONTROL_STATES), model.dim, padding_idx=0,
                _weight=torch.zeros(len(CONTROL_STATES), model.dim, device=model.device, dtype=torch.float32))
            self.register_buffer("active_control", torch.zeros(1, 1, device=model.device, dtype=torch.long))

    def learned_parameters(self):
        return {**{"language."+n:p for n,p in adapter_parameters(self.model).items()},
                **{"bridge."+n:p for n,p in self.bridge.named_parameters()},
                **{"policy."+n:p for n,p in self.policy.named_parameters()},
                **({"control.weight":self.control.weight} if self.control is not None else {})}

    def set_control(self, state):
        if self.control is None or not 0 <= state < self.control.num_embeddings:
            raise ValueError("Unknown or disabled native client feedback")
        self.active_control.fill_(state)

    def temporal_input(self, module, inputs):
        embedded = self.bridge.temporal_input(module, inputs)[0]
        if self.control is not None:
            embedded = embedded + self.control(self.active_control).to(embedded)
        return (embedded,)

    def load_learned(self, state):
        params = self.learned_parameters()
        if set(params) != set(state):
            raise RuntimeError("Missing or unexpected native adapter parameters")
        with torch.no_grad():
            for name, parameter in params.items():
                if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                    raise RuntimeError("Invalid learned parameter: "+name)
                parameter.copy_(state[name])

    def forward(self, inputs, reference, targets, prefix_frames, control_states=None):
        embedded = self.model.embed_codes(inputs)
        embedded = embedded + self.bridge.transform(reference).to(embedded)
        if self.control is not None:
            if control_states is None or control_states.shape != inputs[:, 0].shape:
                raise ValueError("Missing or misaligned causal client feedback")
            embedded = embedded + self.control(control_states).to(embedded)
        elif control_states is not None:
            raise ValueError("Feedback supplied to an unconfigured native model")
        hidden, text = self.model.forward_embeddings(embedded)
        hidden, text = hidden[:, prefix_frames:], text[:, :, prefix_frames:]
        return {"text": text, "audio": native_audio_logits(self.model, targets, hidden),
                "policy": self.policy(hidden.float())}
