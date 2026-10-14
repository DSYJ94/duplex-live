"""Zero-initialized reference bridge into native PersonaPlex temporal inputs.

ARC vectors are not assumed to share PersonaPlex's learned embedding basis.
This bridge must be trained; its presence alone grants no reference-use ability.
"""
import torch


class ReferenceBridge(torch.nn.Module):
    def __init__(self, input_dim, output_dim, rank=64, *, device, dtype):
        super().__init__()
        self.down = torch.nn.Linear(input_dim, rank, bias=False, device=device, dtype=dtype)
        self.up = torch.nn.Linear(rank, output_dim, bias=False, device=device, dtype=dtype)
        torch.nn.init.zeros_(self.up.weight)
        # Stable storage is essential: captured CUDA graphs read this buffer.
        self.register_buffer("active", torch.zeros(1, 1, input_dim, device=device, dtype=dtype))

    def clear(self):
        self.active.zero_()

    def set_row(self, row):
        if row.ndim != 1 or row.numel() != self.active.shape[-1] or not torch.isfinite(row).all():
            raise ValueError("Reference row must be a finite vector of the declared size")
        self.active.copy_(row.detach().to(self.active)[None, None])

    def transform(self, reference):
        return self.up(self.down(reference))

    def temporal_input(self, module, inputs):
        if len(inputs) != 1 or inputs[0].shape[:2] != (1, 1):
            raise RuntimeError("Streaming reference bridge requires one native slot and frame")
        return (inputs[0] + self.transform(self.active).to(inputs[0]),)
