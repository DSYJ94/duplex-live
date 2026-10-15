"""Experimental causal worker-text memory in the frozen native embedding basis.

Each completed native subword contributes one extra temporal input embedding.
The model continues sampling its own speech/text, and user audio is unchanged.
No learned ARC alignment or acoustic/text teacher forcing is claimed here.
"""
from collections import deque


class NativeTextMemory:
    def __init__(self, embedding, encode):
        import torch
        self.embedding, self.encode = embedding, encode
        self.active = torch.zeros(1, 1, embedding.weight.shape[-1],
            device=embedding.weight.device, dtype=embedding.weight.dtype)
        self.queue = deque()

    def append(self, task_id, text, receipt):
        from moshi.offline import wrap_with_system_tags
        tokens = self.encode(wrap_with_system_tags("Verified background result: "+text))
        if not tokens or len(tokens) > 512 or any(t < 5 for t in tokens):
            raise ValueError("Invalid native worker memory tokens")
        self.queue.extend((task_id, token, receipt) for token in tokens)

    def clear(self):
        self.active.zero_()

    def advance(self, deliverable):
        import torch
        self.clear()
        while self.queue:
            task_id, token, receipt = self.queue.popleft()
            if receipt is not None and not deliverable(receipt):
                continue
            with torch.inference_mode():
                ids = torch.tensor([[token]], dtype=torch.long, device=self.active.device)
                self.active.copy_(self.embedding(ids))
            return task_id, token
        return None

    def temporal_input(self, module, inputs):
        if len(inputs) != 1 or inputs[0].shape[:2] != (1,1):
            raise RuntimeError("Native text memory requires one streamed frame")
        return (inputs[0]+self.active,)
