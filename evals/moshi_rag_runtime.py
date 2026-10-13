"""Plain words: discard superseded or closed-session background references.

Use as the pinned upstream server's entry point after its model/tokenizer
preflight. This guards delivery; learned task dispatch and natural-language
cancellation remain evaluation requirements, not claims made by this adapter.
"""
import contextvars
import torch

from moshi.inference_utils.channel import Channel
from moshi.inference_utils.rag_manager import RAGManager
from moshi.inference_utils.utils import get_conditioning_remote_async


reference_revision = contextvars.ContextVar("moshi_reference_revision", default=None)


class VersionedRAGManager(RAGManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.revision = 0

    async def trigger(self, task_group, wait_steps=0, handle_reference_fn=None, context_provider=None):
        self.revision += 1
        revision = self.revision

        async def guarded_callback(text, label):
            if revision != self.revision:
                return
            token = reference_revision.set(revision)
            try:
                if handle_reference_fn is not None:
                    await handle_reference_fn(text, label)
            finally:
                reference_revision.reset(token)

        await super().trigger(task_group=task_group, wait_steps=wait_steps,
                              handle_reference_fn=guarded_callback, context_provider=context_provider)

    def cancel_pending(self):
        self.revision += 1
        super().cancel_pending()


class SafeChannel(Channel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.rag_manager
        self.rag_manager = VersionedRAGManager(
            reference_generator=previous.reference_generator, rag_timeout=previous.rag_timeout,
            max_tokens=previous.max_tokens, gt_reference_text=previous.gt_reference_text)

    def _reference_is_current(self, revision):
        return (revision == self.rag_manager.revision
                and 0 <= self.slot_idx < self.server.batch_size
                and self.server.slots[self.slot_idx] is self)

    async def _async_update_reference(self, text):
        revision = reference_revision.get()
        if revision is None:
            raise RuntimeError("Reference delivery has no task revision")
        if not self._reference_is_current(revision):
            return
        tensor = await get_conditioning_remote_async(text=text, encoder_url=self.server.reference_encoder_url)
        # A newer retrieval, explicit reset/cancel or slot reuse can happen
        # during encoding. Check again after the actual asynchronous boundary.
        if not self._reference_is_current(revision):
            self._log.info("Discarded superseded reference encoding, revision=%s", revision)
            return
        dimension = self.server.runner.lm_gen.lm_model.dim
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] == 0 or tensor.shape[2] != dimension:
            raise ValueError("Reference encoder returned the wrong shape")
        if not torch.isfinite(tensor).all():
            raise ValueError("Reference encoder returned nonfinite values")
        per_slot = [None] * self.server.batch_size
        per_slot[self.slot_idx] = tensor.squeeze(0)
        self.server.runner.lm_gen.update_streaming_sum_tensors(per_slot)


if __name__ == "__main__":
    from moshi import server
    server.Channel = SafeChannel
    server.main()
