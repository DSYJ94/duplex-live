"""Plain words: an encoded old result must not overwrite a newer task result.

Run in the pinned MoshiRAG environment. Uses the actual channel callback;
only the remote encoder and GPU update sink are controlled fixtures.
"""
import asyncio
import logging
import unittest
from types import SimpleNamespace

import torch
from moshi.inference_utils.channel import Channel
import moshi.inference_utils.channel as upstream


class ReferenceRaceTest(unittest.IsolatedAsyncioTestCase):
    async def test_old_encoding_cannot_overwrite_new_result(self):
        try:
            from evals.moshi_rag_runtime import SafeChannel, VersionedRAGManager
        except ImportError:
            SafeChannel = Channel
            VersionedRAGManager = None
        channel = object.__new__(SafeChannel)
        old_ready = asyncio.Event()
        old_started = asyncio.Event()
        applied = []
        channel.slot_idx = 0
        channel._log = logging.getLogger("reference-race-test")
        channel.server = SimpleNamespace(batch_size=1, reference_encoder_url="local-test",
            slots=[channel], runner=SimpleNamespace(lm_gen=SimpleNamespace(lm_model=SimpleNamespace(dim=4), update_streaming_sum_tensors=
                                  lambda values: applied.append(float(values[0][0, 0])))))
        async def encoder(text, **kwargs):
            if text == "old":
                old_started.set()
                await old_ready.wait()
                return torch.full((1, 1, 4), 1.)
            return torch.full((1, 1, 4), 2.)
        original = upstream.get_conditioning_remote_async
        upstream.get_conditioning_remote_async = encoder
        self.addCleanup(setattr, upstream, "get_conditioning_remote_async", original)
        if VersionedRAGManager is not None:
            import evals.moshi_rag_runtime as guarded
            original_guarded = guarded.get_conditioning_remote_async
            guarded.get_conditioning_remote_async = encoder
            self.addCleanup(setattr, guarded, "get_conditioning_remote_async", original_guarded)
            channel.rag_manager = VersionedRAGManager(reference_generator=None)
            channel.rag_manager.revision = 1
            token = guarded.reference_revision.set(1)
            old = asyncio.create_task(channel._async_update_reference("old"))
            guarded.reference_revision.reset(token)
            await old_started.wait()
            channel.rag_manager.revision = 2
            token = guarded.reference_revision.set(2)
            await channel._async_update_reference("new")
            guarded.reference_revision.reset(token)
        else:
            old = asyncio.create_task(channel._async_update_reference("old"))
            await old_started.wait()
            await channel._async_update_reference("new")
        old_ready.set()
        await old
        self.assertEqual(applied, [2.], "Old encoding overwrote the current result")


if __name__ == "__main__":
    unittest.main()
