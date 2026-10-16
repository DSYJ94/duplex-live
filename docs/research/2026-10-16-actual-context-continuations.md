# Actual Context Spanning histories and approved future targets

Plain words: build the next training component from what the released model
actually heard and said before a real result arrived. Restore that history,
verify a short ordinary continuation, then supply independently approved future
targets. This is a proposed implementation gate. The new v2 snapshots are
still being checked; neither exact restart nor future-target approval is a
proven result.

The source under review is Context Spanning commit
`79df302fa2820cae10c61bb2b7f302c52edb6ed2`. Repository
[`Recorder`](../../evals/context_student_history.py) v2 records actual delayed
temporal inputs plus generator cache, provided flags, offset, CPU/CUDA RNG and
pending semantic readout state after each complete real `Engine.step`.
Old v1 records remain useful raw conditioning evidence but do not contain the
new restart state. Keep both versions immutable and distinguish their schemas.

## Boundary and provenance

Choose an explicit quiet boundary after the required actual receipt has been
injected and consumed. Record its real user sample index, input-availability
time, actual receipt/injection events and cumulative virtual frame count.
Preserve all preceding input, including refusals, wrong guesses and intervening
conversation. Those model outputs are observed conditioning, not approved gold.
Do not rewrite them to make the future target easier.

Require TRAIN membership, current source approvals, model/voice/source hashes,
collector commit and artifact hashes. Select only a future interval with
verified quiet user input if this component is restricted to quiet delivery.
Later real speech or a second result/cancellation cannot be silently dropped.
Mask every past label, supplied span and other non-target position. The initial
component supervises future text only; it does not qualify learned retrieval,
turn timing or speech generation.

## Replay consumed inputs, then restore the pending state

Native [`LMGen.prepare_step_input`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/models/lm.py)
forwards the previous cache column and then advances the generator offset.
Consequently a checked post-step snapshot has offset `O = M + 1`, where `M`
is the number of temporal input columns already consumed. Its newest sampled
cache column has not yet been forwarded. Replay exactly the captured `M`
columns into fresh temporal KV, and restore the small generator state without
forwarding that pending column twice.

Keep the captured coordinates: these inputs are **already delayed** and already
contain actual prefix/span behavior. Use native `forward_codes`; passing them
to `forward_train` would apply another delay transformation. Preserve the
original recorded call boundaries where possible, rather than assuming one
large prefill is numerically identical to the original prefix, live and span
calls. Assert every temporal attention position/end offset equals `M` before
continuation. Start with contexts below the pinned 3,000-frame ring capacity;
wrapping introduces an additional attention-window parity question.

The [`span prefill`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/model/context_span_block.py)
updates KV with actual delayed columns while separately advancing the generator
ring. Reconstructing its text from a receipt and reinjecting it would not be
the same as replaying the recorded inputs. Do not add a second persona or span
on top of the saved history.

## Preserve pending-token ownership

Restore the complete cache and provided mask, including unprovided slots;
never convert a sampled prediction into an observed target by setting its flag.
The v2 invariant at a complete real step is zero provided pending assistant
acoustics and seven provided pending user acoustics. That is an ownership
assertion, not a label-approval decision.

At the first forced future step, current gold text and assistant semantic codes
occupy the current target position, while the current gold residual acoustic
codes are scheduled one position later. Pending assistant residuals from the
preceding real frame remain unprovided and are sampled by native LMGen. They
belong to the past/transition and stay conditioning-only; do not obtain them
from later recorded outputs or substitute previous teacher audio. Use native
forcing arguments to retain this delay behavior.

The returned token bundle is a delayed readout. Future text labels must come
from the current approved forcing arguments or a cloned current target before
mutation, not from returned `tok` or `Engine.text_token`. Record target position,
source and ownership explicitly. Native [`Engine.step`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/model/engine.py)
also adjusts a pending semantic code for PCM readout after a span; that output
detail must not be mistaken for language-model input state.

## Graph and RNG restoration

`Engine.reset()` stops and restarts streams, creating fresh generator states
and graph wrappers. In contrast, existing states' `reset_streaming()` resets
them in place. The native
[`streaming implementation`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/modules/streaming.py)
and [`graph helper`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/utils/compile.py)
make this distinction material: captured graphs retain references to tensor
storage. Do not deserialize old graph objects or replace fresh graph wrappers
with objects bound to the collector's KV memory.

If explicitly warming the replay process, use this order:

1. Create fresh LMGen and warm its main/depth graph paths.
2. Reset existing streaming state in place, retaining graph wrappers and KV
   tensor storage. Verify their identities/pointers remain stable.
3. Replay the exact recorded raw temporal inputs into that same KV storage.
4. Copy the saved generator cache, provided flags and offset into the current
   state; verify the model's initial tokens match.
5. Restore CPU/CUDA RNG last, after construction, warmup and replay. Continue
   with the collector's sampler configuration and execution settings.

The helper executes a warmup function once or captures and replays once; a cold
call alone does not prove extra sampling or loss of the restored seed. Fresh
and mature graph paths still require an actual restart comparison. Do not
reseed after the first real continuation step, which would reuse draws. The
native sampler draws text and audio samples even for supplied targets, so
switching to greedy decoding or skipping forced-channel sampling changes the
continuation's RNG behavior.
[Sampler source](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/utils/sampling.py).

## Required gate before approved forcing

First replay one or two **unforced** next frames from the original run, with no
intervening unaccounted result event. Supply actual streamed user codec outputs
and compare the resulting generator cache, provided masks, offsets, RNG and
sampled tokens against the next v2 snapshots. Record raw input equality and
positions separately from numerical/logit and sampled-output equality. A
batch-replay rounding difference is not evidence of exact stochastic restart.

For the later gold component, use actual streamed user codec inputs from the
chosen future interval, or separately prove exact replay of its streaming
encoder. Generic silence tokens, a newly zeroed waveform and an offline Mimi
cache need not equal the original streaming encoder state. If extracting
future user codes from delayed records, preserve the real/virtual frame map
and span boundaries; do not globally reverse delays across inserted spans.

Only after restart checks pass should the assembler force approved future
words and assistant codes. Retain their independent source, transcript/audio
approval, measured timing and actual receipt causality. Revalidate that all
past and initially sampled residual labels remain excluded. A passing result
would establish a bounded data/conditioning component, not live task success.

Generator state and RNG do not restore Mimi encoder/decoder streaming buffers.
PCM continuation needs those states or separately checked codec replay, and
forced gold audio is not evidence of sampled voice quality. Actual microphone,
speaker echo cancellation, playback interruption and end-to-end latency remain
separate unproven gates.

## Restart gate result and falsifiable diagnosis

Plain words: strict restart 7464207 fails. The reviewed experiment record says
both histories retain exact provided flags, clocks, CPU/CUDA RNG and delayed
returned text over two steps, but sampled assistant acoustic cache values
diverge. Cold-graph control 7464272 also fails, rejecting graph warmup alone as
the fix. Actual histories and future gold targets remain unapproved. These
results do not identify a root cause.

Official Torch 2.9
[`CUDAGeneratorImpl`](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)
serializes seed and Philox offset. `set_state` updates the existing generator
state; graph replay fills its seed/offset device tensors from that state and
advances the recorded graph increment.
[`CUDAGraph::replay`](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/cuda/CUDAGraph.cpp)
calls this replay prologue before launching the graph. This argues against
assuming an independent, omitted sampling RNG. The actual NGC Torch fork must
still be distinguished from official release source. Matching global RNG
establishes draw accounting, not identical distributions or category ordering.

The next diagnostic saves each actual native temporal call's final hidden
vector and text logits outside the graph, then compares replay before depth
sampling. Clone captures immediately: `CUDAGraphed` reuses its output buffers,
so keeping tensor references would corrupt earlier evidence. If the hidden
vector already differs, investigate temporal KV values, execution settings and
numerics before blaming depth RNG. Exact token inputs and positions alone do
not establish exact floating-point KV contents.

If hidden vectors match, force identical preceding depth codes and compare the
first differing head's logits and top-k indices. Torch's
[`topk` documentation](https://docs.pytorch.org/docs/2.9/generated/torch.topk.html)
does not promise stable indices for ties. If those inputs also match, repeat
fixed-probability sampling on the same captured graph with identical restored
RNG. Different samples would support a sampler/graph-state issue; reproducible
samples would direct attention back to the upstream inputs or depth graph.
Native depth inference creates a separate streaming scope per frame; the
review has not identified another persistent depth state that explains the
observed mismatch. Keep that hypothesis open until these checks distinguish it.
