# Native memory follow-up: one head-shape discriminator

The user-requested existing research agent checked the old whole-causal implementation against preserved evidence; it changed no files, jobs or provider state. The original known-failing `venus-control-v1-train-ordinary-0131` has161×4096 embeddings,21labels and21 native logit proofs. Its protected v8 file SHA256 is `0dddaf8b69f7b068b09de0427016346e0ce7cd251b29a7f9ce73977b04096907`.

The original zero-update whole-sequence proof reported minimum cosine0.999446 (required0.9999), target-CE delta0.099097 (maximum0.03), relativeL2 0.009093 (maximum0.01), argmax agreement1.0 (minimum0.95). The change at5939ae7 retained both native backbone feed segmentation and native language-head GEMM shapes; the subsequent native proof had zero logit/CE error. Therefore the unchanged old whole-causal implementation must not be used as an approved training shortcut.

Exact arithmetic makes causal whole-sequence and cached attention equivalent with matching positions, masks, window/reset policy and dropout. BF16 reduction shapes can differ. One causal discriminator is a whole-causal backbone with the ORIGINAL per-segment head GEMM shapes. It isolates a head-shape confound, rather than repeating the rejected implementation or changing precision/tolerances.

The implementation is forward-only and nonblocking. On the already loaded80GB base it checks ONLY the exact historical21 if that file already exists on the approved source image, against its native references and ALL FOUR UNCHANGED `training_input_proof` limits. Missing/mismatched data means NOTPERFORMED, never substituting the current304 namesake. That current v24 example has163embeddings/23labels/zero logit proofs and different tensors. Even a pass does not enable a new training path: full-gradient/memory, independent restore and broader native numerical/behavior/speech approval remain required. A failure retires this avenue for this phase, with no kernel/precision sweep.

Current larger-adapter training continues to use full native segmented BPTT. No optimizer update, target/fold edit, lowered threshold, frozen-prefix bypass or new payload follows from this probe.

Evidence: docs/NATIVE-VENUS-NATIVE-GRADIENT-CONTINUATION-2026-10-26.json; runs/venus-training-20261026/recovery-v7/copyback/native-proof-v7/whole-sequence-proof.json; runs/venus-training-20261026/sustain-v10/copyback/native-cache-v8/venus-control-v1-train-ordinary-0131.pt; training/venus/native_cache.py; training/venus/lora.py; primary code history5939ae7.
