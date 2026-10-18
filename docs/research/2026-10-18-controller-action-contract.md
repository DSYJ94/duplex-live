# Controller action contract experiment

Plain words: the first actual concurrent cancellation run7511107 took three controller calls to cancel:1.262s to output a JSON cancellation plan,1.154s to supply the actual cancellation arguments after a forced-tool retry, then1.765s to produce the final confirmation. This is observed latency from saved backend events, not a model-compute estimate. Ordinary answering took one2.020s call. GPU speech steps p950.03087s and actual streaming ASR p950.01835s were below the80ms frame period. No new training is indicated by this latency bottleneck.

The opt-in `required-action` protocol exposes three strict function choices: read an identified private field, cancel identified active lookups, or return an ordinary answer/clarifying question. The first call requires exactly one model-selected function. The ordinary-reply function has no side effect and produces no tool-execution receipt. Private reads and cancellations still pass the existing project-identification, revision, cancellation and actual-execution guards, followed by a separate structured final response. The original protocol remains the default and its source-pinned runs remain valid comparisons.

Official API documentation supports requiring a function call while allowing the model to choose among declared functions, and distinguishes function calling for application actions from structured text for user-facing responses. This design is an application-level experiment, not a documented guarantee of latency or semantic correctness:

- [Function calling: tool choice and strict schemas](https://developers.openai.com/api/docs/guides/function-calling)
- [Structured output versus function calling](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)

Local checks verify the ordinary branch uses exactly one request and creates no receipt, and reject missing actions or a reply action claiming the delegation route. Twenty-five targeted lifecycle, endpoint and playout checks pass. Actual semantic and latency evidence must come from the same paced conversations, retaining all output and receipt histories. No microphone, native routing, fresh holdout or full-system acceptance is established.
