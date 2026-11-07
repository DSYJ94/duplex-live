# Independent teacher reviewer, not a new voice foundation

7 November 2026. The released native LLM failed the unchanged16-control semantic calibration:13/16, accepting a false database statement and two causally premature replies. Job8173621 ran45seconds, returnedFAILED1:0, and reviewedZERO96teachers. All25 output files were copied back and hash-verified; no subset or approval issued. See NATIVE-RETENTION96-LOCAL-REVIEWER-REJECTION-2026-11-07.json.

Use public Qwen3-32B as a separate, frozen reviewer on the existing Isambard award. The official [model card](https://huggingface.co/Qwen/Qwen3-32B) identifies32.8B parameters, Apache2.0 and Transformers>=4.51 support; our existing4.51.3 runtime avoids an upgrade. Its documented thinking-mode settings are temperature0.6/top_p0.95/top_k20, with a closing-think token separating the final answer. These are configuration facts, not a guarantee of reviewer accuracy.

Pin revision9216db5781bf21249d130ec9da846c4624c16137 and all26 selected files/65540296908bytes from the official public metadata. CPU-only staging on Isambard verifies every upstream weight SHA256 and records config/tokenizer hashes. No weights on the Mac, credentials, hosted inference, purchase, GCP resource or native optimizer update. Download uses token=False and built-in Transformers code, not downloaded Python.

After staging, one serial bounded GH200 review allocation may load the public model. Exactly the SAME16 controls and all96 complete synthetic TRAIN teachers, same semantic instructions and strict JSON parser. Fail calibration before judging any teacher; all96 must pass. Native replay/cache and product acceptance remain separate. Persist final verdicts/token counts only, not hidden reasoning. Do not change controls or rerun a failed reviewer just to obtain a pass. This is one stronger-reviewer intervention after observed13/16; native foundation/candidate remain unchanged.

This reviewer can also provide a later offline reasoning comparison, but is not yet a tested live backend. Current deployed voice model is still experimental646; neither model is product-adopted.
