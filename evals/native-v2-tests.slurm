#!/bin/bash
#SBATCH --job-name=dl-native-v2-cpu-tests
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-native-v2-cpu-tests-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -c '
import json,sys,unittest
sys.path.insert(0,json.load(open("/projects/u6si/duplex-live/runs/context-spanning-assets-7454205/manifest.json"))["source"])
names = ["test_context_native_stream_training", "test_context_native_stream_gradient", "test_native_fresh_session", "test_native_precision_chain", "test_context_native_bf16_stream_diagnostic", "test_native_precision_diagnostic", "test_context_native_complete_batch", "test_context_native_parent_adam", "test_context_native_audio49_gradient_probe", "test_native_ancestor_diagnostic", "test_native_audio49_chain", "test_context_native_audio49_training", "test_context_native_audio49_data", "test_context_native_audio49_loss", "test_recovery_worker", "test_recovery_replacement", "test_duplex_dispatch", "test_context_backend_reference", "test_context_backend_failure", "test_native_conditioning_probe", "test_native_persona_diagnostic", "test_native_seen_retention_diagnostic", "test_context_native_learning_curve", "test_context_native_exploration", "test_context_native_v2_teacher_objective", "test_context_native_gradient_audit", "test_context_native_retention_drift", "test_context_warm_ordinary_retention", "test_context_warm_ordinary_clock", "test_native_text_sampling_diagnostic", "test_context_native_teacher_coverage", "test_context_native_extension_review", "test_context_rollout_conditioning", "test_native_v2_source_repair_v2",
         "test_native_v2_teacher", "test_native_fresh_inputs", "test_native_fresh_inputs_v2", "test_native_fresh_asr_control", "test_native_fresh_asr_control_v2", "test_native_fresh_source_profile_probe", "test_native_fresh_source_reapproval", "test_native_fresh_probe", "test_native_fresh_review", "test_recovery_teacher_quality", "test_context_training_state",
         "test_context_native_teacher_data", "test_context_native_teacher_objective",
         "test_context_teacher_native_approval", "test_context_teacher_engine_inputs", "test_context_teacher_text_balance",
         "test_context_native_joint_loss", "test_measured_question_completion", "test_orthographic_agreement", "test_native_old25_timing_repair",
         "test_native_old25_timing_cache", "test_native_v2_teacher_repair", "test_speech_asr_window", "test_speech_silent_failure"]
suite=unittest.defaultTestLoader.loadTestsFromNames(["evals.harness."+n for n in names])
result=unittest.TextTestRunner(verbosity=2).run(suite)
import os
if os.environ.get("DL_CPU_TEST_REPORT"):
 import torch
 from pathlib import Path
 expected=int(os.environ["DL_CPU_EXPECTED_TESTS"])
 r=dict(schema="dl-native-stream-cpu-suite-v1",source=os.environ["DL_TASK_COMMIT"],allocation_job=os.environ["SLURM_JOB_ID"],tests_run=result.testsRun,expected_tests=expected,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful() and result.testsRun==expected and not torch.cuda.is_initialized(),cuda_initialized=torch.cuda.is_initialized())
 with Path(os.environ["DL_CPU_TEST_REPORT"]).open("x") as f:json.dump(r,f,indent=2)
 if not r["passed"]:raise SystemExit(1)
raise SystemExit(not result.wasSuccessful())
'
