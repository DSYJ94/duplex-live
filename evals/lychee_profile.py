"""Bounded operator timing on the unchanged released incremental path.

Profile two arrived-audio chunks after warmup. Profiled timings are diagnostic,
not comparable latency benchmarks. No caching, sampling or model changes.
API: https://docs.pytorch.org/docs/2.9/profiler.html
"""
import json


def incremental_profile(out):
    import torch

    def completed(prof):
        prof.export_chrome_trace(str(out / "operator-trace.json"))
        averages = prof.key_averages(group_by_input_shape=True)
        rows = [{"op":e.key, "shapes":e.input_shapes, "calls":e.count,
            "self_cpu_us":e.self_cpu_time_total, "cpu_total_us":e.cpu_time_total,
            "self_device_us":e.self_device_time_total, "device_total_us":e.device_time_total}
            for e in averages]
        (out / "operator-profile.json").write_text(json.dumps({"schema":"dl-lychee-profile-v1",
            "wait_chunks":10, "warmup_chunks":1, "recorded_chunks":2,
            "profiled_latency_is_not_benchmark":True, "operators":rows}, indent=2)+"\n")

    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=10, warmup=1, active=2, repeat=1),
        on_trace_ready=completed, record_shapes=True, with_stack=False)
