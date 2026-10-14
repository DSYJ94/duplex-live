"""Exercise real native preprocessing without loading model weights or using a GPU."""
import argparse
import sys
from pathlib import Path
from evals.lychee_environment_check import configure_bf16
from evals.lychee_model_download import SOURCE_COMMIT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned", action="store_true")
    args = ap.parse_args()
    configure_bf16()
    import numpy as np
    import torch
    sys.path.insert(0, str(Path("/projects/u6si/duplex-live/vendor") / ("lychee-fd-"+SOURCE_COMMIT)))
    from lychee_fd.runtime.hf_v9_generation import SingleTurnGenerationFramework
    framework = object.__new__(SingleTurnGenerationFramework)
    framework.device = torch.device("cpu")
    framework.window_second = .4
    framework.align_audio_input = args.aligned
    framework.audio_token_id, framework.audio_pad_token_id = 151688, 158360
    result = framework.stepaudio_audio_prepreocess(np.zeros(6400, dtype=np.float32))
    count = len(result["input_ids"])
    print({"input_seconds": .4, "control_clock_hz": 25, "input_slots": count,
           "align_audio_input": args.aligned}, flush=True)
    assert count == 10, "Audio/control clock mismatch: 400ms must contain ten 40ms slots"


if __name__ == "__main__":
    main()
