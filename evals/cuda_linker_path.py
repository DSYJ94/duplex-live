"""Plain words: expose the actually loaded CUDA driver to Triton's linker.

Apptainer --nv can load a driver without putting libcuda.so in ldconfig's
cache. Create job-local linker symlinks, leaving all system libraries intact.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    torch.cuda.init()
    drivers = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
               if "/libcuda.so" in line and line.split()[-1].startswith("/")}
    if len(drivers) != 1:
        raise RuntimeError(f"Expected exactly one loaded CUDA driver, found {drivers}")
    driver = Path(drivers.pop())
    if not driver.is_file():
        raise RuntimeError("Loaded CUDA driver is not accessible inside the container")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("libcuda.so", "libcuda.so.1"):
        (args.out / name).symlink_to(driver)
    (args.out / "driver.json").write_text(json.dumps({"runtime_driver": str(driver),
        "torch": torch.__version__, "cuda": torch.version.cuda}, indent=2)+"\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
