"""Point Triton at the CUDA driver already loaded by Torch inside Apptainer.

No driver, system linker cache, symlink or library file is changed. The override
is local to this inference process and must resolve to the exact mapped bytes.
"""
import os
from pathlib import Path
from evals.moshi_model_download import sha256


def mapped_driver_directory(maps):
    paths=set()
    for line in maps.splitlines():
        if '/libcuda.so' in line:
            fields=line.split(maxsplit=5)
            if len(fields)!=6:raise RuntimeError('Malformed CUDA mapping')
            paths.add(Path(fields[-1]))
    if len(paths)!=1:raise RuntimeError('Expected exactly one mapped CUDA driver library')
    actual=next(iter(paths))
    if not actual.is_file():raise RuntimeError('Mapped CUDA driver is not visible inside the container')
    folder=actual.parent;digest=sha256(actual)
    for name in ('libcuda.so','libcuda.so.1'):
        path=folder/name
        if not path.is_file() or sha256(path)!=digest:raise RuntimeError('CUDA linker names do not match the already loaded driver')
    return folder,digest


def configure_loaded_driver():
    import torch
    if not torch.cuda.is_initialized():raise RuntimeError('Torch must initialize the actual CUDA driver first')
    folder,digest=mapped_driver_directory(Path('/proc/self/maps').read_text())
    previous=os.environ.get('TRITON_LIBCUDA_PATH');os.environ['TRITON_LIBCUDA_PATH']=str(folder)
    from triton.backends.nvidia.driver import libcuda_dirs
    actual=libcuda_dirs()
    if actual!=[str(folder)]:raise RuntimeError('Triton did not select the verified loaded driver directory')
    return dict(variable='TRITON_LIBCUDA_PATH',value=str(folder),previous=previous,
        loaded_driver_sha256=digest,linker_names_match_mapped_bytes=True,
        driver_changed=False,system_files_changed=False,symlinks_created=False)
