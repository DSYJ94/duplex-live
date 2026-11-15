"""Expose the actually loaded driver before importing a native GPU stage."""
import os,runpy,subprocess,sys,uuid
from pathlib import Path


def main():
    module=sys.argv[1]
    lease=Path(os.environ['DUPLEX_LIVE_LEASE_FILE'])
    folder=lease.parent/('driver-'+uuid.uuid4().hex)
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output(
        [sys.executable,'-B','-m','evals.cuda_linker_path','--out',str(folder)],
        universal_newlines=True).strip()
    sys.argv=[module]+sys.argv[2:]
    runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()
