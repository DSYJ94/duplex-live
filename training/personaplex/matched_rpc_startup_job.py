"""No submissions here: exact approved startup recovery8 native worker."""
import argparse
from .matched_rpc_job import main

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','inputs-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args(),startup_recovery=True)
