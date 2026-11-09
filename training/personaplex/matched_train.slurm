#!/usr/bin/env bash
# Source is deployed through Git; sbatch --chdir selects the immutable worktree.
# Native scheduler/aggregate guards recheck every actual allocation and child.
set -euo pipefail
exec /usr/bin/python3 -B -u -m training.personaplex.matched_job \
  --source "${1:?exact source SHA}" --root "${2:?new root}" \
  --inputs "${3:?complete323 input directory}" --inputs-sha "${4:?original full report SHA}"
