#!/usr/bin/env bash
# ONE training-only allocation. This entry submits no jobs or native admissions.
set -euo pipefail
exec /usr/bin/python3 -B -u -m training.personaplex.matched_training_only \
  --source "${1:?exact new execution producer}" \
  --inputs "${2:?unchanged complete323 input directory}" \
  --inputs-sha "${3:?original complete323 report SHA}"
