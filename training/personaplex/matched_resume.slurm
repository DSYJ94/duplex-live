#!/usr/bin/env bash
#SBATCH --job-name=duplex-matched-resume-v1
# Resource flags are supplied by the bounded submission controller.
set -euo pipefail
exec /usr/bin/python3 -B -u -m training.personaplex.matched_resume \
  --controller-source "${1:?exact host/controller Git SHA}" \
  --inputs "${2:?unchanged complete323 directory}" \
  --inputs-sha "${3:?original full data report SHA}" \
  --previous-root "${4:?actual prior successful window root}" \
  --previous-result-sha "${5:?exact prior phase-result SHA}"
