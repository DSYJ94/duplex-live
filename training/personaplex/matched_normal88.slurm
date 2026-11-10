#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/python3 -B -u -m training.personaplex.matched_postupdate --normal88 \
 --source "${1:?exact source}" --inputs "${2:?qualified inputs}" --inputs-sha "${3:?input report SHA}"
