#!/bin/bash
#SBATCH --account=brics.u6ada
#SBATCH --partition=workq
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=10:30:00
#SBATCH --job-name=duplex-ret-audit-recovery-v1
#SBATCH --output=/projects/u6ada/duplex-live/native-ret-recovery-%j.log
set -euo pipefail
umask 077
: "${DUPLEX_PAIR_SOURCE:?Explicit immutable recovery source required}"
cd "/projects/u6ada/duplex-live/source-pair-${DUPLEX_PAIR_SOURCE:0:7}"
test "$(git rev-parse HEAD)" = "$DUPLEX_PAIR_SOURCE"
python3 -B -u -m training.personaplex.action_pair_recovery_host \
 --source "$DUPLEX_PAIR_SOURCE" \
 --root /projects/u6ada/duplex-live/native-ret-paired-recovery-20261117-v1 \
 --backup /home/u6ada/dsyj94.u6ada/duplex-native-ret-paired-backups-v1 \
 --inputs /projects/u6ada/duplex-live/native-expanded1072-evaluation-v1
