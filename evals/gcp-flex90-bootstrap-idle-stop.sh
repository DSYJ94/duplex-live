#!/bin/sh
# Infrastructure fail-safe only; no model/application source or credential.
set -eu
metadata() {
  curl --silent --show-error --fail --connect-timeout 1 --max-time 3 \
    -H 'Metadata-Flavor: Google' "http://169.254.169.254/computeMetadata/v1/$1"
}
test "$(id -u)" = 0
test "$(cat /proc/1/comm)" = systemd
test "$(metadata project/project-id)" = aurium-duplex-live-20261024
test "$(metadata instance/name)" = duplex-native-saved8-gpu-flex90-20261025-v1
case "$(metadata instance/machine-type)" in
  */machineTypes/a2-ultragpu-1g) ;;
  *) exit 1 ;;
esac
# A lost Mac must not leave an allocated GPU idle for its entire native lease.
# The controller cancels this timer only after both live budget guards are armed.
if ! /usr/bin/systemd-run --unit=duplex-native-flex90-bootstrap-stop-v1 \
    --on-active=180s --timer-property=AccuracySec=1s /sbin/shutdown -h now; then
  /sbin/shutdown -h now
  exit 1
fi
