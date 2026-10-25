#!/usr/bin/env bash
# Prints (does not install) the two sudoers rules the benchmarks use, on the
# Orange Pi only:
#   1. read the RKNPU load counter
#   2. write the NPU devfreq governor (to lock the NPU clock: performance)
# Both name one exact file; sudo matches the arguments literally.
#
#   scripts/sudoers/npu.sudoers.sh | sudo tee /etc/sudoers.d/npu
#   sudo chmod 0440 /etc/sudoers.d/npu
#   sudo visudo -cf /etc/sudoers.d/npu
set -euo pipefail

CAT=$(command -v cat)
TEE=$(command -v tee)
NPU_LOAD=/sys/kernel/debug/rknpu/load
NPU_GOV=$(ls /sys/class/devfreq/*npu*/governor)

printf '%s ALL=(root) NOPASSWD: %s %s\n' "$USER" "$CAT" "$NPU_LOAD"
printf '%s ALL=(root) NOPASSWD: %s %s\n' "$USER" "$TEE" "$NPU_GOV"
