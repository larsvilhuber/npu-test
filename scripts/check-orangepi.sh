#!/usr/bin/env bash
# Checks that setup-orangepi.sh and docs/PREREQUISITES.md were done, then
# writes system information to results/orangepi/sysinfo.csv (key,value).
REPO=$(cd "$(dirname "$0")/.." && pwd)
SITE=$(basename "$0" .sh); SITE=${SITE#check-}
OUT="$REPO/results/$SITE"
mkdir -p "$OUT"
source "$HOME/rknn-env/bin/activate" 2>/dev/null

check() { if eval "$2" >/dev/null 2>&1; then echo "PASS $1"; else echo "FAIL $1"; fi; }

check "render,video groups"   "id -nG | grep -qw render && id -nG | grep -qw video"
check "librknnrt.so"          "ldconfig -p | grep -q librknnrt"
check "rknnlite import"       "python3 -c 'from rknnlite.api import RKNNLite'"
check "torch import"          "python3 -c 'import torch, torchvision'"
check "litert import"         "python3 -c 'import ai_edge_litert'"
check "model + image"         "test -f $REPO/assets/resnet18_for_rk3588.rknn && test -f $REPO/assets/space_shuttle_224.jpg"
check "sudoers NPU load"      "sudo -n $(command -v cat) /sys/kernel/debug/rknpu/load"
check "sudoers NPU governor"  "sudo -n -l $(command -v tee) $(ls /sys/class/devfreq/*npu*/governor)"

{
echo "key,value"
echo "hostname,$(hostname)"
echo "cpu_model,\"$(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2}' | sort -u | paste -sd'+')\""
echo "cores,$(nproc)"
echo "ram_gib,$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)"
echo "kernel,$(uname -sr)"
echo "os,\"$(. /etc/os-release; echo "$PRETTY_NAME")\""
echo "python,$(python3 -c 'import platform; print(platform.python_version())')"
echo "torch,$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "rknn_toolkit_lite2,$(pip show rknn-toolkit-lite2 2>/dev/null | awk '/^Version/{print $2}')"
echo "librknnrt,$(strings "$(ldconfig -p | awk '/librknnrt/{print $NF; exit}')" 2>/dev/null | grep -m1 -o 'librknnrt version: [^ ]*' | awk '{print $3}')"
echo "rknpu_driver,$(cat /sys/module/rknpu/version 2>/dev/null)"
echo "cpu_governor,$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "npu_governor,$(cat /sys/class/devfreq/*npu*/governor 2>/dev/null)"
echo "npu_freq_hz,$(cat /sys/class/devfreq/*npu*/cur_freq 2>/dev/null)"
echo "git_commit,$(git -C "$REPO" rev-parse --short HEAD)"
} > "$OUT/sysinfo.csv"
echo "wrote $OUT/sysinfo.csv"
