#!/usr/bin/env bash
# Checks that setup-oiseau.sh and docs/PREREQUISITES.md were done, then
# writes system information to results/oiseau/sysinfo.csv (key,value).
REPO=$(cd "$(dirname "$0")/.." && pwd)
SITE=$(basename "$0" .sh); SITE=${SITE#check-}
OUT="$REPO/results/$SITE"
mkdir -p "$OUT"
source "$REPO/bench-env/bin/activate" 2>/dev/null

check() { if eval "$2" >/dev/null 2>&1; then echo "PASS $1"; else echo "FAIL $1"; fi; }

check "bench-env"         "test -n \"\$VIRTUAL_ENV\""
check "torch import"      "python3 -c 'import torch, torchvision'"
check "openvino import"   "python3 -c 'import openvino'"
check "litert import"     "python3 -c 'import ai_edge_litert'"

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
echo "openvino,$(python3 -c 'import openvino; print(openvino.__version__)' 2>/dev/null)"
echo "cpu_governor,$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "cpu_max_mhz,$(lscpu | awk -F: '/CPU max MHz/{gsub(/ /,"",$2); print $2}')"
echo "git_commit,$(git -C "$REPO" rev-parse --short HEAD)"
} > "$OUT/sysinfo.csv"
echo "wrote $OUT/sysinfo.csv"
