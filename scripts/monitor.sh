#!/usr/bin/env bash
# usage: monitor.sh OUT.csv      Samples at 1 Hz until killed.
# Columns: t_s,cpu_pct,cpu_mhz,temp_c,npu0_pct,npu1_pct,npu2_pct
# NPU columns stay empty without an NPU or without the sudoers rule
# (scripts/sudoers/npu.sudoers.sh).
OUT=$1
CAT=$(command -v cat)
NPU_LOAD=/sys/kernel/debug/rknpu/load

cpu_times() { awk '/^cpu /{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }

echo "t_s,cpu_pct,cpu_mhz,temp_c,npu0_pct,npu1_pct,npu2_pct" > "$OUT"
t0=$(date +%s.%N)
read -r pb pt < <(cpu_times)
while sleep 1; do
  read -r b t < <(cpu_times)
  cpu=$(awk -v b="$b" -v pb="$pb" -v t="$t" -v pt="$pt" 'BEGIN{d=t-pt; printf "%.1f", (d>0) ? 100*(b-pb)/d : 0}')
  pb=$b; pt=$t
  mhz=$($CAT /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{s+=$1;n++} END{if(n) printf "%.0f", s/n/1000}')
  temp=$($CAT /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 | awk '{printf "%.1f", $1/1000}')
  npu=$(sudo -n "$CAT" "$NPU_LOAD" 2>/dev/null | grep -o '[0-9]*%' | tr -d '%' | paste -sd,)
  echo "$(awk -v n="$(date +%s.%N)" -v s="$t0" 'BEGIN{printf "%.1f", n-s}'),$cpu,$mhz,$temp,${npu:-,,}" >> "$OUT"
done
