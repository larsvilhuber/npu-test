# Sourced by run-*.sh. Defines run_test NAME CMD... : runs CMD under monitor.sh,
# writing results/$SITE/NAME.monitor.csv and appending bench rows to results/$SITE/bench.csv
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$REPO/results/$SITE"
mkdir -p "$OUT"

run_test() {
  local name=$1; shift
  echo "== $name"
  "$REPO/scripts/monitor.sh" "$OUT/$name.monitor.csv" &
  local mon=$!
  sleep 5   # idle baseline
  "$@" --test "$name" --out "$OUT/bench.csv"
  sleep 5
  kill "$mon"; wait "$mon" 2>/dev/null
  sleep 30  # cool-down between tests
}

# Orange Pi only: set the NPU devfreq governor via the sudoers rule.
npu_governor() {
  local f; f=$(ls /sys/class/devfreq/*npu*/governor)
  echo "$1" | sudo -n "$(command -v tee)" "$f" >/dev/null
}
npu_governor_get() { cat /sys/class/devfreq/*npu*/governor; }
