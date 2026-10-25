# Sourced by run-*.sh. Defines run_test NAME CMD... : runs CMD under monitor.sh,
# writing results/$SITE/NAME.monitor.csv and appending bench rows to results/$SITE/bench.csv
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$REPO/results/$SITE"
mkdir -p "$OUT"

TEST_TIMEOUT=${TEST_TIMEOUT:-1500}  # seconds; longest test is 300 s plus model load

# run_streams NAME N CMD... : N concurrent copies of CMD under one monitor
run_streams() {
  local name=$1 n=$2; shift 2
  [[ -z "${ONLY:-}" || $name =~ $ONLY ]] || return 0
  echo "== $name ($n stream(s))"
  "$REPO/scripts/monitor.sh" "$OUT/$name.monitor.csv" &
  local mon=$!
  sleep 5   # idle baseline
  local pids=() i
  for ((i = 0; i < n; i++)); do
    # a hung NPU submit sits in uninterruptible sleep until killed
    timeout -s KILL "$TEST_TIMEOUT" "$@" --test "$name" --out "$OUT/bench.csv" &
    pids+=($!)
  done
  wait "${pids[@]}"
  sleep 5
  kill "$mon"; wait "$mon" 2>/dev/null
  sleep 30  # cool-down between tests
}

run_test() { local name=$1; shift; run_streams "$name" 1 "$@"; }

# Orange Pi only: set the NPU devfreq governor via the sudoers rule.
npu_governor() {
  local f; f=$(ls /sys/class/devfreq/*npu*/governor)
  echo "$1" | sudo -n "$(command -v tee)" "$f" >/dev/null
}
npu_governor_get() { cat /sys/class/devfreq/*npu*/governor; }
