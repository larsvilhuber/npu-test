#!/usr/bin/env bash
# Full benchmark suite for the Orange Pi 5. Needs: check-orangepi.sh all PASS.
SITE=orangepi
source "$(dirname "$0")/lib-run.sh"
source "$HOME/rknn-env/bin/activate"
B="$REPO/scripts/bench"
rm -f "$OUT/bench.csv"

for m in auto 0 0_1_2; do
  run_test "rknn_mask_$m" python3 "$B/resnet18_rknn.py" --core-mask "$m" --iters 2000
done
run_test rknn_sustained python3 "$B/resnet18_rknn.py" --core-mask auto --duration 300

# NPU clock locked to the performance governor, restored on exit
GOV0=$(npu_governor_get)
trap 'npu_governor "$GOV0"' EXIT
npu_governor performance
for m in auto 0 0_1_2; do
  run_test "rknn_perf_mask_$m" python3 "$B/resnet18_rknn.py" --core-mask "$m" --iters 2000
done
run_test rknn_perf_sustained python3 "$B/resnet18_rknn.py" --core-mask auto --duration 300
npu_governor "$GOV0"

for t in 1 2 4 8; do
  run_test "torch_threads_$t" python3 "$B/resnet18_torch.py" --threads "$t" --iters 200
done
for b in 4 16; do
  run_test "torch_batch_$b" python3 "$B/resnet18_torch.py" --threads 8 --batch "$b" --iters 50
done
run_test torch_sustained python3 "$B/resnet18_torch.py" --threads 8 --duration 300

for t in 1 4 8; do
  run_test "birdnet_threads_$t" python3 "$B/birdnet_tflite.py" --threads "$t" --iters 200
done
