#!/usr/bin/env bash
# Full benchmark suite for the Intel N150. Needs: check-oiseau.sh all PASS.
SITE=oiseau
source "$(dirname "$0")/lib-run.sh"
source "$REPO/bench-env/bin/activate"
B="$REPO/scripts/bench"
rm -f "$OUT/bench.csv"

for t in 1 2 4; do
  run_test "torch_threads_$t" python3 "$B/resnet18_torch.py" --threads "$t" --iters 500
done
for b in 4 16; do
  run_test "torch_batch_$b" python3 "$B/resnet18_torch.py" --threads 4 --batch "$b" --iters 100
done
run_test torch_sustained python3 "$B/resnet18_torch.py" --threads 4 --duration 300

for d in CPU GPU; do
  run_test "openvino_$d" python3 "$B/resnet18_openvino.py" --device "$d" --iters 2000
done
run_test openvino_sustained python3 "$B/resnet18_openvino.py" --device CPU --duration 300

for t in 1 2 4; do
  run_test "birdnet_threads_$t" python3 "$B/birdnet_tflite.py" --threads "$t" --iters 200
done
