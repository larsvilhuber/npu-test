#!/usr/bin/env bash
# Full benchmark suite for the Intel N150. Needs: check-oiseau.sh all PASS.
SITE=oiseau
source "$(dirname "$0")/lib-run.sh"
source "$REPO/bench-env/bin/activate"
B="$REPO/scripts/bench"
# ONLY=<regex> reruns matching tests; their old rows must be removed first
[[ -n "${ONLY:-}" ]] || rm -f "$OUT/bench.csv"

for t in 1 2 4; do
  run_test "torch_threads_$t" python3 "$B/resnet18_torch.py" --threads "$t" --iters 500
done
for b in 4 16; do
  run_test "torch_batch_$b" python3 "$B/resnet18_torch.py" --threads 4 --batch "$b" --iters 100
done
run_test torch_sustained python3 "$B/resnet18_torch.py" --threads 4 --duration 300

OV_DEVICES=$(python3 -c "import openvino as ov; print(*ov.Core().available_devices)")
for d in $OV_DEVICES; do
  run_test "openvino_$d" python3 "$B/resnet18_openvino.py" --device "$d" --iters 2000
done
run_test openvino_sustained python3 "$B/resnet18_openvino.py" --device CPU --duration 300
for d in $OV_DEVICES; do
  [ "$d" = CPU ] || run_test "openvino_${d}_sustained" python3 "$B/resnet18_openvino.py" --device "$d" --duration 300
done

for t in 1 2 4; do
  run_test "birdnet_threads_$t" python3 "$B/birdnet_tflite.py" --threads "$t" --iters 200
done

# YOLOv5s at 640x640: model call only (infer) and with letterbox + decode + NMS (e2e)
for d in $OV_DEVICES; do
  run_test "yolo_ov${d}_infer" python3 "$B/yolov5_openvino.py" --device "$d" --stage infer --iters 500
  run_test "yolo_ov${d}_e2e" python3 "$B/yolov5_openvino.py" --device "$d" --stage e2e --iters 300
  for n in 2 3 4; do   # concurrent processes
    run_streams "yolo_ov${d}_e2e_streams_$n" "$n" python3 "$B/yolov5_openvino.py" --device "$d" --stage e2e --duration 60
  done
done
