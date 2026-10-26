#!/usr/bin/env bash
# Full benchmark suite for the Orange Pi 5. Needs: check-orangepi.sh all PASS.
SITE=orangepi
source "$(dirname "$0")/lib-run.sh"
source "$HOME/rknn-env/bin/activate"
B="$REPO/scripts/bench"
[ -n "${ONLY:-}" ] || rm -f "$OUT/bench.csv"   # ONLY reruns append

# NPU core masks (all work with rknpu driver 0.9.7; on 0.9.3 only auto and 0 did)
for m in auto 0 1 2 0_1 0_1_2; do
  run_test "rknn_mask_$m" python3 "$B/resnet18_rknn.py" --core-mask "$m" --iters 2000
done
run_test rknn_sustained python3 "$B/resnet18_rknn.py" --core-mask auto --duration 300
for n in 2 3; do   # concurrent processes; NPU_CORE_AUTO spreads them over idle cores
  run_streams "rknn_streams_$n" "$n" python3 "$B/resnet18_rknn.py" --core-mask auto --duration 60
done

# NPU clock locked to the performance governor, restored on exit
GOV0=$(npu_governor_get)
trap 'npu_governor "$GOV0"' EXIT
npu_governor performance
for m in auto 0; do
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

# YOLOv5s at 640x640: model call only (infer) and with letterbox + decode + NMS (e2e)
for m in i8 fp16; do
  run_test "yolo_rknn_${m}_infer" python3 "$B/yolov5_rknn.py" --model "yolov5s_relu_$m.rknn" --stage infer --iters 1000
  run_test "yolo_rknn_${m}_e2e" python3 "$B/yolov5_rknn.py" --model "yolov5s_relu_$m.rknn" --stage e2e --iters 500
done
for m in 1 2 0_1 0_1_2; do
  run_test "yolo_rknn_i8_infer_mask_$m" python3 "$B/yolov5_rknn.py" --model yolov5s_relu_i8.rknn --stage infer --core-mask "$m" --iters 1000
done
for n in 2 3 4; do   # concurrent processes; NPU_CORE_AUTO spreads them over idle cores
  run_streams "yolo_rknn_i8_e2e_streams_$n" "$n" python3 "$B/yolov5_rknn.py" --model yolov5s_relu_i8.rknn --stage e2e --duration 60
done
run_test yolo_ovCPU_infer python3 "$B/yolov5_openvino.py" --device CPU --stage infer --iters 100
run_test yolo_ovCPU_e2e python3 "$B/yolov5_openvino.py" --device CPU --stage e2e --iters 100
