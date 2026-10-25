---
title: "Benchmarking Edge AI: Orange Pi 5 (RK3588 NPU) vs. Intel N150 CPU"
date: 2026-10-25
categories: [Edge Computing, Machine Learning]
tags: [rk3588, npu, rknn, intel-n150, pytorch, openvino, birdnet, benchmark]
---

# Edge AI Showdown: Pi-like device with NPU vs. Intel N150 CPU: 12x faster than stock PyTorch, 7x faster than OpenVINO

I never know what the purpose is of **NPU** (neural-processing unit). Turns out, I have an Orange Pi 5 sitting around, which actually does have one. Compared to a small computer running a low-power Intel CPU, that NPU can accelerate certain tasks, at lower power. But note that I did not have an "AI CPU" at hand, to compare what that would look like on an Intel chip that also has an NPU.

Now I need to figure out where I need this... (updating my [Birdnet Go](https://github.com/tphakala/birdnet-go) [node](https://app.birdweather.com/) on [Birdnet](https://birdnet.cornell.edu/)?)

> In a head-to-head [ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) inference benchmark, the **Orange Pi 5 (RK3588 NPU)** classified an image in **4.5 ms (223 images/s)**: **12x faster** than an **Intel N150 CPU** running stock PyTorch and **7x faster** than the same CPU running Intel's OpenVINO, while using about **a quarter of one CPU core** and staying 30 °C cooler.

This is an expanded version of my first run. The benchmark scripts are no longer pasted into the post; everything is in the [GitHub repository](https://github.com/larsvilhuber/npu-test), including the raw CSV output of every run.

## Benchmark Results Summary

ResNet-18, one 224x224 image per call, thousands of calls per test. I report the **median** latency: the Intel box also runs other services (see caveats), which adds a long upper tail to its means.

| System | Compute device | Runtime engine | Median latency (ms) | 95th pct (ms) | Throughput (img/s) | CPU busy (% of all cores) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **Orange Pi 5** | RK3588 NPU | RKNN Lite 2 (v2.5.0), INT8 | **4.49** | 4.70 | **223** | 4.0 |
| **Orange Pi 5** | RK3588 NPU, core 0 pinned | RKNN Lite 2 (v2.5.0), INT8 | 4.05 | 4.17 | 247 | 3.8 |
| **Orange Pi 5** | 8 CPU cores (A76+A55) | PyTorch FP32 | 112.3 | 128.4 | 8.9 | 59 |
| **Intel N150** | 4 CPU cores | PyTorch FP32 | 53.5 | 477.6 | 18.7 | 95 |
| **Intel N150** | 4 CPU cores | OpenVINO FP32 | 31.0 | 60.1 | 32.3 | 93 |

Note that the NPU runs the model in INT8 (the pre-compiled model from Rockchip's examples), while both CPU runs use FP32. Same network, different numeric precision; I did not check accuracy.

![Latency distribution](https://raw.githubusercontent.com/larsvilhuber/npu-test/main/figures/latency_distribution.png)

### CPU and NPU usage

While the NPU runs inference, the Orange Pi's eight CPU cores are about 96% idle (2.5-4% busy overall, about a quarter of one core), and the NPU's busiest core sits at roughly 72-85% (sustained run: 85%). Doing the same work on the CPUs keeps the Orange Pi at 59% busy and the N150 at 95%.

![CPU vs NPU utilisation](https://raw.githubusercontent.com/larsvilhuber/npu-test/main/figures/utilisation.png)

### Five minutes of sustained load

Each engine also ran continuously for 300 seconds. Nothing throttles: the NPU stays at 4.5 ms and warms from 32 to 42 °C, while the Orange Pi CPU run heats to 76 °C and the N150 to 75-79 °C, with no visible drop in clock speed or throughput. (The N150's baseline is already ~67 °C because of the other services on that machine.)

![Sustained runs](https://raw.githubusercontent.com/larsvilhuber/npu-test/main/figures/sustained.png)

The OpenVINO panel shows latency alternating between ~30 and ~52 ms in a regular pattern of roughly ten seconds. I did not chase this down; the likely suspect is the other workload on the machine competing for the cores.

### Scaling on the CPUs

PyTorch on the N150 scales with threads (9.3 → 13.9 → 18.7 img/s at 1, 2, 4 threads); on the Orange Pi it barely does (6.8 → 8.9 img/s from 1 to 8 threads), because the big and little cores are very different. Batching helps both (N150: 18.7 → 28 img/s at batch 16).

![Thread and batch scaling](https://raw.githubusercontent.com/larsvilhuber/npu-test/main/figures/scaling.png)

## A workload closer to Birdnet-Go: BirdNET audio classification

[Birdnet-Go](https://github.com/tphakala/birdnet-go) runs the BirdNET V2.4 TFLite model on 3-second, 48 kHz audio windows. I ran the very same FP32 model file that Birdnet-Go bundles (via TFLite/XNNPACK on the CPU) on both machines, with random noise as input (compute cost does not depend on the audio).

| System | Threads | Median per 3 s window (ms) | Real-time factor |
| :--- | ---: | ---: | ---: |
| Intel N150 | 1 | 50.8 | 59x |
| Intel N150 | 2 | 39.7 | 76x |
| Intel N150 | 4 | 33.9 | 88x |
| Orange Pi 5 (CPU) | 1 | 93.2 | 32x |
| Orange Pi 5 (CPU) | 4 | 62.4 | 48x |
| Orange Pi 5 (CPU) | 8 | 54.5 | 55x |

![BirdNET real-time factor](https://raw.githubusercontent.com/larsvilhuber/npu-test/main/figures/birdnet.png)

**Both machines are far faster than real time on the CPU alone**: even a single thread processes 30-60 seconds of audio per second. A Birdnet-Go node listening to one or a few streams does not need an NPU for speed. I did **not** run BirdNET on the NPU: that requires converting the model with the x86-only RKNN toolkit, and I have not tried (the model's audio front end may contain operators the NPU does not support). The case for the NPU on a bird node is therefore freeing CPU and energy, not speed.

## What did not work

- **Only one NPU core is usable with this model.** My first version of this post said the NPU ran on "3 cores". It does not: with Rockchip's stock pre-compiled ResNet-18, the automatic mask (`NPU_CORE_AUTO`) and core 0 work (4.5 ms and 4.05 ms), but cores 1 and 2 and the multi-core masks (`0_1`, `0_1_2`) fail with `failed to submit` (some hang the driver until it soft-resets). Using all three cores probably requires re-exporting the model for multi-core, which needs the toolkit's x86 version.
- **Locking the NPU clock did not help.** I had written that performance mode could reduce latency to ~1.5 ms. I tested it (devfreq governor `performance` versus the default `rknpu_ondemand`) and saw no meaningful change (4.28 vs 4.35 ms mean, 225 vs 225 img/s sustained). I did not record the NPU clock during the runs, but it reports 1 GHz, the maximum the driver offers, so presumably the default governor already reaches it under load. The earlier ~1.5 ms estimate is not reproduced.
- **Intel iGPU:** OpenVINO only saw the CPU on the N150 (the compute runtime for the integrated GPU is not installed), so there is no iGPU number.

## Caveats

- The Intel N150 box (`oiseau`) is shared: it also runs Home Assistant, UniFi and other services, which use ~27% CPU and keep it at ~67 °C before my tests start. That inflates its tail latencies (note the 95th percentile of 478 ms for four-thread PyTorch against a 53 ms median) and could hide some of its true performance; the Orange Pi was idle (load 0.1).
- The CPU governors differ: `performance` on the Orange Pi, `powersave` (intel_pstate, which still boosts to ~2.8 GHz under load) on the N150.
- I did not measure power, though lower power is the usual selling point of an NPU.
- INT8 (NPU) vs FP32 (CPU): not equal precision; accuracy not compared.
- One run per configuration on one unit of each machine.

## Reproducing this

All scripts are in the [repository](https://github.com/larsvilhuber/npu-test); the post no longer inlines them.

| What | Script |
| :--- | :--- |
| One-time root steps (groups, `librknnrt.so`, sudoers rules) | [`docs/PREREQUISITES.md`](https://github.com/larsvilhuber/npu-test/blob/main/docs/PREREQUISITES.md) |
| Setup (venv, packages, model, assets) | [`scripts/setup-orangepi.sh`](https://github.com/larsvilhuber/npu-test/blob/main/scripts/setup-orangepi.sh), [`scripts/setup-oiseau.sh`](https://github.com/larsvilhuber/npu-test/blob/main/scripts/setup-oiseau.sh) |
| Check that setup was done; write system info | [`scripts/check-orangepi.sh`](https://github.com/larsvilhuber/npu-test/blob/main/scripts/check-orangepi.sh), [`scripts/check-oiseau.sh`](https://github.com/larsvilhuber/npu-test/blob/main/scripts/check-oiseau.sh) |
| Run the whole suite (with CPU/NPU/temperature monitoring) | [`scripts/run-orangepi.sh`](https://github.com/larsvilhuber/npu-test/blob/main/scripts/run-orangepi.sh), [`scripts/run-oiseau.sh`](https://github.com/larsvilhuber/npu-test/blob/main/scripts/run-oiseau.sh) |
| Individual benchmarks | [`scripts/bench/`](https://github.com/larsvilhuber/npu-test/tree/main/scripts/bench) |
| Raw results (CSV) | [`results/`](https://github.com/larsvilhuber/npu-test/tree/main/results) |
| Offline collation and graphs | [`analysis/collate.py`](https://github.com/larsvilhuber/npu-test/blob/main/analysis/collate.py) |

NPU load is read from `/sys/kernel/debug/rknpu/load`, and the NPU clock governor is set through `devfreq`; both need root, which I granted through two narrow sudoers rules (one exact file each) generated by [`scripts/sudoers/npu.sudoers.sh`](https://github.com/larsvilhuber/npu-test/blob/main/scripts/sudoers/npu.sudoers.sh).

{% details System information (both machines) %}

**oiseau**

| Item | Value |
| :--- | :--- |
| hostname | oiseau |
| cpu_model | Intel(R) N150 |
| cores | 4 |
| ram_gib | 14.9 |
| kernel | Linux 7.0.0-38-generic |
| os | Ubuntu 26.04.1 LTS |
| python | 3.14.4 |
| torch | 2.14.1+cu130 |
| openvino | 2026.4.1-22982-e213a147257-releases/2026/4 |
| cpu_governor | powersave |
| cpu_max_mhz | 3600.0000 |
| git_commit | f2aaf33 |

**orangepi**

| Item | Value |
| :--- | :--- |
| hostname | ubuntu |
| cpu_model | Cortex-A55+Cortex-A76 |
| cores | 8 |
| ram_gib | 15.6 |
| kernel | Linux 6.1.43-rockchip-rk3588 |
| os | Ubuntu 24.04.5 LTS |
| python | 3.12.3 |
| torch | 2.14.1+cu130 |
| rknn_toolkit_lite2 | 2.5.0 |
| librknnrt | 2.5.0 |
| rknpu_driver | 0.9.3 |
| cpu_governor | performance |
| npu_governor | rknpu_ondemand |
| npu_freq_hz | 1000000000 |
| git_commit | ac942a6 |

{% enddetails %}

---

## Key Takeaways

1. **Hardware acceleration matters:** the RK3588 NPU is an order of magnitude faster than either CPU for this vision task: 12x over stock PyTorch and 7x over OpenVINO on the N150, 25x over the Orange Pi's own CPUs.
2. **Resource allocation:** inference on the NPU leaves the CPUs essentially free (about a quarter of one core) and cool (42 °C vs 76 °C), which is where an always-on edge node benefits. The Orange Pi's own CPU is not THAT much slower than the N150 on audio ([see the CPU benchmark comparison](https://www.cpubenchmark.net/compare/6304vs4906/Intel-N150-vs-Rockchip-RK3588)).
3. **Software stack matters on the CPU too:** simply switching the N150 from stock PyTorch to OpenVINO gave 1.7x.
4. **For Birdnet-Go, speed is not the problem:** both CPUs run BirdNET 30-90x faster than real time. The NPU is attractive only if the model can be converted for it, to save CPU and power; I did not test that.
