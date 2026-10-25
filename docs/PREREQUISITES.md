# Prerequisites (root and package installs)

Everything that needs root, or installs a library, is listed here and done by
hand. The `setup-*.sh`, `check-*.sh` and `run-*.sh` scripts never call `sudo`,
except through the two sudoers rules below.

## Both hosts

```bash
sudo apt install python3-venv python3-pip git
```

## orangepi (RK3588)

Group membership for `/dev/dri/renderD128` (re-login afterwards):

```bash
sudo usermod -aG render,video $USER
```

RKNPU runtime library, from the cloned `rknn-toolkit2` (see `setup-orangepi.sh`
for the clone):

```bash
sudo cp rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp312-*/rknnlite/api/libs/librknnrt.so /usr/lib/
sudo ldconfig
```

Two sudoers rules, the only sudo use at run time: read the NPU load counter,
and write the NPU devfreq governor (locks the NPU clock to `performance`; the
original governor is restored when the run ends). Each names one exact file.

```bash
scripts/sudoers/npu.sudoers.sh | sudo tee /etc/sudoers.d/npu
sudo chmod 0440 /etc/sudoers.d/npu
sudo visudo -cf /etc/sudoers.d/npu
```

## BirdNET model (for the audio benchmark)

Download the BirdNET V2.4 FP32 TFLite model (the one BirdNET-Go bundles) from
the [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) repository
and place it at `models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite` on each host.

Python packages are installed into the venvs by `setup-*.sh` (no root).

## Offline analysis (any machine)

```bash
pip install -r analysis/requirements.txt
```
