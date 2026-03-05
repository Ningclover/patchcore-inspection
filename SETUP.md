# Environment Setup — patchcore-inspection on WC Server

Setting up [PatchCore](https://arxiv.org/abs/2106.08265) (Roth et al. 2021) for industrial
anomaly detection on wcgpu1, using the WCWC software ecosystem.

---

## Prerequisites

- Account on a WC server (e.g. `wcgpu1`)
- WCWC installed at `/wcwc/`
- `direnv` available (`which direnv`)
- Working directory at `/nfs/data/1/xning/elec_AI/`
- GPU: NVIDIA RTX 4090 (compute capability sm_89), CUDA 12.5

---

## Hardware & Software Versions

| Component | Version |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| CUDA Driver | 555.42.06 |
| CUDA Runtime | 12.5 |
| Python | 3.11.9 (from WCWC spack) |
| PyTorch | 2.4.0 (CUDA 12.5, sm_89) |
| torchvision | 0.19.1 (built from source) |
| NumPy | 1.26.4 (from spack, 1.x required) |

---

## Step 1 — Create a Spack Environment from WCWC packages

WCWC provides a pre-built PyTorch with CUDA support. Find the right build hash:

```bash
wcwc spack find --format "{name}@{version}%{compiler} {variants} {hash:7}" py-torch
```

Expected output (confirming CUDA and sm_89 for RTX 4090):
```
py-torch@2.4.0%gcc@=12.2.0 ~caffe2+cuda+cudnn ... cuda_arch=89 2dj527f
```

Also find tqdm:
```bash
wcwc spack find --format "{name}@{version} {hash:7}" py-tqdm
# py-tqdm@4.66.3 o3647dr
```

Create the environment:
```bash
wcwc env -e /nfs/data/1/xning/elec_AI/env \
    py-torch/2dj527f py-tqdm/o3647dr
```

Verify the spack env activates and CUDA is available:
```bash
wcwc shell -e /nfs/data/1/xning/elec_AI/env -c \
    "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
# Expected: 2.4.0 True
```

---

## Step 2 — Install pip packages to a local `pypackages/` directory

Several packages are not in WCWC and must be installed via pip. Use `--target` to keep
them in the project directory (avoids polluting `~/.local` or the system Python).

> **Important:** Use `TMPDIR=/nfs/data/1/xning/tmp` — `/tmp` is only 1.7 GB and pip
> will run out of space when downloading large packages.

```bash
mkdir -p /nfs/data/1/xning/tmp
mkdir -p /nfs/data/1/xning/elec_AI/pypackages
```

Install scikit-image and scikit-learn (with their deps):
```bash
wcwc shell -e /nfs/data/1/xning/elec_AI/env -c "
    TMPDIR=/nfs/data/1/xning/tmp \
    pip install --target /nfs/data/1/xning/elec_AI/pypackages \
        scikit-image scikit-learn
"
```

> **Warning:** This will also pull in `numpy 2.x` and `torch 2.10` as transitive
> dependencies. Remove them immediately after — the spack env provides numpy 1.26.4
> (required by torch 2.4.0) and the spack torch must take priority:
> ```bash
> rm -rf /nfs/data/1/xning/elec_AI/pypackages/numpy \
>        /nfs/data/1/xning/elec_AI/pypackages/numpy-*.dist-info \
>        /nfs/data/1/xning/elec_AI/pypackages/numpy.libs \
>        /nfs/data/1/xning/elec_AI/pypackages/torch \
>        /nfs/data/1/xning/elec_AI/pypackages/torch-*.dist-info \
>        /nfs/data/1/xning/elec_AI/pypackages/functorch \
>        /nfs/data/1/xning/elec_AI/pypackages/torchgen \
>        /nfs/data/1/xning/elec_AI/pypackages/nvidia \
>        /nfs/data/1/xning/elec_AI/pypackages/nvidia_*.dist-info \
>        /nfs/data/1/xning/elec_AI/pypackages/torchvision \
>        /nfs/data/1/xning/elec_AI/pypackages/torchvision-*.dist-info \
>        /nfs/data/1/xning/elec_AI/pypackages/torchvision.libs
> ```

Install remaining packages with `--no-deps` to avoid triggering torch re-downloads:
```bash
wcwc shell -e /nfs/data/1/xning/elec_AI/env -c "
    TMPDIR=/nfs/data/1/xning/tmp \
    pip install --target /nfs/data/1/xning/elec_AI/pypackages --no-deps \
        pretrainedmodels \
        'faiss-gpu-cu12' \
        timm
"
```

---

## Step 3 — Build torchvision from source

pip-distributed `torchvision` wheels ship their own C++ extensions compiled against a
specific torch binary. They are **not compatible** with the WCWC spack torch because the
two are compiled against different ABI variants. torchvision must be built from source
inside the activated spack env so it links against the correct torch.

Clone the tag that matches torch 2.4.0 (torchvision 0.19.x):
```bash
git clone --branch v0.19.1 --depth 1 \
    https://github.com/pytorch/vision \
    /nfs/data/1/xning/elec_AI/vision
```

Build and install into a project-local prefix:
```bash
wcwc shell -e /nfs/data/1/xning/elec_AI/env -c "
    TMPDIR=/nfs/data/1/xning/tmp \
    PYTHONPATH=/nfs/data/1/xning/elec_AI/pypackages:\$PYTHONPATH \
    cd /nfs/data/1/xning/elec_AI/vision && \
    python3 setup.py install \
        --prefix=/nfs/data/1/xning/elec_AI/vision_install
"
```

The installed egg will be at:
```
vision_install/lib/python3.11/site-packages/
    torchvision-0.19.1a0+6194369-py3.11-linux-x86_64.egg/
```

---

## Step 4 — Configure `.envrc` for auto-activation with direnv

Create `/nfs/data/1/xning/elec_AI/.envrc`:

```bash
source /wcwc/spack/share/spack/setup-env.sh
spack env activate /nfs/data/1/xning/elec_AI/env

# faiss-gpu-cu12 tries to preload libcudart.so.12 by finding nvidia-cuda-runtime-cu12
# metadata, but pip --target installs don't register in importlib.metadata properly.
# The spack env already provides CUDA libs on LD_LIBRARY_PATH, so disable the preload.
export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1

# torchvision was built from source against the spack torch — add its egg to PYTHONPATH
TORCHVISION_EGG=/nfs/data/1/xning/elec_AI/vision_install/lib/python3.11/site-packages/torchvision-0.19.1a0+6194369-py3.11-linux-x86_64.egg

# patchcore src + pip --target packages + torchvision egg
export PYTHONPATH="/nfs/data/1/xning/elec_AI/patchcore-inspection/src:${TORCHVISION_EGG}:/nfs/data/1/xning/elec_AI/pypackages${PYTHONPATH:+:${PYTHONPATH}}"
```

Allow it once (direnv requires explicit user approval):
```bash
direnv allow /nfs/data/1/xning/elec_AI
```

From this point, `cd /nfs/data/1/xning/elec_AI` automatically activates the full environment.

---

## Step 5 — Verify

```bash
cd /nfs/data/1/xning/elec_AI    # direnv loads the env

python3 - << 'EOF'
import torch
print("torch:          ", torch.__version__, " CUDA:", torch.cuda.is_available())
print("GPU:            ", torch.cuda.get_device_name(0))

import torchvision
print("torchvision:    ", torchvision.__version__)

import sklearn
print("scikit-learn:   ", sklearn.__version__)

import skimage
print("scikit-image:   ", skimage.__version__)

import pretrainedmodels
print("pretrainedmodels: OK")

import timm
print("timm:           ", timm.__version__)

import faiss
print("faiss:           OK  GPU support:", hasattr(faiss, "StandardGpuResources"))

import patchcore
import patchcore.backbones
import patchcore.patchcore
import patchcore.sampler
print("patchcore:       OK  (", patchcore.__file__, ")")

print()
print("=== ALL OK ===")
EOF
```

Expected output:
```
torch:           2.4.0  CUDA: True
GPU:             NVIDIA GeForce RTX 4090
torchvision:     0.19.1a0+6194369
scikit-learn:    1.8.0
scikit-image:    0.26.0
pretrainedmodels: OK
timm:            1.0.25
faiss:           OK  GPU support: True
patchcore:       OK  ( .../patchcore-inspection/src/patchcore/__init__.py )

=== ALL OK ===
```

---

## Quick-start: run PatchCore on MVTec AD

```bash
cd /nfs/data/1/xning/elec_AI    # activates env via direnv

datapath=/path_to_mvtec_folder/mvtec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
          'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush'
          'transistor' 'wood' 'zipper')
dataset_flags=($(for d in "${datasets[@]}"; do echo '-d '$d; done))

python patchcore-inspection/bin/run_patchcore.py \
    --gpu 0 --seed 0 --save_patchcore_model \
    --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
    --log_project MVTecAD_Results results \
    patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 3 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
```

---

## Gotchas & Troubleshooting

### torchvision C++ operator error
```
RuntimeError: operator torchvision::nms does not exist
```
The pip-installed torchvision binary is not compatible with the spack torch ABI.
**Fix:** build torchvision from source as described in Step 3.

### NumPy ABI mismatch warning
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
NumPy 2.x was installed into `pypackages/` as a transitive dependency.
**Fix:** remove it — spack provides NumPy 1.26.4, which is correct:
```bash
rm -rf pypackages/numpy pypackages/numpy-*.dist-info pypackages/numpy.libs
```

### faiss PackageNotFoundError: nvidia-cuda-runtime-cu12
```
importlib.metadata.PackageNotFoundError: No package metadata was found for nvidia-cuda-runtime-cu12
```
`faiss-gpu-cu12` uses `importlib.metadata` to locate `libcudart.so.12` before loading.
With `--target` installs, the `.dist-info` directories are not on the default metadata
search path. The spack env already provides the CUDA runtime on `LD_LIBRARY_PATH`.
**Fix:** the `.envrc` sets `_FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1` to skip the preload.

### patchcore ModuleNotFoundError
```
ModuleNotFoundError: No module named 'patchcore'
```
Editable installs created with `pip install --target` place a `.pth` file in the target
directory; `.pth` files are only processed for standard site-packages, not for directories
added via `PYTHONPATH`.
**Fix:** the `.envrc` adds `patchcore-inspection/src` directly to `PYTHONPATH`.

### pip download fails: No space left on device
`/tmp` has only ~1.7 GB — too small for large CUDA wheels.
**Fix:** `TMPDIR=/nfs/data/1/xning/tmp` (NFS has 1.7 TB free).

---

## Directory layout

```
elec_AI/
├── .envrc                        ← direnv config (auto-activates everything)
├── SETUP.md                      ← this file
├── env/                          ← Spack environment
│   ├── spack.yaml                ← package specs (py-torch@2.4.0, py-tqdm)
│   └── .spack-env/view/          ← symlink tree to all spack packages
├── patchcore-inspection/         ← PatchCore source (cloned repo)
│   ├── src/patchcore/            ← Python package (on PYTHONPATH)
│   ├── bin/                      ← run_patchcore.py, load_and_evaluate_patchcore.py
│   └── requirements.txt
├── pypackages/                   ← pip --target installs
│   ├── faiss/                    ← faiss-gpu-cu12
│   ├── sklearn/                  ← scikit-learn
│   ├── skimage/                  ← scikit-image
│   ├── pretrainedmodels/
│   └── timm/
├── vision/                       ← torchvision source (v0.19.1)
└── vision_install/               ← torchvision built from source
    └── lib/python3.11/site-packages/
        └── torchvision-0.19.1a0+*.egg/
```

---

## Environment variable reference

| Variable | Value | Why |
|----------|-------|-----|
| `PYTHONPATH` | `patchcore-inspection/src` | patchcore package (editable, .pth not processed) |
| `PYTHONPATH` | `torchvision-*.egg` | torchvision built from source |
| `PYTHONPATH` | `pypackages/` | faiss, scikit-*, pretrainedmodels, timm |
| `_FAISS_WHEEL_DISABLE_CUDA_PRELOAD` | `1` | Skip faiss metadata preload; spack CUDA is on `LD_LIBRARY_PATH` |

> **Key gotcha:** Never install `numpy>=2` into `pypackages/`. The spack torch 2.4.0 was
> compiled against NumPy 1.x — putting NumPy 2.x earlier on `PYTHONPATH` causes an ABI
> crash at import time. The spack view provides NumPy 1.26.4 and takes priority as long
> as `pypackages/` contains no NumPy.
