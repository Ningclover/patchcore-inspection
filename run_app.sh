#!/bin/bash
source /wcwc/spack/share/spack/setup-env.sh
spack env activate /nfs/data/1/xning/elec_AI/env
export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1
export GRADIO_TEMP_DIR=/nfs/data/1/xning/tmp/gradio
TORCHVISION_EGG=/nfs/data/1/xning/elec_AI/vision_install/lib/python3.11/site-packages/torchvision-0.19.1a0+6194369-py3.11-linux-x86_64.egg
export PYTHONPATH="/nfs/data/1/xning/elec_AI/patchcore-inspection/src:${TORCHVISION_EGG}:/nfs/data/1/xning/elec_AI/pypackages${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p /nfs/data/1/xning/tmp/gradio
cd /nfs/data/1/xning/elec_AI/patchcore-inspection
exec python3 app.py
