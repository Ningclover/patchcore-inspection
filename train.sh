#!/bin/bash
# Training script for PatchCore on a custom dataset.
# Usage: bash train.sh

# ── User settings ────────────────────────────────────────────────────────────
DATAPATH=/nfs/data/1/xning/elec_AI/my_data   # root of dataset (contains Chip/, FEMB/, ...)
DATASET=LarASIC                                   # subfolder name inside DATAPATH
GPU=0                                          # GPU index to use
SEED=0

# Model / feature settings
BACKBONE=wideresnet50
LAYERS="layer2 layer3"
PRETRAIN_DIM=1024
TARGET_DIM=384
PATCHSIZE=3
NUM_NN=1
CORESET_PCT=0.1      # fraction of patches to keep (0.1 = 10%, 0.01 = 1%)

# Output
RESULTS=/nfs/data/1/xning/elec_AI/results
LOG_PROJECT=MyData_Results
LOG_GROUP=IM224_WR50_${DATASET}
# ─────────────────────────────────────────────────────────────────────────────

# Environment setup
source /wcwc/spack/share/spack/setup-env.sh
spack env activate /nfs/data/1/xning/elec_AI/env
export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1
TORCHVISION_EGG=/nfs/data/1/xning/elec_AI/vision_install/lib/python3.11/site-packages/torchvision-0.19.1a0+6194369-py3.11-linux-x86_64.egg
export PYTHONPATH="/nfs/data/1/xning/elec_AI/patchcore-inspection/src:${TORCHVISION_EGG}:/nfs/data/1/xning/elec_AI/pypackages${PYTHONPATH:+:${PYTHONPATH}}"

# Build -le flags from LAYERS
LAYER_FLAGS=$(for l in $LAYERS; do echo "-le $l"; done)

cd /nfs/data/1/xning/elec_AI/patchcore-inspection

python bin/run_patchcore.py \
    --gpu $GPU \
    --seed $SEED \
    --save_patchcore_model \
    --log_group $LOG_GROUP \
    --log_project $LOG_PROJECT \
    $RESULTS \
    patch_core \
        -b $BACKBONE \
        $LAYER_FLAGS \
        --faiss_on_gpu \
        --pretrain_embed_dimension $PRETRAIN_DIM \
        --target_embed_dimension $TARGET_DIM \
        --anomaly_scorer_num_nn $NUM_NN \
        --patchsize $PATCHSIZE \
    sampler -p $CORESET_PCT approx_greedy_coreset \
    dataset \
        --resize 512 \
        --imagesize 512 \
        -d $DATASET \
        mvtec $DATAPATH
