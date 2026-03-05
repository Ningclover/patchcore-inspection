#!/bin/bash
# Inference script for PatchCore on a custom dataset.
# Usage: bash infer.sh

# ── User settings ────────────────────────────────────────────────────────────
DATAPATH=/nfs/data/1/xning/elec_AI/my_data   # root of dataset (contains Chip/, FEMB/, ...)
DATASET=Chip                                   # subfolder name inside DATAPATH
GPU=1                                          # GPU index to use
SEED=0

# Path to the trained model folder (produced by train.sh)
MODEL_PATH=/nfs/data/1/xning/elec_AI/results/MyData_Results/IM224_WR50_Chip_1/models/mvtec_Chip

# Output directory for per-image scores CSV (and optional heatmap images)
OUTPUT=/nfs/data/1/xning/elec_AI/results/evaluated_results/Chip_infer

# Set to "--save_segmentation_images" to also save anomaly heatmap PNGs, or leave empty
SAVE_IMAGES=""
# ─────────────────────────────────────────────────────────────────────────────

# Environment setup
source /wcwc/spack/share/spack/setup-env.sh
spack env activate /nfs/data/1/xning/elec_AI/env
export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1
TORCHVISION_EGG=/nfs/data/1/xning/elec_AI/vision_install/lib/python3.11/site-packages/torchvision-0.19.1a0+6194369-py3.11-linux-x86_64.egg
export PYTHONPATH="/nfs/data/1/xning/elec_AI/patchcore-inspection/src:${TORCHVISION_EGG}:/nfs/data/1/xning/elec_AI/pypackages${PYTHONPATH:+:${PYTHONPATH}}"

cd /nfs/data/1/xning/elec_AI/patchcore-inspection

python bin/load_and_evaluate_patchcore.py \
    --gpu $GPU \
    --seed $SEED \
    $SAVE_IMAGES \
    $OUTPUT \
    patch_core_loader \
        -p $MODEL_PATH \
        --faiss_on_gpu \
    dataset \
        --resize 256 \
        --imagesize 224 \
        -d $DATASET \
        mvtec $DATAPATH

echo ""
echo "Per-image scores:"
cat $OUTPUT/mvtec_${DATASET}_per_image_scores.csv
