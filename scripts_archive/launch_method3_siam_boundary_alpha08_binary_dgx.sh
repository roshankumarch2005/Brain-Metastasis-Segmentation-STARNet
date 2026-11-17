#!/bin/bash
# launch_method3_siam_boundary_alpha08_binary_dgx.sh
# ---------------------------------------------------
# Launches Synergistic Model: Method 3 (3D Depthwise-Separable, 3.8M params)
#                           + SIAM3D Spatial Attention
#                           + Convex 80/20 Boundary-Aware Loss (Model B3 engine)
# on Dataset105_UCSF_Saluja_Exact (Binary GTV Segmentation).
#
# Target: Ultra-compact footprint (~11 MB) with superior sub-millimeter boundary precision.

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Default GPU is 6, or respect existing CUDA_VISIBLE_DEVICES
GPU_ID=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set PATH to include brats_env
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

# Ensure trainers and losses are installed into DGX brats_env
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp /Data3/EE23B009/EE23B009_BTP/src/nnUNetTrainer_Method3*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp /Data3/EE23B009/EE23B009_BTP/src/nnUNetTrainer_DepthwiseSeparable3D.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp /Data3/EE23B009/EE23B009_BTP/src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
        cp /Data3/EE23B009/EE23B009_BTP/src/models/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models/ 2>/dev/null || true
    fi
done

echo "=========================================================="
echo "🚀 Launching Method 3 + SIAM + Boundary Alpha 0.8 (Binary B3) on DGX"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor GTV)"
echo "GPU: $GPU_ID (DGX A100)"
echo "Trainer: nnUNetTrainer_Method3_SIAM_Boundary_Alpha08"
echo "Architecture: 3D Depthwise-Separable PlainConvUNet (~3.8M params)"
echo "Loss: Convex 80/20 (Boundary-Aware EDT + DC)"
echo "=========================================================="

cd /Data3/EE23B009/EE23B009_BTP

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_SIAM_Boundary_Alpha08 "$@"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_SIAM_Boundary_Alpha08 "$@"
else
    nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_SIAM_Boundary_Alpha08 "$@"
fi
