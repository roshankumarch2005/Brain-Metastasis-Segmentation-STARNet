#!/bin/bash
# launch_method3_siam_curriculum_multiclass_dgx.sh
# --------------------------------------------------
# Launches Synergistic Model: Method 3 (3D Depthwise-Separable, 1.49M params)
#                           + SIAM3D Spatial Attention
#                           + Two-Stage Curriculum BAA Loss (Model M8 engine)
# on Dataset102_Restored (Multiclass BraTS Segmentation: NCR, ED, ET).
#
# Target: High-efficiency multiclass segmentation on ultra-compact ~11 MB footprint.

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Default GPU is 7 (or 6), or respect existing CUDA_VISIBLE_DEVICES
GPU_ID=${CUDA_VISIBLE_DEVICES:-7}
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
echo "🚀 Launching Method 3 + SIAM + Curriculum BAA (Multiclass M8) on DGX"
echo "Dataset: 102 (Dataset102_Restored: BraTS Multiclass)"
echo "GPU: $GPU_ID (DGX A100)"
echo "Trainer: nnUNetTrainer_Method3_SIAM_CurriculumBAA"
echo "Architecture: 3D Depthwise-Separable PlainConvUNet (~1.49M params)"
echo "Loss: Two-Stage Curriculum BAA (Epoch 0-499: DC+CE -> 500-1000: BAA beta=1.5)"
echo "=========================================================="

cd /Data3/EE23B009/EE23B009_BTP

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_SIAM_CurriculumBAA "$@"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_SIAM_CurriculumBAA "$@"
else
    nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_SIAM_CurriculumBAA "$@"
fi
