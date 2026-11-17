#!/bin/bash
# launch_dgx_config13_depthwise_method1_siam.sh
# ----------------------------------------------------
# Configuration 13: Depthwise + Method 1 + SIAM on DGX
# Runs on GPU 1 (or $CUDA_VISIBLE_DEVICES) on DGX A100.
# Dataset: 102 (Dataset102_Restored: BraTS Multiclass Subregions)

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Default to GPU 1
GPU_ID=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set PATH to include brats_env
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$DGX_DIR"

# Install trainer into DGX environment
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp "$DGX_DIR"/src/nnUNetTrainer_Method3_InterLevel_SIAM.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_Method3*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_DepthwiseSeparable3D.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_InterLevel*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
        cp "$DGX_DIR"/src/models/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models/ 2>/dev/null || true
    fi
done

echo "=========================================================="
echo "🚀 Launching Configuration 13: Depthwise + Method 1 + SIAM (DGX A100)"
echo "Dataset: 102 (Dataset102_Restored)"
echo "GPU: $GPU_ID (DGX A100)"
echo "Trainer: nnUNetTrainer_Method3_InterLevel_SIAM"
echo "Architecture: 3D Depthwise-Separable + SIAM3D Attention (~1.49M params)"
echo "Regularization: Method 1 Inter-Level Hinge (Tr = 0.20, lambda = 1.0)"
echo "=========================================================="

mkdir -p "$DGX_DIR/training_logs"
LOG_FILE="$DGX_DIR/training_logs/train_dgx_config13_depthwise_method1_siam_gpu${GPU_ID}.log"

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevel_SIAM "$@" 2>&1 | tee -a "$LOG_FILE"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevel_SIAM "$@" 2>&1 | tee -a "$LOG_FILE"
else
    nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevel_SIAM "$@" 2>&1 | tee -a "$LOG_FILE"
fi

