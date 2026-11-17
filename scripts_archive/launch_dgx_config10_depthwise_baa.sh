#!/bin/bash
# launch_dgx_config10_depthwise_baa.sh
# ----------------------------------------------------
# Configuration 10: Depthwise + BAA Sensitive on DGX
# Runs on GPU 2 (or $CUDA_VISIBLE_DEVICES) on DGX A100.
# Dataset: 102 (Dataset102_Restored: BraTS Multiclass Subregions)

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Default to GPU 2
GPU_ID=${CUDA_VISIBLE_DEVICES:-2}
export CUDA_VISIBLE_DEVICES=$GPU_ID

export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$DGX_DIR"

# Install trainer into DGX environment
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp "$DGX_DIR"/src/nnUNetTrainer_Method3_BAA_Sensitive.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
    fi
done

echo "=========================================================="
echo "🚀 Launching Configuration 10: Depthwise + BAA Sensitive (DGX A100)"
echo "Dataset: 102 (Dataset102_Restored)"
echo "GPU: $GPU_ID (DGX A100)"
echo "Trainer: nnUNetTrainer_Method3_BAA_Sensitive"
echo "Architecture: 3D Depthwise-Separable (~1.23M params, No SIAM)"
echo "Loss: Volume-Adaptive BAA Sensitive Loss (beta = 1.5)"
echo "=========================================================="

mkdir -p "$DGX_DIR/training_logs"
LOG_FILE="$DGX_DIR/training_logs/train_dgx_config10_depthwise_baa_gpu${GPU_ID}.log"

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_BAA_Sensitive "$@" 2>&1 | tee -a "$LOG_FILE"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_BAA_Sensitive "$@" 2>&1 | tee -a "$LOG_FILE"
else
    nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_BAA_Sensitive "$@" 2>&1 | tee -a "$LOG_FILE"
fi

