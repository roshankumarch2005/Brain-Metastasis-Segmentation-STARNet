#!/bin/bash
# launch_dgx_binary_config10_depthwise_baa.sh
# ----------------------------------------------------
# Configuration 10 (Binary): Depthwise + BAA Sensitive on DGX A100
# Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor Binary GTV)
# Architecture: 3D Depthwise-Separable UNet (~1.23M params, No SIAM)
# Loss: Volume-Adaptive BAA Sensitive Loss (beta = 1.5)
# Target GPU: 4 (or $CUDA_VISIBLE_DEVICES)

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Default GPU is 4, or respect existing CUDA_VISIBLE_DEVICES
GPU_ID=${CUDA_VISIBLE_DEVICES:-4}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set PATH to include brats_env
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$DGX_DIR"

# Install trainer files into DGX python environment
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        mkdir -p "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses 2>/dev/null || true
        mkdir -p "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_Method3_BAA_Sensitive.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_Method3*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_DepthwiseSeparable3D.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
        cp "$DGX_DIR"/src/models/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models/ 2>/dev/null || true
    fi
done

echo "=========================================================="
echo "🚀 Launching Binary Config 10: Depthwise + BAA Sensitive (DGX A100)"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact - Binary SRS GTV)"
echo "GPU: $GPU_ID (DGX A100)"
echo "Trainer: nnUNetTrainer_Method3_BAA_Sensitive"
echo "Architecture: 3D Depthwise-Separable UNet (~1.23M params, No SIAM)"
echo "Loss: Volume-Adaptive BAA Sensitive Loss (beta = 1.5)"
echo "=========================================================="

mkdir -p "$DGX_DIR/training_logs"
LOG_FILE="$DGX_DIR/training_logs/train_dgx_binary_config10_depthwise_baa_gpu${GPU_ID}.log"

# Safe checkpoint detection: only pass --c if checkpoint_latest.pth actually exists
CP_FILE="$nnUNet_results/Dataset105_UCSF_Saluja_Exact/nnUNetTrainer_Method3_BAA_Sensitive__nnUNetPlans__3d_fullres/fold_0/checkpoint_latest.pth"
EXTRA_ARGS=()
if [ -f "$CP_FILE" ]; then
    echo "Found existing checkpoint at $CP_FILE. Resuming training with --c..."
    EXTRA_ARGS+=("--c")
else
    echo "No valid checkpoint found. Starting fresh training from Epoch 0..."
fi

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_BAA_Sensitive "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_BAA_Sensitive "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
else
    nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_BAA_Sensitive "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
fi

