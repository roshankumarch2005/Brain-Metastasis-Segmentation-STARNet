#!/bin/bash
# launch_dgx_binary_config14_depthwise_method1_curriculum_baa.sh
# ----------------------------------------------------------------
# Configuration 14 (Binary): Depthwise + Method 1 + Curriculum BAA on DGX A100
# Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor Binary GTV)
# Architecture: 3D Depthwise-Separable UNet (~1.23M params)
# Regularization: Method 1 Inter-Level Hinge Discrepancy (Tr = 0.20, lambda = 1.0)
# Loss: Two-Stage Curriculum BAA (beta = 1.5)

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Default GPU is 7, or respect existing CUDA_VISIBLE_DEVICES
GPU_ID=${CUDA_VISIBLE_DEVICES:-4}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set PATH to include brats_env
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$DGX_DIR"

# Install trainer files into DGX python environment
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp "$DGX_DIR"/src/nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_Method3*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_DepthwiseSeparable3D.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/nnUNetTrainer_InterLevel*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp "$DGX_DIR"/src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
        cp "$DGX_DIR"/src/models/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models/ 2>/dev/null || true
    fi
done

echo "=========================================================="
echo "🚀 Launching Binary Config 14: Depthwise + Method 1 + Curriculum BAA (DGX A100)"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact - Binary SRS GTV)"
echo "GPU: $GPU_ID (DGX A100)"
echo "Trainer: nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA"
echo "Architecture: 3D Depthwise-Separable UNet (~1.23M params)"
echo "Regularization: Method 1 Inter-Level Hinge (Tr = 0.20, lambda = 1.0)"
echo "Loss: Two-Stage Curriculum BAA (beta = 1.5)"
echo "=========================================================="

mkdir -p "$DGX_DIR/training_logs"
LOG_FILE="$DGX_DIR/training_logs/train_dgx_binary_config14_depthwise_method1_curriculum_baa_gpu${GPU_ID}.log"

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA "$@" 2>&1 | tee -a "$LOG_FILE"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA "$@" 2>&1 | tee -a "$LOG_FILE"
else
    nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA "$@" 2>&1 | tee -a "$LOG_FILE"
fi

