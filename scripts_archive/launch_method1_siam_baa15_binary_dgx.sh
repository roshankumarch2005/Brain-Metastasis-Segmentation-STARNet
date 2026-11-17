#!/bin/bash
# launch_method1_siam_baa15_binary_dgx.sh
# ----------------------------------------
# Launches Model 2: Method 1 + SIAM + BAA Sensitive (beta_scale=1.5) on DGX
# Run this inside a tmux session on the DGX server.

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Isolated GPU (GPU 4 on DGX)
export CUDA_VISIBLE_DEVICES=4

# Ensure trainers and losses are installed into DGX brats_env
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp /Data3/EE23B009/EE23B009_BTP/src/nnUNetTrainer_InterLevel*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp /Data3/EE23B009/EE23B009_BTP/src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
    fi
done

echo "=========================================================="
echo "🚀 Launching Model 2: Method 1 + SIAM + BAA Sensitive (1.5) on DGX"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor GTV)"
echo "GPU: 4 (DGX A100)"
echo "Trainer: nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive"
echo "Loss: Smoothed BAA Sensitive (beta_scale=1.5, area_threshold=5000)"
echo "Inter-Level Regularization: Tr=0.20 anchored to d0"
echo "SIAM Attention: Active on all encoder skips and bottleneck"
echo "=========================================================="

cd /Data3/EE23B009/EE23B009_BTP

# Locate nnUNetv2_train binary
if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive "$@"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive "$@"
elif [ -f "$HOME/brats_env/bin/nnUNetv2_train" ]; then
    $HOME/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive "$@"
else
    nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive "$@"
fi
