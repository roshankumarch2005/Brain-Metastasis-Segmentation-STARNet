#!/bin/bash
# launch_method1_boundary_binary_dgx_gpu6.sh
# -------------------------------------------
# Launches Model: Method 1 + SIAM + Boundary Alpha 0.8 (Convex 80/20) on DGX GPU 6
# Resumes from Epoch 521 smoothly to reach 1000 epochs in ~7.5 hours.

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# Isolated GPU 6 on DGX (A100-SXM4-80GB)
export CUDA_VISIBLE_DEVICES=6

# Set PATH to include brats_env
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

# Ensure trainers and losses are installed into DGX brats_env
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp /Data3/EE23B009/EE23B009_BTP/src/nnUNetTrainer_InterLevel*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp /Data3/EE23B009/EE23B009_BTP/src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
    fi
done

echo "=========================================================="
echo "🚀 Resuming Method 1 + SIAM + Boundary Alpha 0.8 on DGX GPU 6"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor GTV)"
echo "GPU: 6 (DGX A100)"
echo "Trainer: nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08"
echo "Resuming from: Epoch 521 -> Target: 1000 Epochs"
echo "=========================================================="

cd /Data3/EE23B009/EE23B009_BTP

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08 --c "$@"
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08 --c "$@"
else
    nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08 --c "$@"
fi
