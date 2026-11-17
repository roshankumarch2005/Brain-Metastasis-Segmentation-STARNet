#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "🚀 Launching Method 1 (Multi-Stage) + SIAM on Multiclass"
echo "Dataset: 102 (Dataset102_Restored: Core, Edema, Enhancing)"
echo "GPU: 0 (Isolated Local GPU)"
echo "Trainer: nnUNetTrainer_InterLevel_SIAM"
echo "Threshold Tr: 0.20 | Inter-loss weight: 1.0"
echo "SIAM Attention: Active on all encoder skips and bottleneck"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM

