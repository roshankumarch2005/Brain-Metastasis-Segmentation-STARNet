#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "🚀 Launching Model 1: Method 1 + SIAM + Convex 80/20 Boundary Loss on Binary"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor GTV)"
echo "GPU: 0 (Isolated Local RTX 2080 Ti)"
echo "Trainer: nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08"
echo "Boundary Loss: Convex 80% Dice / 20% CE (Boundary scale = 1.5)"
echo "Inter-Level Regularization: Tr = 0.20 anchored to d0 across all 5 deeper levels"
echo "SIAM Attention: Active on all encoder skips and bottleneck"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08
