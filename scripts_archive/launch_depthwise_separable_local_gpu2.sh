#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=2

echo "=========================================================="
echo "🚀 Resuming Method 3: 3D Depthwise Separable on Local GPU 2"
echo "Dataset: 105 (UCSF Exact Binary Baseline - RAW + Subtraction)"
echo "Trainer: nnUNetTrainer_DepthwiseSeparable3D"
echo "Parameters: ~1.23 Million (96% parameter reduction)"
echo "Batch Size: 1 (Local plans)"
echo "GPU: 2 (Dedicated Local GPU)"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_DepthwiseSeparable3D --c

