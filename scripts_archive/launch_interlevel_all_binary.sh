#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "🚀 Launching Method 1 (All Stages Anchored to d0)"
echo "Dataset: 105 (UCSF Exact Binary Baseline)"
echo "GPU: 0 (Isolated GPU)"
echo "Trainer: nnUNetTrainer_InterLevelAll"
echo "Threshold Tr: 0.20 | Inter-loss weight: 1.0"
echo "All 5 differences active: d1-d0, d2-d0, d3-d0, d4-d0, d5_bottleneck-d0"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevelAll

