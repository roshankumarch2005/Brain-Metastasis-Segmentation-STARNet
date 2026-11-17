#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "🚀 Launching Method 1 Master (All Stages Anchored to d0)"
echo "Dataset: 102 (Multiclass UCSF-BMSR Restored: Core, Edema, Enhancing)"
echo "GPU: 0 (Isolated Local GPU)"
echo "Trainer: nnUNetTrainer_InterLevelAll"
echo "Threshold Tr: 0.20 | Inter-loss weight: 1.0"
echo "Active Anchor Stages: d1-d0, d2-d0, d3-d0, d4-d0, d5_bottleneck-d0"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_InterLevelAll --c

