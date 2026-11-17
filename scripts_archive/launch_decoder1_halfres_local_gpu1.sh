#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=1

echo "=========================================================="
echo "🚀 Resuming Method 2: Decoder 1 (H/2 x W/2) on Local GPU 1"
echo "Dataset: 105 (UCSF Exact Binary Baseline - RAW + Subtraction)"
echo "Trainer: nnUNetTrainer_Decoder1_HalfRes"
echo "Batch Size: 1 (Local plans)"
echo "GPU: 1 (Dedicated Local GPU)"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Decoder1_HalfRes --c

