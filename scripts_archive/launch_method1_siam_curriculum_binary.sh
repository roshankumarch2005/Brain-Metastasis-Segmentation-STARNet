#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=3

echo "=========================================================="
echo "🚀 Launching Model 3: Method 1 + SIAM + Two-Stage Curriculum BAA on Binary"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor GTV)"
echo "GPU: 3 (Isolated Local RTX 2080 Ti)"
echo "Trainer: nnUNetTrainer_InterLevel_SIAM_CurriculumBAA"
echo "Curriculum: Stage 1 (0-499) DC+CE -> Stage 2 (500-1000) BAA Sensitive 1.5"
echo "Inter-Level Regularization: Tr = 0.20 anchored to d0 across all 5 deeper levels"
echo "SIAM Attention: Active on all encoder skips and bottleneck"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_InterLevel_SIAM_CurriculumBAA

