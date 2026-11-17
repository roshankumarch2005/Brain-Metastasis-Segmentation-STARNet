#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

mkdir -p /mnt/DATA/EE23B009/EE23B009_BTP/training_logs
LOG_FILE="/mnt/DATA/EE23B009/EE23B009_BTP/training_logs/train_binary_flagship_m3_il_siam_curr_baa.log"

echo "=========================================================="
echo "🚀 Launching Model B13: Ultimate Unified Edge Binary Flagship"
echo "Stack: 3D Depthwise-Separable + IL-Hinge (Method 1) + SIAM + Curriculum BAA"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor SRS GTV)"
echo "GPU: 0 (Isolated Local RTX 2080 Ti)"
echo "Trainer: nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA"
echo "Parameters: ~1.23M (-96% reduction vs 31.2M baseline)"
echo "Log file: $LOG_FILE"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA --c 2>&1 | tee -a "$LOG_FILE"

