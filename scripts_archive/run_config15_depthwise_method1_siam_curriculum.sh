#!/bin/bash
# run_config15_depthwise_method1_siam_curriculum.sh
# ----------------------------------------------------
# Configuration 15: Depthwise + Method 1 + SIAM + Curriculum BAA
# Runs on GPU 1 with original hyperparameters (Tr = 0.20, lambda = 1.0, Curriculum switch at epoch 500)
# Dataset: 102 (Dataset102_Restored: BraTS Multiclass Subregions)

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=1

mkdir -p /mnt/DATA/EE23B009/EE23B009_BTP/training_logs
LOG_FILE="/mnt/DATA/EE23B009/EE23B009_BTP/training_logs/train_config15_depthwise_method1_siam_curriculum.log"

echo "=========================================================="
echo "🚀 Launching Configuration 15: Depthwise + Method 1 + SIAM + Curriculum BAA"
echo "Dataset: 102 (Dataset102_Restored)"
echo "GPU: 1 (Isolated Local RTX 2080 Ti)"
echo "Trainer: nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA"
echo "Hyperparameters: Tr = 0.20, lambda = 1.0 (Original), Curriculum 500"
echo "Log: $LOG_FILE"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA 2>&1 | tee -a "$LOG_FILE"

