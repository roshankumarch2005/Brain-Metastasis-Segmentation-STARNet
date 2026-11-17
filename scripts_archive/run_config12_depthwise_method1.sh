#!/bin/bash
# run_config12_depthwise_method1.sh
# ----------------------------------------------------
# Configuration 12: Depthwise nnU-Net + Method 1 (IL-Hinge)
# Runs on GPU 0 with original hyperparameters (Tr = 0.20, lambda = 1.0)
# Dataset: 102 (Dataset102_Restored: BraTS Multiclass Subregions)

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

mkdir -p /mnt/DATA/EE23B009/EE23B009_BTP/training_logs
LOG_FILE="/mnt/DATA/EE23B009/EE23B009_BTP/training_logs/train_config12_depthwise_method1.log"

echo "=========================================================="
echo "🚀 Launching Configuration 12: Depthwise nnU-Net + Method 1"
echo "Dataset: 102 (Dataset102_Restored)"
echo "GPU: 0 (Isolated Local RTX 2080 Ti)"
echo "Trainer: nnUNetTrainer_Method3_InterLevelAll"
echo "Hyperparameters: Tr = 0.20, lambda = 1.0 (Original)"
echo "Log: $LOG_FILE"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Method3_InterLevelAll 2>&1 | tee -a "$LOG_FILE"

