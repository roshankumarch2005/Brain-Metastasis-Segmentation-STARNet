#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

mkdir -p /mnt/DATA/EE23B009/EE23B009_BTP/training_logs
LOG_FILE="/mnt/DATA/EE23B009/EE23B009_BTP/training_logs/train_binary_config2_siam_gpu0.log"

echo "=========================================================="
echo "🚀 Launching Binary Configuration 2: Standalone SIAM Attention"
echo "Architecture: Standard PlainConvUNet 3D + Spatial-Channel Attention (SIAM)"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact: Enhancing Tumor SRS GTV)"
echo "GPU: 0 (Isolated Local RTX 2080 Ti)"
echo "Trainer: nnUNetTrainerSIAM"
echo "Parameters: ~31.2M"
echo "Log file: $LOG_FILE"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainerSIAM 2>&1 | tee -a "$LOG_FILE"
