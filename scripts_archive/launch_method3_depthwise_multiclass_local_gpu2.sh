#!/bin/bash
# launch_method3_depthwise_multiclass_local_gpu2.sh
# --------------------------------------------------
# Launches Method 3 (3D Depthwise Separable Convolutions) on Multiclass (Dataset 102)
# Runs locally on GPU 2.

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=2

echo "=========================================================="
echo "🚀 Launching Method 3 (3D Depthwise Separable) on Multiclass"
echo "Dataset: 102 (Dataset102_Restored: BraTS Multiclass Subregions)"
echo "GPU: 2 (Isolated Local RTX 2080 Ti)"
echo "Trainer: nnUNetTrainer_DepthwiseSeparable3D"
echo "Architecture: 3D Depthwise Separable Convolutions (-88% parameters)"
echo "=========================================================="

cd /mnt/DATA/EE23B009/EE23B009_BTP
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_DepthwiseSeparable3D

