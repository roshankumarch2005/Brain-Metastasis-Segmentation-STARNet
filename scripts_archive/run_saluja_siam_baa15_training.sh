#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "Saluja SIAM Attention + BAA 1.5 Training (Dataset 105, Fold 0)"
echo "GPU: 0 (Single GPU - Peak Throughput, Zero Sync Latency)"
echo "Trainer: nnUNetTrainerSIAM_BAA_Sensitive"
echo "Target: 4 Modalities (T1post, T1pre, Sub, FLAIR) -> Binary Enhancing"
echo "=========================================================="

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainerSIAM_BAA_Sensitive
