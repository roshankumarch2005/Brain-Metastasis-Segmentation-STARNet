#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "Saluja Exact Baseline Training (Dataset 105, Fold 0)"
echo "GPU: 0 (Single GPU - Peak Throughput, Zero Sync Latency)"
echo "Trainer: Standard nnUNetTrainer"
echo "=========================================================="

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0
