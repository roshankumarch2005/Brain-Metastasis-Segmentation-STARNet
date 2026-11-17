#!/bin/bash
set -e

# Method 2: Decoder-1 Half-Res Bypass on Multiclass Brain Tumor Dataset (Dataset 102)
# Using 1 Local GPU (GPU 0) with Batch Size 1 (80 x 192 x 160 patch)

export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export PYTHONPATH="/mnt/DATA/EE23B009/EE23B009_BTP/src:$PYTHONPATH"

cd /mnt/DATA/EE23B009/EE23B009_BTP

echo "======================================================================"
echo "🚀 Launching Method 2: Decoder 1 (Half-Res) on Dataset 102 (Multiclass)"
echo "Dataset: 102 (Restored Brain Tumor Subregions: WT, TC, ET)"
echo "Trainer: nnUNetTrainer_Decoder1_HalfRes"
echo "Configuration: 3d_fullres (Batch Size: 1, Patch: [80, 192, 160])"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (Local RTX 2080 Ti)"
echo "Timestamp: $(date)"
echo "======================================================================"

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_Decoder1_HalfRes

