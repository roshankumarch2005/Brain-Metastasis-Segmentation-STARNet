#!/bin/bash
set -e

# Default to GPU 4 on DGX if not specified
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=4
fi

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# On DGX adjust paths:
if [ -d "/Data3/EE23B009/EE23B009_BTP" ]; then
    export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
    export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
    export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
    cd /Data3/EE23B009/EE23B009_BTP
fi

echo "=========================================================="
echo "🚀 Launching Method 2: Decoder 1 (H/2 x W/2) Primary Segmentation"
echo "Dataset: 105 (UCSF Exact Binary Baseline - RAW + New Modalities)"
echo "Label Strategy: 3D Max-Pooling (kernel=2, stride=2) - 0 Lesions Lost"
echo "Primary Prediction: Decoder 1 (Half Resolution Upsampled via Trilinear)"
echo "Trainer: nnUNetTrainer_Decoder1_HalfRes"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_train" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Decoder1_HalfRes
else
    /mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_Decoder1_HalfRes
fi

