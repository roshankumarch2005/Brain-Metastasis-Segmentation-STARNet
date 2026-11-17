#!/bin/bash
set -e

# Default to GPU 0 if not specified (e.g. CUDA_VISIBLE_DEVICES=0)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

# On DGX adjust paths if running there:
if [ -d "/Data3/EE23B009/EE23B009_BTP" ]; then
    export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
    export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
    export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
    cd /Data3/EE23B009/EE23B009_BTP
fi

echo "=========================================================="
echo "🚀 Launching Method 3: 3D Depthwise Separable UNet"
echo "Dataset: 105 (UCSF Exact Binary Baseline - RAW + New Modalities)"
echo "Architecture: Base 16, Cap 256 + 3D Depthwise Separable Convolutions"
echo "Parameters: ~1.23 Million (96.0% reduction from 30.88M default!)"
echo "Trainer: nnUNetTrainer_DepthwiseSeparable3D"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_DepthwiseSeparable3D

