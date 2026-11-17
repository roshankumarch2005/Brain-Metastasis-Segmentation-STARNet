#!/bin/bash
# run_predict_decoder1_halfres_multiclass.sh
# ------------------------------------------
# Runs inference for Method 2: Decoder-1 Half-Res Bypass (Multiclass Dataset 102)
# on Local GPU 0, followed by test set metric evaluation.

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

PRED_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/predictions_decoder1_halfres_multiclass_final"
mkdir -p "$PRED_DIR"

echo "=========================================================="
echo "🚀 Running Method 2 Decoder-1 Half-Res Multiclass Inference"
echo "Dataset: 102 (Dataset102_Restored - 30 Unseen Test Scans)"
echo "Model: nnUNetTrainer_Decoder1_HalfRes"
echo "Output: $PRED_DIR"
echo "GPU: 0"
echo "=========================================================="

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_predict \
 -i /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTs \
 -o "$PRED_DIR" \
 -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Decoder1_HalfRes -chk checkpoint_final.pth

echo "Inference complete! Predictions saved in $PRED_DIR"

