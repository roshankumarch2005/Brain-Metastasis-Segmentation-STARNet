#!/bin/bash
# run_predict_depthwise_multiclass.sh
# -----------------------------------
# Runs inference for Method 3: 3D Depthwise Separable (Multiclass Dataset 102)
# on Local GPU 2, followed by test set metric evaluation.

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=2

PRED_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/predictions_depthwise_separable_multiclass_final"
mkdir -p "$PRED_DIR"

echo "=========================================================="
echo "🚀 Running 3D Depthwise Separable Multiclass Test Inference"
echo "Dataset: 102 (Dataset102_Restored - 30 Unseen Test Scans)"
echo "Model: nnUNetTrainer_DepthwiseSeparable3D (1.23M params)"
echo "Output: $PRED_DIR"
echo "GPU: 2"
echo "=========================================================="

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_predict \
 -i /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTs \
 -o "$PRED_DIR" \
 -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_DepthwiseSeparable3D -chk checkpoint_final.pth

echo "Inference complete! Now running comprehensive test evaluation..."
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python3 /mnt/DATA/EE23B009/EE23B009_BTP/evaluate_depthwise_multiclass_test30.py "$PRED_DIR"

echo "ALL DONE!"

