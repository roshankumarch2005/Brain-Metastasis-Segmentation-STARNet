#!/bin/bash
# run_predict_eval_method1_boundary_binary.sh
# --------------------------------------------
# Evaluates Model: Method 1 + SIAM + Boundary Alpha 0.8 (Convex 80/20)
# on the 30 unseen test scans of Dataset105_UCSF_Saluja_Exact using Local GPU 1.

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=1

MODEL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08__nnUNetPlans__3d_fullres/fold_0"
PRED_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/predictions_interlevel_siam_boundary_alpha08_final"
mkdir -p "$PRED_DIR"

echo "=========================================================="
echo "🚀 Running Test Inference: Method 1 + SIAM + Boundary Alpha 0.8"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact - 30 Unseen Test Scans)"
echo "Trainer: nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08"
echo "Model Dir: $MODEL_DIR"
echo "Output: $PRED_DIR"
echo "GPU: 1"
echo "=========================================================="

CHK="checkpoint_final.pth"
if [ ! -f "$MODEL_DIR/$CHK" ]; then
    echo "checkpoint_final.pth not found, trying checkpoint_best.pth..."
    CHK="checkpoint_best.pth"
fi

if [ ! -f "$MODEL_DIR/$CHK" ]; then
    echo "❌ Error: Neither checkpoint_final.pth nor checkpoint_best.pth exists in $MODEL_DIR"
    exit 1
fi

echo "Using checkpoint: $CHK"

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_predict \
 -i /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/imagesTs \
 -o "$PRED_DIR" \
 -d 105 -c 3d_fullres -f 0 -tr nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08 -chk "$CHK"

echo "=========================================================="
echo "🎯 Inference Complete! Running Official 30 Test Scan Evaluation..."
echo "=========================================================="
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python3 /mnt/DATA/EE23B009/EE23B009_BTP/evaluate_saluja_test30.py "$PRED_DIR" | tee /mnt/DATA/EE23B009/EE23B009_BTP/interlevel_siam_boundary_alpha08_test_metrics.txt

echo "=========================================================="
echo "✅ ALL EVALUATION COMPLETE!"
echo "Results saved to: /mnt/DATA/EE23B009/EE23B009_BTP/interlevel_siam_boundary_alpha08_test_metrics.txt"
echo "=========================================================="

